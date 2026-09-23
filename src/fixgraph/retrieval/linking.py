"""Entity linking: question -> seed KG nodes (spec §10.1).

1. Mention extraction with the local LLM (structured, cached) plus rule-based ontology matches
   (products, OS versions, error codes) that need no model.
2. Each mention is matched against the node-name index (dense, label-filtered); matches above a
   similarity threshold become seeds weighted by similarity.
3. Node specificity (HippoRAG): weights are divided by log(1 + #chunks mentioning the node) so
   hub nodes like "iPhone" do not dominate the personalization vector.

The LLM stage runs separately from retrieval (GPU memory rule, spec §5.4): `extract_mentions`
is called for all questions first, then retrieval consumes the saved mentions.
"""

import math
import re

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from fixgraph.core.ontology import Ontology
from fixgraph.embeddings import Embedder
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured
from fixgraph.retrieval.index import NODES

_ERROR_RE = re.compile(r"\berror\s*(?:code\s*)?(-?\d{1,6})\b", re.IGNORECASE)


class Mentions(BaseModel):
    products: list[str] = Field(default_factory=lambda: list[str](), max_length=4)
    symptoms: list[str] = Field(default_factory=lambda: list[str](), max_length=3)
    components: list[str] = Field(default_factory=lambda: list[str](), max_length=3)
    features: list[str] = Field(default_factory=lambda: list[str](), max_length=3)
    error_codes: list[str] = Field(default_factory=lambda: list[str](), max_length=2)
    os_versions: list[str] = Field(default_factory=lambda: list[str](), max_length=3)


_PROMPT = """Extract what this Apple support question mentions. Copy short phrases from the
question; use empty lists when absent.
products: Apple devices (iPhone, Apple Watch, AirPods Pro, Mac...).
symptoms: the problem, as a short phrase ("won't pair", "battery drains fast").
components: hardware parts (Bluetooth, battery, Wi-Fi, camera).
features: software features/services (iCloud Photos, Find My, Siri).
error_codes: numbered errors ("error 4013").
os_versions: OS versions ("iOS 26", "macOS Tahoe").

Question: {question}"""


def build_mention_request(question: str, model: str) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=[ChatMessage(role="user", content=_PROMPT.format(question=question))],
        temperature=0.0,
        num_ctx=2048,
        max_tokens=250,
        think=False,
        json_schema=Mentions.model_json_schema(),
    )


def extract_mentions(client: LLMClient, question: str, model: str) -> Mentions:
    try:
        return complete_structured(client, build_mention_request(question, model), Mentions)
    except (StructuredOutputError, RuntimeError):
        return Mentions()


def rule_mentions(question: str, ontology: Ontology) -> Mentions:
    """Model-free mentions from the ontology (always merged in)."""
    products = [name for _, name in ontology.products_in(question)] or ontology.families_in(
        question
    )
    return Mentions(
        products=products[:4],
        components=ontology.components_in(question)[:3],
        features=ontology.features_in(question)[:3],
        error_codes=[f"error {c}" for c in _ERROR_RE.findall(question)][:2],
        os_versions=ontology.os_versions_in(question)[:3],
    )


def merge_mentions(a: Mentions, b: Mentions) -> Mentions:
    def m(x: list[str], y: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for s in x + y:
            if s.strip():
                seen.setdefault(s.strip(), None)
        return list(seen)

    return Mentions.model_construct(
        products=m(a.products, b.products),
        symptoms=m(a.symptoms, b.symptoms),
        components=m(a.components, b.components),
        features=m(a.features, b.features),
        error_codes=m(a.error_codes, b.error_codes),
        os_versions=m(a.os_versions, b.os_versions),
    )


_LABELS = {
    "products": ["Product", "ProductFamily"],
    "symptoms": ["Symptom"],
    "components": ["Component"],
    "features": ["Feature"],
    "error_codes": ["ErrorCode"],
    "os_versions": ["OSVersion"],
}


class Seed(BaseModel):
    node_id: str
    weight: float
    mention: str
    similarity: float


class EntityLinker:
    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        node_chunk_counts: dict[str, int],
        threshold: float = 0.72,
        per_mention: int = 3,
        question_symptoms: int = 5,
    ) -> None:
        self.client, self.embedder = client, embedder
        self.node_chunk_counts = node_chunk_counts
        self.threshold, self.per_mention = threshold, per_mention
        self.question_symptoms = question_symptoms

    def _search(self, text: str, labels: list[str], limit: int) -> list[tuple[str, float]]:
        vec = self.embedder.encode([text], query=False)[0].tolist()
        res = self.client.query_points(
            NODES,
            query=vec,
            using="dense",
            limit=limit * 3,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(key="label", match=qm.MatchAny(any=labels))]
            ),
            with_payload=True,
        )
        best: dict[str, float] = {}
        for p in res.points:
            nid = str((p.payload or {})["node_id"])
            best[nid] = max(best.get(nid, 0.0), float(p.score))
        return sorted(best.items(), key=lambda x: -x[1])[:limit]

    def specificity(self, node_id: str) -> float:
        return 1.0 / math.log(2 + self.node_chunk_counts.get(node_id, 0))

    def link(self, question: str, mentions: Mentions) -> list[Seed]:
        seeds: dict[str, Seed] = {}

        def add(nid: str, sim: float, mention: str) -> None:
            w = sim * self.specificity(nid)
            if nid not in seeds or seeds[nid].weight < w:
                seeds[nid] = Seed(node_id=nid, weight=w, mention=mention, similarity=sim)

        for field, labels in _LABELS.items():
            for mention in getattr(mentions, field):
                for nid, sim in self._search(mention, labels, self.per_mention):
                    if sim >= self.threshold:
                        add(nid, sim, mention)
        # The whole question against problem-like nodes catches symptoms the LLM paraphrased.
        for nid, sim in self._search(question, ["Symptom", "ErrorCode"], self.question_symptoms):
            if sim >= self.threshold:
                add(nid, sim * 0.8, "<question>")
        return sorted(seeds.values(), key=lambda s: -s.weight)
