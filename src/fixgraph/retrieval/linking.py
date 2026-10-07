"""Entity linking: question -> seed KG nodes (spec §10.1).

1. Mention extraction with the local LLM (structured, cached) plus rule-based ontology matches
   (products, OS versions, error codes) that need no model.
2. Each mention is matched against the node-name index (dense, label-filtered); matches above a
   similarity threshold become seeds weighted by similarity.
3. Node specificity (HippoRAG): weights are divided by log(1 + #chunks mentioning the node) so
   hub nodes like "iPhone" do not dominate the personalization vector.

The LLM stage runs separately from retrieval (GPU memory rule, spec §5.4): `extract_mentions`
is called for all questions first, then retrieval consumes the saved mentions.

Used by: bench.run (mentions stage), bench/cli.py (EntityLinker, rule mentions),
retrieval.graphrag (seeds). Uses: retrieval.index (node collection), embeddings, core.ontology,
llm.structured.
"""

# Imports: Qdrant for the node-name index, the ontology for rule matches, and the LLM helpers.
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

# Regex that finds error codes like "error 4013" or "error code -36" in the question.
_ERROR_RE = re.compile(r"\berror\s*(?:code\s*)?(-?\d{1,6})\b", re.IGNORECASE)


# The phrases pulled out of a question, grouped by entity type, with small caps per type.
class Mentions(BaseModel):
    products: list[str] = Field(default_factory=lambda: list[str](), max_length=4)
    symptoms: list[str] = Field(default_factory=lambda: list[str](), max_length=3)
    components: list[str] = Field(default_factory=lambda: list[str](), max_length=3)
    features: list[str] = Field(default_factory=lambda: list[str](), max_length=3)
    error_codes: list[str] = Field(default_factory=lambda: list[str](), max_length=2)
    os_versions: list[str] = Field(default_factory=lambda: list[str](), max_length=3)


# Prompt asking the LLM to copy short phrases from the question into each mention list.
_PROMPT = """Extract what this Apple support question mentions. Copy short phrases from the
question; use empty lists when absent.
products: Apple devices (iPhone, Apple Watch, AirPods Pro, Mac...).
symptoms: the problem, as a short phrase ("won't pair", "battery drains fast").
components: hardware parts (Bluetooth, battery, Wi-Fi, camera).
features: software features/services (iCloud Photos, Find My, Siri).
error_codes: numbered errors ("error 4013").
os_versions: OS versions ("iOS 26", "macOS Tahoe").

Question: {question}"""


# Builds a deterministic, schema-constrained LLM request for one question's mentions.
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


# Question in, Mentions out via the LLM; on any failure returns empty Mentions instead of crashing.
def extract_mentions(client: LLMClient, question: str, model: str) -> Mentions:
    try:
        return complete_structured(client, build_mention_request(question, model), Mentions)
    except (StructuredOutputError, RuntimeError):
        return Mentions()


# Question in, Mentions out using only ontology lookups and the error-code regex (no model).
def rule_mentions(question: str, ontology: Ontology) -> Mentions:
    """Model-free mentions from the ontology (always merged in)."""
    # Prefer specific product names; fall back to product families if none are found.
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


# Combines LLM and rule mentions per field, dropping blanks and duplicates but keeping order.
def merge_mentions(a: Mentions, b: Mentions) -> Mentions:
    # Order-preserving union of two phrase lists.
    def m(x: list[str], y: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for s in x + y:
            if s.strip():
                seen.setdefault(s.strip(), None)
        return list(seen)

    # Build the merged result without re-running validation (lists may exceed the field caps).
    return Mentions.model_construct(
        products=m(a.products, b.products),
        symptoms=m(a.symptoms, b.symptoms),
        components=m(a.components, b.components),
        features=m(a.features, b.features),
        error_codes=m(a.error_codes, b.error_codes),
        os_versions=m(a.os_versions, b.os_versions),
    )


# Which KG node labels each mention type is allowed to link to.
_LABELS = {
    "products": ["Product", "ProductFamily"],
    "symptoms": ["Symptom"],
    "components": ["Component"],
    "features": ["Feature"],
    "error_codes": ["ErrorCode"],
    "os_versions": ["OSVersion"],
}


# One seed node for graph search: its weight, the phrase that found it and the raw similarity.
class Seed(BaseModel):
    node_id: str
    weight: float
    mention: str
    similarity: float


# Maps a question's mentions to seed KG nodes by searching the "nodes" Qdrant collection.
class EntityLinker:
    # Keeps the Qdrant client, embedder, per-node chunk counts and the linking thresholds.
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

    # Embeds a phrase and finds the closest node names, filtered to the allowed labels.
    def _search(self, text: str, labels: list[str], limit: int) -> list[tuple[str, float]]:
        vec = self.embedder.encode([text], query=False)[0].tolist()
        # Over-fetch (3x) because one node has several aliases, then keep each node's best score.
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

    # Down-weights hub nodes: the more chunks mention a node, the smaller its weight.
    def specificity(self, node_id: str) -> float:
        return 1.0 / math.log(2 + self.node_chunk_counts.get(node_id, 0))

    # Question + mentions in, seed nodes sorted by weight out; graphrag uses them to start PageRank.
    def link(self, question: str, mentions: Mentions) -> list[Seed]:
        seeds: dict[str, Seed] = {}

        # Add a seed weighted by similarity x specificity; keep the best weight if seen twice.
        def add(nid: str, sim: float, mention: str) -> None:
            w = sim * self.specificity(nid)
            if nid not in seeds or seeds[nid].weight < w:
                seeds[nid] = Seed(node_id=nid, weight=w, mention=mention, similarity=sim)

        # Link each mention to its top nodes of the right type, keeping matches over the threshold.
        for field, labels in _LABELS.items():
            for mention in getattr(mentions, field):
                for nid, sim in self._search(mention, labels, self.per_mention):
                    if sim >= self.threshold:
                        add(nid, sim, mention)
        # The whole question against problem-like nodes catches symptoms the LLM paraphrased.
        for nid, sim in self._search(question, ["Symptom", "ErrorCode"], self.question_symptoms):
            if sim >= self.threshold:
                add(nid, sim * 0.8, "<question>")
        # Highest-weight seeds first.
        return sorted(seeds.values(), key=lambda s: -s.weight)
