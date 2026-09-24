"""Schema-constrained triple extraction from chunks (spec §8.1).

Single source of truth for the prompt and output schema: the extraction runner
(kg.run_extract) and the validator (kg.validation) import them from here.

Prompt v2 (see DECISIONS.md D13): the model fills a *problem-centric nested* schema
(product -> problems -> causes/fixes) instead of an entity list plus index-based relations.
Relation types follow from where a string sits in the JSON, so type violations are impossible
by construction; `to_graph` converts deterministically into typed entities and relations.
Every string must be copied from the chunk, and validation fuzzy-grounds it there.

Uses: core.models.Chunk, core.ontology, llm.base (request types).
"""

from pydantic import BaseModel, Field

from fixgraph.core.models import Chunk
from fixgraph.core.ontology import Ontology
from fixgraph.llm.base import ChatMessage, LLMRequest

PROMPT_VERSION = "v2"

# ---------------------------------------------------------------------------
# What the LLM fills (compact, nested, bounded to prevent runaway generation)
# ---------------------------------------------------------------------------


class LLMFix(BaseModel):
    action: str = Field(description="The fix, copied from the text (2-15 words).")
    addresses_cause: str | None = Field(
        default=None, description="Exact text of the cause (from causes) this fix addresses."
    )
    requires: list[str] = Field(
        default_factory=lambda: list[str](),
        max_length=4,
        description="Prerequisites named in the text: OS versions, products, features.",
    )


class LLMProblem(BaseModel):
    symptom: str = Field(description="The observable problem, copied from the text.")
    products: list[str] = Field(default_factory=lambda: list[str](), max_length=5)
    error_codes: list[str] = Field(default_factory=lambda: list[str](), max_length=3)
    components: list[str] = Field(default_factory=lambda: list[str](), max_length=4)
    features: list[str] = Field(default_factory=lambda: list[str](), max_length=4)
    os_versions: list[str] = Field(default_factory=lambda: list[str](), max_length=4)
    causes: list[str] = Field(default_factory=lambda: list[str](), max_length=5)
    fixes: list[LLMFix] = Field(default_factory=lambda: list[LLMFix](), max_length=8)


class LLMProduct(BaseModel):
    name: str
    components: list[str] = Field(default_factory=lambda: list[str](), max_length=5)
    os_versions: list[str] = Field(default_factory=lambda: list[str](), max_length=4)
    depends_on: list[str] = Field(default_factory=lambda: list[str](), max_length=3)


class LLMExtraction(BaseModel):
    products: list[LLMProduct] = Field(default_factory=lambda: list[LLMProduct](), max_length=6)
    problems: list[LLMProblem] = Field(default_factory=lambda: list[LLMProblem](), max_length=6)


# ---------------------------------------------------------------------------
# Normalized graph form (what validation, gold annotation and the KG consume)
# ---------------------------------------------------------------------------


class GraphEntity(BaseModel):
    type: str
    text: str


class GraphRelation(BaseModel):
    head: int
    rel: str
    tail: int
    evidence: str  # string whose presence in the chunk supports the relation


class ChunkExtraction(BaseModel):
    entities: list[GraphEntity] = Field(default_factory=lambda: list[GraphEntity]())
    relations: list[GraphRelation] = Field(default_factory=lambda: list[GraphRelation]())


class _GraphBuilder:
    def __init__(self) -> None:
        self.g = ChunkExtraction()
        self._index: dict[tuple[str, str], int] = {}

    def ent(self, type_: str, text: str) -> int | None:
        text = text.strip().strip(".")
        if not text:
            return None
        key = (type_, text.lower())
        if key not in self._index:
            self._index[key] = len(self.g.entities)
            self.g.entities.append(GraphEntity(type=type_, text=text))
        return self._index[key]

    def rel(self, head: int | None, rel: str, tail: int | None, evidence: str) -> None:
        if head is None or tail is None or head == tail:
            return
        if any(r.head == head and r.rel == rel and r.tail == tail for r in self.g.relations):
            return
        self.g.relations.append(GraphRelation(head=head, rel=rel, tail=tail, evidence=evidence))


def classify_requirement(text: str, ontology: Ontology) -> str:
    """Type a prerequisite string: OSVersion, Product or Feature (rule-based)."""
    if ontology.os_versions_in(text) or text.split(" ")[0] in ontology.os_platforms:
        return "OSVersion"
    if ontology.families_in(text) and not ontology.features_in(text):
        return "Product"
    return "Feature"


def to_graph(x: LLMExtraction, ontology: Ontology) -> ChunkExtraction:
    b = _GraphBuilder()
    for p in x.products:
        prod = b.ent("Product", p.name)
        for c in p.components:
            b.rel(prod, "HAS_COMPONENT", b.ent("Component", c), c)
        for v in p.os_versions:
            b.rel(prod, "RUNS", b.ent("OSVersion", v), v)
        for d in p.depends_on:
            b.rel(prod, "DEPENDS_ON", b.ent("Product", d), d)
    for pr in x.problems:
        sym = b.ent("Symptom", pr.symptom)
        for name in pr.products:
            b.rel(b.ent("Product", name), "EXHIBITS", sym, pr.symptom)
        for code in pr.error_codes:
            b.rel(b.ent("ErrorCode", code), "SIGNALS", sym, code)
        for c in pr.components:
            b.rel(sym, "INVOLVES", b.ent("Component", c), c)
        for f in pr.features:
            b.rel(sym, "INVOLVES", b.ent("Feature", f), f)
        for v in pr.os_versions:
            b.rel(sym, "APPLIES_TO", b.ent("OSVersion", v), v)
        causes: dict[str, int | None] = {}
        for c in pr.causes:
            causes[c.strip().lower()] = cid = b.ent("Cause", c)
            b.rel(sym, "CAUSED_BY", cid, c)
        for fx in pr.fixes:
            fid = b.ent("Fix", fx.action)
            b.rel(sym, "RESOLVED_BY", fid, fx.action)
            if fx.addresses_cause:
                cause_id = causes.get(fx.addresses_cause.strip().lower())
                b.rel(fid, "ADDRESSES", cause_id, fx.addresses_cause)
            for req in fx.requires:
                b.rel(fid, "REQUIRES", b.ent(classify_requirement(req, ontology), req), req)
    return b.g


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You extract troubleshooting facts from Apple support text as JSON.
Use ONLY what the text states. Copy short phrases from the text; do not invent or generalize.

products: Apple devices named in the text (e.g. "iPhone", "Apple Watch Series 10", "Mac").
  components: hardware parts of that product the text mentions (Bluetooth, battery, Wi-Fi).
  os_versions: specific OS versions the product runs (e.g. "iOS 26.1", "watchOS 26").
  depends_on: other products it needs to work (e.g. Apple Watch depends_on "iPhone").
problems: each problem/symptom the text describes and how to solve it.
  symptom: the observable problem ("won't pair", "battery drains quickly", "can't sign in").
  products: devices that have this problem.
  error_codes: numbered errors that signal it ("error 4013").
  components / features: hardware parts / software features involved.
  os_versions: OS versions this problem applies to.
  causes: why it happens, only if the text says.
  fixes: actions that solve it, one main action each (not every tap).
    addresses_cause: copy the cause text this fix addresses, if clear.
    requires: prerequisites for the fix (OS version, product, feature).
If the text is a how-to with no problem, return problems: [].
Keep every string under 15 words."""

_FEWSHOT: list[tuple[str, str]] = [
    (
        "Article: If your AirPods won't connect\nSection: Check your iPhone\nText:\n"
        "If your AirPods won't connect to your iPhone, make sure Bluetooth is on. "
        "Your AirPods might be out of charge: put both AirPods in the charging case and let "
        "them charge for 30 seconds. Then update your iPhone to iOS 26 or later.",
        '{"products":[{"name":"AirPods","components":["charging case"],"os_versions":[],'
        '"depends_on":["iPhone"]}],"problems":[{"symptom":"AirPods won\'t connect",'
        '"products":["AirPods"],"error_codes":[],"components":["Bluetooth"],"features":[],'
        '"os_versions":[],"causes":["AirPods might be out of charge"],"fixes":[{"action":'
        '"make sure Bluetooth is on","addresses_cause":null,"requires":[]},{"action":'
        '"put both AirPods in the charging case and let them charge","addresses_cause":'
        '"AirPods might be out of charge","requires":[]},{"action":"update your iPhone",'
        '"addresses_cause":null,"requires":["iOS 26"]}]}]}',
    ),
    (
        "Article: If you see error 4013 on your iPhone\nSection: Update your Mac\nText:\n"
        "If you see error 4013 when you restore your iPhone, your Mac might be running "
        "outdated software. Update your Mac, then try again.",
        '{"products":[],"problems":[{"symptom":"error 4013 when you restore your iPhone",'
        '"products":["iPhone"],"error_codes":["error 4013"],"components":[],"features":[],'
        '"os_versions":[],"causes":["your Mac might be running outdated software"],"fixes":'
        '[{"action":"Update your Mac","addresses_cause":"your Mac might be running outdated '
        'software","requires":[]}]}]}',
    ),
    (
        "Article: About Apple Pay\nSection: Where you can use it\nText:\n"
        "You can use Apple Pay on iPhone and Apple Watch in stores that accept contactless "
        "payments.",
        '{"products":[{"name":"iPhone","components":[],"os_versions":[],"depends_on":[]},'
        '{"name":"Apple Watch","components":[],"os_versions":[],"depends_on":[]}],'
        '"problems":[]}',
    ),
]


def chunk_prompt(chunk: Chunk, article_title: str) -> str:
    return f"Article: {article_title}\nSection: {chunk.heading}\nText:\n{chunk.text}"


def build_messages(chunk: Chunk, article_title: str) -> list[ChatMessage]:
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
    for user, assistant in _FEWSHOT:
        messages.append(ChatMessage(role="user", content=user))
        messages.append(ChatMessage(role="assistant", content=assistant))
    messages.append(ChatMessage(role="user", content=chunk_prompt(chunk, article_title)))
    return messages


def build_request(
    chunk: Chunk, article_title: str, model: str, num_ctx: int = 8192, max_tokens: int = 1200
) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=build_messages(chunk, article_title),
        temperature=0.0,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        think=False,
        json_schema=LLMExtraction.model_json_schema(),
    )
