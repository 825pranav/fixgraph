# Extraction gold-set annotation guidelines

Used for `data/gold/extraction_gold.jsonl` (50 chunks, sampled by `fixgraph kg gold-sample`).
Review drafts with `uv run fixgraph kg annotate` (`a` accept, `e` edit in Notepad, `s` skip).

## What to annotate

Annotate what the chunk (plus its article title and section heading) **states**, never outside
knowledge. Entity strings are short phrases copied from the text. Matching during evaluation is
fuzzy (normalized text similarity ≥ 0.8 or substantial containment), so exact boundaries matter
less than picking the right phrase.

| Type | Include | Exclude |
|---|---|---|
| Product | Apple devices named in the chunk or the title (iPhone, Apple Watch Series 3, Mac Pro) | apps, services (those are Features) |
| OSVersion | versions with a number or macOS name (iOS 17, macOS Catalina, macOS Ventura 13.5) | bare "iOS", "latest version of macOS" |
| Component | hardware parts (Bluetooth, display, microphone, internal SATA drive) | |
| Feature | software features/services (Find My, iCloud Backup, Siri, Focus) | UI chrome (buttons, tabs) |
| Symptom | an observable problem ("can't back up to iCloud", "calls go straight to voicemail") | goals of how-to articles ("turn on Night Shift") |
| ErrorCode | numbered errors ("HTTP 500 error", "Status error 110") | |
| Cause | an explicitly stated reason ("Zoom feature turned on") | speculation you add |
| Fix | one main action per distinct remedy ("turn off Zoom", "restart your Apple Watch") | every tap of a procedure; purely diagnostic steps |

## Relations

Same schema as the KG (`src/fixgraph/kg/schema.py`): EXHIBITS (Product→Symptom), CAUSED_BY
(Symptom→Cause), RESOLVED_BY (Symptom→Fix), ADDRESSES (Fix→Cause), INVOLVES
(Symptom→Component/Feature), REQUIRES (Fix→OSVersion/Product/Feature), APPLIES_TO
(Symptom/Fix→OSVersion), SIGNALS (ErrorCode→Symptom), RUNS (Product→OSVersion),
DEPENDS_ON (Product→Product), HAS_COMPONENT (Product→Component).

- The symptom in an "If ..." article title counts as stated for every chunk of that article.
- A fix listed under a sub-problem heading attaches to that sub-problem's symptom.
- How-to chunks with no problem usually have products/features and few or no relations.

## Status

Every record carries `annotator` and `status`. The first 50 drafts were written by Claude
(AI assistant) following these rules and are marked `status: draft` until the developer reviews
them. Headline extraction metrics state whether they were computed on draft or reviewed gold.
