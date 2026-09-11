"""FixGraph: troubleshooting knowledge graph, GNN link prediction and GraphRAG benchmark.

Flow: ingest (scrape/parse/chunk) -> kg (extract/resolve/build) -> gnn -> retrieval + answer ->
bench. Entry point: cli.py (`fixgraph`); cli.py reads `__version__` for `fixgraph version`.
"""

__version__ = "0.1.0"
