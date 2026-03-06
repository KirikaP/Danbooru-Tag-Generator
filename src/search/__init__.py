"""Semantic search module for Danbooru tag retrieval.

Decoupled architecture:
- embedding_client: Embedding API interactions
- reranker_client: Reranker API interactions
- strategies: Various search strategies
- tagger: Main coordinator (SemanticTagger)
"""

from src.search.tagger import SemanticTagger

__all__ = ["SemanticTagger"]
