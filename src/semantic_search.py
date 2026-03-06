"""
语义搜索模块 - 基于Embedding API的Danbooru标签检索
使用硅基流动(SiliconFlow)的embedding API进行向量检索

This module now re-exports from the refactored src.search package.
The architecture has been decoupled into:
- src.search.embedding_client: Embedding API client
- src.search.reranker_client: Reranker API client
- src.search.strategies: Search strategies (cache, realtime, fallback)
- src.search.tagger: Main SemanticTagger coordinator
- src.search.utils: Utility functions

For direct access to components, import from src.search instead.
"""

# Re-export main classes for backwards compatibility
from src.search.tagger import SemanticTagger, create_semantic_tagger
from src.search.embedding_client import EmbeddingClient
from src.search.reranker_client import RerankerClient
from src.search.utils import (
    build_stop_words,
    smart_split,
    cosine_similarity,
    extract_queries,
)

__all__ = [
    # Main API
    "SemanticTagger",
    "create_semantic_tagger",
    # Clients
    "EmbeddingClient",
    "RerankerClient",
    # Utilities
    "build_stop_words",
    "smart_split",
    "cosine_similarity",
    "extract_queries",
]
