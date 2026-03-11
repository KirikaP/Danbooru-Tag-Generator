"""Base class for search strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.search.embedding_client import EmbeddingClient
from src.search.reranker_client import RerankerClient


class SearchStrategy(ABC):
    """Abstract base for search strategies."""

    def __init__(
        self,
        df: pd.DataFrame,
        embedding_client: Optional[EmbeddingClient],
        reranker_client: Optional[RerankerClient],
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize search strategy.

        Args:
            df: DataFrame with tag data
            embedding_client: Client for API embeddings
            reranker_client: Client for reranking
            config: Configuration dictionary
        """
        self.df = df
        self.embedding_client = embedding_client
        self.reranker_client = reranker_client
        self.config = config or {}
        self.max_log_count = 15.0

    def set_cancel_event(self, cancel_event):
        """Pass cancellation event to clients."""
        if self.embedding_client:
            self.embedding_client.set_cancel_event(cancel_event)
        if self.reranker_client:
            self.reranker_client.set_cancel_event(cancel_event)

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        limit: int = 30,
        popularity_weight: float = 0.15,
    ) -> Tuple[str, List[Dict]]:
        """Execute search strategy.

        Args:
            query: User query text
            top_k: Top-K results per query
            limit: Maximum results to return
            popularity_weight: Weight for popularity scoring

        Returns:
            Tuple of (tag_string, result_list)
        """
        pass
