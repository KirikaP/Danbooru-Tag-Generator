"""Fallback search strategy using keyword matching."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.search.strategies.base import SearchStrategy


class FallbackSearchStrategy(SearchStrategy):
    """Simple keyword-based fallback search."""

    def __init__(
        self,
        df: pd.DataFrame,
        tags_data: List[Dict],
        max_log_count: float,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize fallback strategy.
        
        Args:
            df: DataFrame with tag data
            tags_data: List of tag dictionaries
            max_log_count: Max log count for popularity scoring
            config: Configuration dictionary
        """
        super().__init__(df=df, embeddings=None, embedding_client=None, reranker_client=None, config=config)
        self.tags_data = tags_data
        self.max_log_count = max_log_count
        self.cancel_event: Optional[threading.Event] = None

    def set_cancel_event(self, cancel_event: Optional[threading.Event]):
        """Set cancellation event."""
        self.cancel_event = cancel_event

    def _raise_if_cancelled(self) -> None:
        """Raise InterruptedError if cancellation was requested."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise InterruptedError("用户中断了当前语义搜索")

    def search(
        self,
        query: str,
        top_k: int = 5,
        limit: int = 30,
        popularity_weight: float = 0.15,
    ) -> Tuple[str, List[Dict]]:
        """Execute keyword-based fallback search.
        
        Args:
            query: User query text
            top_k: Unused (for interface compatibility)
            limit: Maximum results to return
            popularity_weight: Unused (for interface compatibility)
            
        Returns:
            Tuple of (tag_string, result_list)
        """
        if not self.tags_data:
            return "Error: No tags loaded.", []

        query_lower = query.lower()
        results = []

        for tag in self.tags_data:
            self._raise_if_cancelled()
            
            tag_name = tag.get("name", "")
            cn_name = tag.get("cn_name", "")
            wiki = tag.get("wiki", "")

            # Skip category 4
            if int(tag.get("category", 0)) == 4:
                continue

            # Simple matching
            score = 0
            if query_lower in tag_name.lower():
                score += 10
            if query_lower in cn_name.lower():
                score += 20
            if query_lower in wiki.lower():
                score += 5

            if score > 0:
                post_count = int(tag.get("post_count", 0))
                pop_score = (
                    np.log1p(post_count) / self.max_log_count
                    if self.max_log_count > 0
                    else 0
                )

                results.append({
                    "tag": tag_name,
                    "final_score": score + pop_score * 0.1,
                    "semantic_score": score,
                    "source": query,
                    "cn_name": cn_name,
                    "layer": "Fallback",
                    "category": tag.get("category", "0"),
                    "nsfw": tag.get("nsfw", "0"),
                    "post_count": post_count,
                })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        final_list = results[:limit]

        tags_string = ", ".join([item["tag"] for item in final_list])
        return tags_string, final_list
