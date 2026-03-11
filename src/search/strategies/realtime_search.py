"""Realtime search strategy using live embedding API calls."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.search.embedding_client import EmbeddingClient
from src.search.strategies.base import SearchStrategy
from src.search.strategies.fallback_search import FallbackSearchStrategy
from src.search.utils import cosine_similarity, extract_queries


class RealtimeSearchStrategy(SearchStrategy):
    """Realtime search encoding queries and candidates on-the-fly."""

    def __init__(
        self,
        df: pd.DataFrame,
        embedding_client: EmbeddingClient,
        fallback_strategy: FallbackSearchStrategy,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize realtime search strategy.

        Args:
            df: DataFrame with tag data
            embedding_client: Client for encoding
            fallback_strategy: Fallback strategy on failure
            config: Configuration dictionary
        """
        super().__init__(
            df=df,
            embedding_client=embedding_client,
            reranker_client=None,
            config=config,
        )
        self.fallback_strategy = fallback_strategy
        self.cancel_event: Optional[threading.Event] = None

    def set_cancel_event(self, cancel_event: Optional[threading.Event]):
        """Set cancellation event for this and child strategies."""
        self.cancel_event = cancel_event
        if self.embedding_client:
            self.embedding_client.set_cancel_event(cancel_event)
        if self.fallback_strategy:
            self.fallback_strategy.set_cancel_event(cancel_event)

    def _raise_if_cancelled(self) -> None:
        """Raise InterruptedError if cancellation was requested."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise InterruptedError("用户中断了当前语义搜索")

    def _get_valid_tags(self, limit: int = 500) -> List[Dict]:
        """Filter valid tags from DataFrame.

        Args:
            limit: Maximum tags to encode

        Returns:
            List of valid tag dictionaries
        """
        valid_tags = []

        for idx, row in self.df.iterrows():
            if idx % 2000 == 0:
                self._raise_if_cancelled()
            if int(row.get("category", 0)) == 4:
                continue
            valid_tags.append(
                {
                    "tag": row["name"],
                    "cn_name": row["cn_name"],
                    "post_count": row["post_count"],
                    "category": row.get("category", "0"),
                    "nsfw": row.get("nsfw", "0"),
                }
            )

        if not valid_tags:
            return []

        # Limit by popularity if too many
        max_encode = self.config.get("max_encode_tags", 500)
        if len(valid_tags) > max_encode:
            valid_tags = sorted(
                valid_tags, key=lambda x: int(x["post_count"]), reverse=True
            )[:max_encode]

        return valid_tags

    def search(
        self,
        query: str,
        top_k: int = 5,
        limit: int = 30,
        popularity_weight: float = 0.15,
    ) -> Tuple[str, List[Dict]]:
        """Execute realtime search with direct semantic matching.

        Args:
            query: User query text
            top_k: Unused (for interface compatibility)
            limit: Maximum results to return
            popularity_weight: Weight for popularity scoring

        Returns:
            Tuple of (tag_string, result_list)
        """
        print(f"[RealtimeSearch] Starting semantic search...")

        valid_tags = self._get_valid_tags()
        if not valid_tags:
            return self.fallback_strategy.search(query, limit=limit)

        return self._direct_semantic_match(query, valid_tags, limit, popularity_weight)

    def _direct_semantic_match(
        self,
        query: str,
        candidates: List[Dict],
        limit: int = 30,
        popularity_weight: float = 0.15,
    ) -> Tuple[str, List[Dict]]:
        """Direct semantic matching: encode queries and candidates, compute max similarity.

        Args:
            query: User query text
            candidates: List of candidate tag dicts
            limit: Maximum results to return
            popularity_weight: Weight for popularity scoring

        Returns:
            Tuple of (tag_string, result_list)
        """
        queries = extract_queries(query)
        n_queries = len(queries)

        # Prepare texts for encoding
        candidate_texts = [f"{c['tag']} {c['cn_name']}" for c in candidates]
        all_texts = queries + candidate_texts

        # Batch encode
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(all_texts), batch_size):
            self._raise_if_cancelled()
            batch = all_texts[i : i + batch_size]
            embeddings = self.embedding_client.get_embeddings(batch)
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        query_emb = all_embeddings[:n_queries]
        candidate_embs = all_embeddings[n_queries:]

        # Compute similarity
        sim_matrix = cosine_similarity(query_emb, candidate_embs)
        similarities = np.max(sim_matrix, axis=0)

        # Score and rank
        final_results = {}
        max_log_count = (
            np.log1p(np.max([int(c["post_count"]) for c in candidates]))
            if candidates
            else 1
        )

        for i, cand in enumerate(candidates):
            semantic_score = similarities[i]
            pop_score = (
                np.log1p(int(cand["post_count"])) / max_log_count
                if max_log_count > 0
                else 0
            )

            final_score = (
                semantic_score * (1 - popularity_weight) + pop_score * popularity_weight
            )

            final_results[cand["tag"]] = {
                "tag": cand["tag"],
                "final_score": float(final_score),
                "semantic_score": float(semantic_score),
                "source": query[:20],
                "cn_name": cand["cn_name"],
                "layer": "DirectMatch",
                "category": cand["category"],
                "nsfw": cand["nsfw"],
                "post_count": cand["post_count"],
            }

        # Sort and filter
        sorted_tags = sorted(
            final_results.values(), key=lambda x: x["final_score"], reverse=True
        )

        similarity_threshold = self.config.get("similarity_threshold", 0.5)
        final_list = [
            t
            for t in sorted_tags[:limit]
            if t.get("semantic_score", 0) >= similarity_threshold
        ]

        print(f"[RealtimeSearch] Complete, {len(final_list)} tags found")

        tags_string = ", ".join([item["tag"] for item in final_list])
        return tags_string, final_list
