"""Cache-based search strategy using pre-computed embeddings."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.search.embedding_client import EmbeddingClient
from src.search.reranker_client import RerankerClient
from src.search.strategies.base import SearchStrategy
from src.search.strategies.fallback_search import FallbackSearchStrategy
from src.search.utils import cosine_similarity, extract_queries


class CacheSearchStrategy(SearchStrategy):
    """Search using pre-computed embeddings with optional reranking."""

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: Dict[str, np.ndarray],
        embedding_client: EmbeddingClient,
        reranker_client: Optional[RerankerClient],
        fallback_strategy: FallbackSearchStrategy,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize cache search strategy.

        Args:
            df: DataFrame with tag data
            embeddings: Dict with 'en', 'cn', 'wiki', 'cn_core' embeddings
            embedding_client: Client for query encoding
            reranker_client: Optional client for reranking
            fallback_strategy: Fallback strategy on encoding failure
            config: Configuration dictionary
        """
        super().__init__(
            df=df,
            embeddings=None,
            embedding_client=embedding_client,
            reranker_client=reranker_client,
            config=config,
        )
        self.emb_en = embeddings.get("en")
        self.emb_cn = embeddings.get("cn")
        self.emb_wiki = embeddings.get("wiki")
        self.emb_cn_core = embeddings.get("cn_core")
        self.fallback_strategy = fallback_strategy
        self.cancel_event: Optional[threading.Event] = None

    def set_cancel_event(self, cancel_event: Optional[threading.Event]):
        """Set cancellation event for this and child strategies."""
        self.cancel_event = cancel_event
        if self.embedding_client:
            self.embedding_client.set_cancel_event(cancel_event)
        if self.reranker_client:
            self.reranker_client.set_cancel_event(cancel_event)
        if self.fallback_strategy:
            self.fallback_strategy.set_cancel_event(cancel_event)

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
        """Execute cache-based search with reranking.

        Args:
            query: User query text
            top_k: Number of top results per query
            limit: Maximum results to return
            popularity_weight: Weight for popularity scoring

        Returns:
            Tuple of (tag_string, result_list)
        """
        queries = extract_queries(query)

        # Encode queries
        try:
            query_embeddings = self.embedding_client.get_embeddings(queries)
        except InterruptedError:
            raise
        except Exception as e:
            print(f"[CacheSearch] Query encoding failed: {e}")
            return self.fallback_strategy.search(query, limit=limit)

        # Multi-layer similarity search
        def max_sim(emb_matrix):
            sims = cosine_similarity(query_embeddings, emb_matrix)
            return np.max(sims, axis=0, keepdims=True)

        sim_en = max_sim(self.emb_en)
        sim_cn = max_sim(self.emb_cn)
        sim_wiki = max_sim(self.emb_wiki)
        sim_cn_core = max_sim(self.emb_cn_core)

        # Get top-K indices
        def get_topk_indices(sim_matrix, k):
            return np.argsort(-sim_matrix, axis=1)[:, :k]

        hits_en = get_topk_indices(sim_en, limit)
        hits_cn = get_topk_indices(sim_cn, limit)
        hits_wiki = get_topk_indices(sim_wiki, limit)
        hits_cn_core = get_topk_indices(sim_cn_core, limit)

        # Merge results from all layers
        final_results = {}
        max_log_count = (
            np.log1p(float(self.df["post_count"].max())) if len(self.df) > 0 else 1.0
        )

        for layer_name, hits in [
            ("EN", hits_en[0]),
            ("CN", hits_cn[0]),
            ("Wiki", hits_wiki[0]),
            ("Core", hits_cn_core[0]),
        ]:
            for idx in hits:
                score = {
                    "EN": sim_en[0, idx],
                    "CN": sim_cn[0, idx],
                    "Wiki": sim_wiki[0, idx],
                    "Core": sim_cn_core[0, idx],
                }[layer_name]

                row = self.df.iloc[idx]
                tag_name = row["name"]

                pop_score = np.log1p(float(row["post_count"])) / max_log_count
                final_score = (score * (1 - popularity_weight)) + (
                    pop_score * popularity_weight
                )

                if (
                    tag_name not in final_results
                    or final_score > final_results[tag_name]["final_score"]
                ):
                    final_results[tag_name] = {
                        "tag": tag_name,
                        "final_score": float(final_score),
                        "semantic_score": float(score),
                        "source": query[:20],
                        "cn_name": row["cn_name"],
                        "layer": layer_name,
                        "category": row.get("category", "0"),
                        "nsfw": row.get("nsfw", "0"),
                        "post_count": row.get("post_count", "0"),
                    }

        # Sort and filter
        sorted_tags = sorted(
            final_results.values(), key=lambda x: x["final_score"], reverse=True
        )

        similarity_threshold = self.config.get("similarity_threshold", 0.5)
        filtered_tags = []

        for tag in sorted_tags:
            self._raise_if_cancelled()
            if int(tag.get("category", 0)) == 4:
                continue
            if tag.get("semantic_score", 0) < similarity_threshold:
                continue
            filtered_tags.append(tag)

        pre_rerank = filtered_tags[: limit * 2]

        # Reranker refinement
        if self.reranker_client and len(pre_rerank) > 0:
            rerank_query = queries[-1] if len(queries) > 1 else query
            try:
                documents = [f"{t['tag']} {t['cn_name']}" for t in pre_rerank]
                ranked = self.reranker_client.rerank(
                    rerank_query, documents, top_n=limit
                )

                idx_score = {r[0]: r[1] for r in ranked}
                for i, t in enumerate(pre_rerank):
                    if i in idx_score:
                        t["reranker_score"] = idx_score[i]
                        t["final_score"] = t["final_score"] * 0.3 + idx_score[i] * 0.7

                pre_rerank.sort(key=lambda x: x["final_score"], reverse=True)
                print(f"[CacheSearch] Reranking complete")
            except InterruptedError:
                raise
            except Exception as e:
                print(f"[CacheSearch] Reranker failed, skipped: {e}")

        final_list = pre_rerank[:limit]
        print(f"[CacheSearch] Complete, {len(final_list)} tags found")

        tags_string = ", ".join([item["tag"] for item in final_list])
        return tags_string, final_list
