"""Main SemanticTagger coordinator for tag search.

This module provides the main SemanticTagger class that coordinates:
- Data loading and caching
- API clients (embedding, reranker)
- Search strategies (cache, realtime, fallback)
"""

from __future__ import annotations

import os
import pickle
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.search.embedding_client import EmbeddingClient
from src.search.reranker_client import RerankerClient
from src.search.strategies.cache_search import CacheSearchStrategy
from src.search.strategies.fallback_search import FallbackSearchStrategy
from src.search.strategies.realtime_search import RealtimeSearchStrategy
from src.search.utils import build_stop_words


class SemanticTagger:
    """Danbooru tag search engine with multiple search strategies."""

    _instance = None  # Singleton pattern

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SemanticTagger, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance._cancel_event = None
        return cls._instance

    def __init__(
        self, csv_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize semantic tagger.

        Args:
            csv_path: Path to CSV database
            config: Configuration dictionary
        """
        if self.initialized:
            return

        self.csv_path = csv_path
        self.config = config or {}

        # Parse configuration
        self._parse_config()

        # Data
        self.df: Optional[pd.DataFrame] = None
        self.tags_data: List[Dict] = []
        self.max_log_count = 15.0

        # Pre-computed embeddings
        self.emb_en: Optional[np.ndarray] = None
        self.emb_cn: Optional[np.ndarray] = None
        self.emb_wiki: Optional[np.ndarray] = None
        self.emb_cn_core: Optional[np.ndarray] = None

        # Stop words
        self.stop_words = build_stop_words()

        # API clients
        self.embedding_client: Optional[EmbeddingClient] = None
        self.reranker_client: Optional[RerankerClient] = None

        # Search strategies (lazy init)
        self._cache_strategy: Optional[CacheSearchStrategy] = None
        self._realtime_strategy: Optional[RealtimeSearchStrategy] = None
        self._fallback_strategy: Optional[FallbackSearchStrategy] = None

        # Cancellation support
        self._cancel_event: Optional[threading.Event] = None

        self.initialized = True

    def _parse_config(self):
        """Parse configuration for API settings."""
        embedding_cfg = self.config.get("embedding", {})
        reranker_cfg = self.config.get("reranker", {})
        llm_cfg = self.config.get("llm", {})

        # Embedding API settings
        self.embedding_api_url = embedding_cfg.get(
            "api_url",
            self.config.get(
                "api_url", llm_cfg.get("base_url", "https://api.siliconflow.cn/v1")
            ),
        )
        self.embedding_api_key = embedding_cfg.get(
            "api_key", self.config.get("api_key", llm_cfg.get("api_key", ""))
        )
        self.embedding_model = (
            embedding_cfg.get("model")
            or self.config.get("embedding_model")
            or self.config.get("model", "Pro/BAAI/bge-m3")
        )

        # Reranker API settings
        self.reranker_api_url = reranker_cfg.get(
            "api_url",
            self.config.get(
                "api_url", llm_cfg.get("base_url", "https://api.siliconflow.cn/v1")
            ),
        )
        self.reranker_api_key = reranker_cfg.get(
            "api_key", self.config.get("api_key", llm_cfg.get("api_key", ""))
        )
        self.reranker_model = reranker_cfg.get("model") or self.config.get(
            "reranker_model", "Pro/BAAI/bge-reranker-v2-m3"
        )

    def set_cancel_event(self, cancel_event: Optional[threading.Event]):
        """Set cancellation event for interrupting operations."""
        self._cancel_event = cancel_event
        self._propagate_cancel_event()

    def _propagate_cancel_event(self):
        """Propagate cancellation event to all strategies."""
        if self._cache_strategy:
            self._cache_strategy.set_cancel_event(self._cancel_event)
        if self._realtime_strategy:
            self._realtime_strategy.set_cancel_event(self._cancel_event)
        if self._fallback_strategy:
            self._fallback_strategy.set_cancel_event(self._cancel_event)
        if self.embedding_client:
            self.embedding_client.set_cancel_event(self._cancel_event)
        if self.reranker_client:
            self.reranker_client.set_cancel_event(self._cancel_event)

    def load(self):
        """Load data and initialize embeddings cache."""
        self._tag_names = set()
        self._load_csv()

        if self.df is not None:
            self._tag_names = set(self.df["name"].str.lower().tolist())

        # Try loading cache
        self._load_embeddings_cache()

    def _load_csv(self):
        """Load CSV data with encoding detection."""
        if not self.csv_path:
            return

        encodings = ["utf-8", "gbk", "gb18030"]
        for enc in encodings:
            try:
                self.df = pd.read_csv(self.csv_path, dtype=str, encoding=enc).fillna("")
                break
            except Exception:
                continue

        if self.df is None:
            return

        # Preprocess
        if "post_count" not in self.df.columns:
            self.df["post_count"] = 0
        self.df["post_count"] = pd.to_numeric(
            self.df["post_count"], errors="coerce"
        ).fillna(0)

        if "cn_name" not in self.df.columns:
            self.df["cn_name"] = ""
        if "wiki" not in self.df.columns:
            self.df["wiki"] = ""
        if "name" not in self.df.columns:
            raise ValueError("CSV missing 'name' column")

        self.df["cn_core"] = self.df["cn_name"].str.split(",", n=1).str[0].str.strip()
        self.tags_data = self.df.to_dict("records")
        self.max_log_count = float(np.log1p(self.df["post_count"].max()))

    def _load_embeddings_cache(self):
        """Load pre-computed embeddings from cache."""
        cache_path = self._get_cache_path()
        new_cache_path = "cache/embeddings_cache.pkl"

        # Try new format cache
        if os.path.exists(new_cache_path):
            try:
                with open(new_cache_path, "rb") as f:
                    data = pickle.load(f)
                combined_emb = np.array(data["embeddings"], dtype=np.float32)
                self.emb_en = combined_emb
                self.emb_cn = combined_emb
                self.emb_wiki = combined_emb
                self.emb_cn_core = combined_emb
                print(
                    f"[SemanticTagger] Loaded cache ({len(data.get('names', []))} tags)"
                )
                return
            except Exception as e:
                print(f"[SemanticTagger] Cache load failed: {e}")

        # Try old format cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                self.emb_en = np.array(data["embeddings_en"], dtype=np.float32)
                self.emb_cn = np.array(data["embeddings_cn"], dtype=np.float32)
                self.emb_wiki = np.array(
                    data.get("embeddings_wiki", np.zeros_like(self.emb_en)),
                    dtype=np.float32,
                )
                self.emb_cn_core = np.array(
                    data.get("embeddings_cn_core", np.zeros_like(self.emb_en)),
                    dtype=np.float32,
                )
                print(f"[SemanticTagger] Loaded old cache format")
                return
            except Exception as e:
                print(f"[SemanticTagger] Old cache load failed: {e}")

        # Build cache automatically
        print(f"[SemanticTagger] No cache found, building...")
        self._auto_build_cache("cache/embeddings_cache.pkl")

    def _get_cache_path(self) -> str:
        """Get cache file path based on model name."""
        cache_dir = os.path.dirname(self.csv_path) if self.csv_path else "."
        base_name = (
            os.path.splitext(os.path.basename(self.csv_path))[0]
            if self.csv_path
            else "tags"
        )
        model_suffix = self.embedding_model.replace("/", "_")
        return os.path.join(cache_dir, f"{base_name}_emb_{model_suffix}.pkl")

    def _auto_build_cache(self, save_path: str):
        """Auto-build embeddings cache."""
        if self.df is None:
            print("[SemanticTagger] No data loaded, cannot build cache")
            return

        print(f"[SemanticTagger] Encoding {len(self.df)} tags...")

        texts = []
        for _, row in self.df.iterrows():
            name = row.get("name", "")
            cn_name = row.get("cn_name", "")
            text = f"{name} {cn_name.replace(',', ' ')}".strip() if cn_name else name
            texts.append(text)

        client = self._get_embedding_client()

        try:
            embeddings = client.get_embeddings(texts, batch_size=32)
        except Exception as e:
            print(f"[SemanticTagger] Encoding failed: {e}")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cache_data = {
            "embeddings": embeddings.tolist(),
            "names": self.df["name"].tolist(),
            "cn_names": self.df["cn_name"].tolist(),
            "post_counts": self.df["post_count"].tolist(),
            "categories": self.df["category"].tolist()
            if "category" in self.df.columns
            else ["0"] * len(self.df),
            "nsfw": self.df["nsfw"].tolist()
            if "nsfw" in self.df.columns
            else ["0"] * len(self.df),
        }

        with open(save_path, "wb") as f:
            pickle.dump(cache_data, f)

        # Update embeddings
        self.emb_en = embeddings
        self.emb_cn = embeddings
        self.emb_wiki = embeddings
        self.emb_cn_core = embeddings

        print(f"[SemanticTagger] Cache saved to {save_path}")

    def _get_embedding_client(self) -> EmbeddingClient:
        """Get or create embedding client."""
        if self.embedding_client is None:
            self.embedding_client = EmbeddingClient(
                api_url=self.embedding_api_url,
                api_key=self.embedding_api_key,
                model=self.embedding_model,
                config=self.config,
            )
            if self._cancel_event:
                self.embedding_client.set_cancel_event(self._cancel_event)
        return self.embedding_client

    def _get_reranker_client(self) -> Optional[RerankerClient]:
        """Get or create reranker client."""
        if self.reranker_client is None and self.reranker_api_key:
            reranker_cfg = self.config.get("reranker", {})
            if reranker_cfg.get("enabled", True):
                self.reranker_client = RerankerClient(
                    api_url=self.reranker_api_url,
                    api_key=self.reranker_api_key,
                    model=self.reranker_model,
                    config=self.config,
                )
                if self._cancel_event:
                    self.reranker_client.set_cancel_event(self._cancel_event)
        return self.reranker_client

    def _get_fallback_strategy(self) -> FallbackSearchStrategy:
        """Get or create fallback strategy."""
        if self._fallback_strategy is None:
            self._fallback_strategy = FallbackSearchStrategy(
                df=self.df,
                tags_data=self.tags_data,
                max_log_count=self.max_log_count,
                config=self.config,
            )
            if self._cancel_event:
                self._fallback_strategy.set_cancel_event(self._cancel_event)
        return self._fallback_strategy

    def _get_cache_strategy(self) -> CacheSearchStrategy:
        """Get or create cache search strategy."""
        if self._cache_strategy is None:
            embeddings = {
                "en": self.emb_en,
                "cn": self.emb_cn,
                "wiki": self.emb_wiki,
                "cn_core": self.emb_cn_core,
            }
            self._cache_strategy = CacheSearchStrategy(
                df=self.df,
                embeddings=embeddings,
                embedding_client=self._get_embedding_client(),
                reranker_client=self._get_reranker_client(),
                fallback_strategy=self._get_fallback_strategy(),
                config=self.config,
            )
            if self._cancel_event:
                self._cache_strategy.set_cancel_event(self._cancel_event)
        return self._cache_strategy

    def _get_realtime_strategy(self) -> RealtimeSearchStrategy:
        """Get or create realtime search strategy."""
        if self._realtime_strategy is None:
            self._realtime_strategy = RealtimeSearchStrategy(
                df=self.df,
                embedding_client=self._get_embedding_client(),
                fallback_strategy=self._get_fallback_strategy(),
                config=self.config,
            )
            if self._cancel_event:
                self._realtime_strategy.set_cancel_event(self._cancel_event)
        return self._realtime_strategy

    def search(
        self,
        query: str,
        top_k: int = 5,
        limit: int = 80,
        popularity_weight: float = 0.15,
    ) -> Tuple[str, List[Dict]]:
        """
        Search for tags matching query.

        Args:
            query: User query text
            top_k: Top-K results per query
            limit: Maximum results to return
            popularity_weight: Weight for popularity scoring

        Returns:
            Tuple of (tag_string, result_list)
        """
        # Use cache strategy if embeddings available, otherwise realtime
        if self.emb_en is not None:
            strategy = self._get_cache_strategy()
        else:
            strategy = self._get_realtime_strategy()

        return strategy.search(
            query=query,
            top_k=top_k,
            limit=limit,
            popularity_weight=popularity_weight,
        )

    def _tag_exists(self, tag_name: str) -> bool:
        """Check if tag exists in database."""
        if not hasattr(self, "_tag_names") or not self._tag_names:
            if self.df is not None:
                self._tag_names = set(self.df["name"].str.lower().tolist())
            else:
                return False
        return tag_name.lower() in self._tag_names


def create_semantic_tagger(
    csv_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None
) -> SemanticTagger:
    """Factory function to create SemanticTagger instance.

    Args:
        csv_path: Path to CSV database
        config: Configuration dictionary

    Returns:
        SemanticTagger instance
    """
    tagger = SemanticTagger(csv_path, config)
    tagger.load()
    return tagger
