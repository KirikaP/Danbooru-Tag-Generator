"""Reranker API client with cancellation support."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


class RerankerClient:
    """Client for reranker API with retry and cancellation support."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize reranker client.

        Args:
            api_url: API endpoint URL
            api_key: API key for authentication
            model: Model name to use
            config: Optional configuration dict with retry settings
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.config = config or {}
        self.timeout = self.config.get("timeout", 60)
        self.cancel_event: Optional[threading.Event] = None

    def set_cancel_event(self, cancel_event: Optional[threading.Event]):
        """Set cancellation event for interrupting operations."""
        self.cancel_event = cancel_event

    def _raise_if_cancelled(self) -> None:
        """Raise InterruptedError if cancellation was requested."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise InterruptedError("用户中断了当前语义搜索")

    def _sleep_with_cancel(self, seconds: float) -> None:
        """Sleep that can be interrupted by cancellation."""
        if seconds <= 0:
            return
        if self.cancel_event is None:
            time.sleep(seconds)
            return
        if self.cancel_event.wait(timeout=seconds):
            raise InterruptedError("用户中断了当前语义搜索")

    def _post_with_cancel(self, post_func, *args, **kwargs):
        """Execute POST request in an interruptible context."""
        self._raise_if_cancelled()

        result = {}
        done = threading.Event()

        def _request_worker():
            try:
                result["response"] = post_func(*args, **kwargs)
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(target=_request_worker, daemon=True).start()

        while not done.wait(timeout=0.1):
            self._raise_if_cancelled()

        self._raise_if_cancelled()

        if "error" in result:
            raise result["error"]
        return result["response"]

    def rerank(
        self, query: str, documents: List[str], top_n: int
    ) -> List[Tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_n: Number of top results to return

        Returns:
            List of (index, relevance_score) tuples sorted by score descending

        Raises:
            InterruptedError: If cancelled
        """
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_retries = self.config.get("max_retries", 3)
        backoff = self.config.get("backoff_factor", 2.0)

        for attempt in range(max_retries + 1):
            self._raise_if_cancelled()
            try:
                resp = self._post_with_cancel(
                    requests.post,
                    f"{self.api_url}/rerank",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                return [(r["index"], r["relevance_score"]) for r in results]
            except InterruptedError:
                raise
            except Exception as e:
                if attempt < max_retries:
                    wait = backoff**attempt
                    print(
                        f"[RerankerClient] Retry {attempt + 1}/{max_retries} "
                        f"after {wait:.1f}s: {e}"
                    )
                    self._sleep_with_cancel(wait)
                else:
                    raise
        return []
