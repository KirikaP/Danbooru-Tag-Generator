"""Embedding API client with cancellation support."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import requests


class EmbeddingClient:
    """Client for embedding API with retry and cancellation support."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize embedding client.

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

        # HTTP session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

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

    def _request_with_retry(
        self, url: str, payload: dict, timeout: int = 60
    ) -> requests.Response:
        """POST request with exponential backoff retry.

        Args:
            url: Request URL
            payload: JSON request body
            timeout: Single request timeout in seconds

        Returns:
            Response object

        Raises:
            RuntimeError: If all retries fail
            InterruptedError: If cancelled
        """
        max_retries: int = self.config.get("max_retries", 3)
        backoff_factor: float = self.config.get("backoff_factor", 2.0)
        retry_on_status: list = self.config.get(
            "retry_on_status", [429, 500, 502, 503, 504]
        )

        last_exc = None
        for attempt in range(max_retries + 1):
            self._raise_if_cancelled()
            try:
                response = self._post_with_cancel(
                    self.session.post,
                    url,
                    json=payload,
                    timeout=timeout,
                )
                if response.status_code in retry_on_status and attempt < max_retries:
                    wait = backoff_factor**attempt
                    print(
                        f"[EmbeddingClient] Status {response.status_code}, "
                        f"retry {attempt + 1}/{max_retries} in {wait:.1f}s..."
                    )
                    self._sleep_with_cancel(wait)
                    continue
                return response
            except InterruptedError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff_factor**attempt
                    print(
                        f"[EmbeddingClient] Error: {exc}, "
                        f"retry {attempt + 1}/{max_retries} in {wait:.1f}s..."
                    )
                    self._sleep_with_cancel(wait)
                else:
                    raise
        raise RuntimeError(
            f"API request failed after {max_retries} retries: {last_exc}"
        )

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)

        Raises:
            InterruptedError: If cancelled
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            self._raise_if_cancelled()
            batch = texts[i : i + batch_size]

            payload = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
            }

            try:
                response = self._request_with_retry(
                    f"{self.api_url}/embeddings",
                    payload=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(embeddings)

                if len(texts) > 100:
                    print(
                        f"    Progress: [{batch_idx + 1}/{total_batches}] "
                        f"Encoded {min(i + batch_size, len(texts))}/{len(texts)} "
                    )

            except InterruptedError:
                raise
            except Exception as e:
                print(f"[EmbeddingClient] API error: {e}")
                raise

        return np.array(all_embeddings, dtype=np.float32)
