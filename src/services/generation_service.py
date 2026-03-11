"""Generation service layer.

This module decouples GUI orchestration from prompt/description generation logic.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from src.api_client import APIClient
from src.config import get_config
from src.generator import PromptGenerator
from src.semantic_search import SemanticTagger


class GenerationService:
    """Coordinates generator, semantic search and LLM client construction."""

    def _resolve_db_path(self) -> Path:
        cfg = get_config()
        db_path = Path(
            cfg.get("database", {}).get("path", "danbooruTags/tags_enhanced.csv")
        )
        if not db_path.is_absolute():
            db_path = Path(__file__).resolve().parents[2] / db_path
        return db_path

    def _build_llm_client(
        self, cancel_event: Optional[threading.Event] = None
    ) -> APIClient:
        llm_cfg = get_config("llm") or {}
        return APIClient(
            provider=llm_cfg.get("provider", "SiliconFlow"),
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
            model=llm_cfg.get("model", "deepseek-ai/DeepSeek-V3.2"),
            cancel_event=cancel_event,
        )

    def _build_prompt_generator(
        self,
        use_semantic: bool,
        use_llm: bool,
        use_space: bool,
        cancel_event: Optional[threading.Event] = None,
    ) -> PromptGenerator:
        cfg = get_config()
        db_path = self._resolve_db_path()

        generator = PromptGenerator(
            db_path=str(db_path),
            max_tags=cfg.get("generator", {}).get("max_tags", 100),
            use_space_separator=use_space,
        )

        if use_semantic and cfg.get("semantic_search", {}).get("enabled", True):
            semantic_tagger = SemanticTagger(
                str(db_path), cfg.get("semantic_search", {})
            )
            semantic_tagger.load()
            semantic_tagger.set_cancel_event(cancel_event)
            generator.set_semantic_tagger(semantic_tagger)

        if use_llm and cfg.get("llm", {}).get("enabled", True):
            client = self._build_llm_client(cancel_event=cancel_event)
            if client.is_available():
                generator.set_api_client(client)

        return generator

    def generate_tags(
        self,
        description: Optional[str],
        use_semantic: bool,
        use_llm: bool,
        use_space: bool,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        generator = self._build_prompt_generator(
            use_semantic=use_semantic,
            use_llm=use_llm,
            use_space=use_space,
            cancel_event=cancel_event,
        )
        return generator.generate(description, use_semantic=use_semantic)

    def generate_description(
        self, cancel_event: Optional[threading.Event] = None
    ) -> str:
        print(f"\n[Generator] LLM 自动生成模式")
        print(f"[Generator] 正在生成场景描述...")

        gen_cfg = get_config("generator") or {}
        user_prompt = gen_cfg.get("auto_generate_prompt", "")
        max_chinese = gen_cfg.get("auto_generate_max_chinese_chars", 40)
        user_prompt = user_prompt.replace("{max_chinese_chars}", str(max_chinese))

        client = self._build_llm_client(cancel_event=cancel_event)
        if not client.is_available():
            raise RuntimeError("LLM 客户端不可用，请检查 API Key 与配置")

        description = client.generate("", user_prompt).strip()
        print(f"[Generator]   LLM生成的场景: {description}")
        return description
