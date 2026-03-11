"""
提示词生成器核心模块
根据自然语言描述生成完整的文生图提示词
只使用大模型API和语义嵌入搜索
"""

import re
from typing import List, Optional, Any
from .api_client import APIClient, create_api_client


class PromptGenerator:
    """文生图提示词生成器 - 纯API/语义搜索模式"""

    CORE_TAGS = ["1girl", "1boy", "solo"]

    def __init__(
        self,
        db_path: str,
        max_tags: int = 15,
        use_space_separator: bool = True,
        api_client: Optional[APIClient] = None,
        semantic_tagger: Optional[Any] = None,
    ):
        """
        初始化提示词生成器

        Args:
            db_path: 数据库文件路径
            max_tags: 最大标签数量
            use_space_separator: 是否使用空格分隔标签
            api_client: 大模型API客户端
            semantic_tagger: 语义搜索标签器
        """
        self.db_path = db_path
        self.max_tags = max_tags
        self.use_space_separator = use_space_separator
        self.api_client = api_client
        self.semantic_tagger = semantic_tagger

    def set_api_client(self, api_client: APIClient):
        """设置API客户端"""
        self.api_client = api_client

    def set_semantic_tagger(self, semantic_tagger):
        """设置语义搜索标签器"""
        self.semantic_tagger = semantic_tagger

    def enable_api(self):
        """启用API"""
        if self.api_client is None:
            self.api_client = create_api_client()

    def is_api_enabled(self) -> bool:
        """检查API是否启用"""
        return self.api_client is not None and self.api_client.is_available()

    def _tag_to_display(self, tag: str) -> str:
        """将标签转换为显示格式"""
        if self.use_space_separator:
            return tag.replace("_", " ")
        return tag

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return re.search(r"[\u4e00-\u9fff]", text) is not None

    @staticmethod
    def _contains_english(text: str) -> bool:
        return re.search(r"[A-Za-z]", text) is not None

    def _translate_to_other_language(self, description: str) -> Optional[str]:
        if not self.is_api_enabled():
            return None

        has_zh = self._contains_chinese(description)
        has_en = self._contains_english(description)

        if has_zh and has_en:
            return None
        if not has_zh and not has_en:
            return None

        target_lang = "英文" if has_zh else "中文"
        prompt = (
            f"请将下面这段内容准确翻译为{target_lang}。"
            f"只输出翻译结果，不要添加解释、前缀或引号。\n\n"
            f"原文：{description}"
        )

        try:
            translated = self.api_client.generate("", prompt).strip()  # type: ignore[union-attr]
            if not translated:
                return None
            if translated == description:
                return None
            return translated
        except InterruptedError:
            raise
        except Exception:
            return None

    def _prepare_bilingual_description(self, description: str) -> str:
        translated = self._translate_to_other_language(description)
        if not translated:
            return description
        print(f"[Generator] 单语输入检测到，已补全双语描述")
        print(f"[Generator] 双语描述: {description} / {translated}")
        return f"{description} / {translated}"

    def generate(
        self, description: Optional[str] = None, use_semantic: bool = True
    ) -> str:
        """
        生成提示词

        新流程（两阶段）:
          阶段一：语义搜索 → 获取 80 个候选标签
          阶段二：LLM 从候选列表中选择与描述最相关的标签

        Args:
            description: 图片场景描述
            use_semantic: 是否使用语义搜索

        Returns:
            提示词字符串
        """
        # 无描述时让 API 先生成描述
        if not description or not description.strip():
            if self.is_api_enabled():
                return self._generate_with_api_auto()
            return self._generate_default_prompt()

        description = description.strip()
        description = self._prepare_bilingual_description(description)

        # ── 两阶段主流程 ──────────────────────────────────────────────
        if use_semantic and self.semantic_tagger:
            return self._generate_semantic_then_select(description)

        # 纯 API 回退（无语义搜索）
        if self.is_api_enabled():
            return self._generate_with_api(description)

        return self._generate_default_prompt()

    def _generate_semantic_then_select(self, description: str) -> str:
        """
        两阶段生成：
        1. 语义搜索 → 获取候选标签
        2. LLM 从候选列表中选择最合适的标签
        """

        # ─── 阶段一：语义搜索获取候选标签 ────────────────────────────
        print(f"[Generator] 语义搜索...")
        try:
            from .config import get_config

            sem_cfg = get_config("semantic_search") or {}
            search_limit = sem_cfg.get("limit", 80)
            search_top_k = sem_cfg.get("top_k", 5)
            search_pop_w = sem_cfg.get("popularity_weight", 0.15)
            tags_string, results = self.semantic_tagger.search(  # type: ignore[union-attr]
                description,
                top_k=search_top_k,
                limit=search_limit,
                popularity_weight=search_pop_w,
            )
        except InterruptedError:
            raise
        except Exception as e:
            print(f"[Generator] 语义搜索失败: {e}")
            tags_string, results = "", []

        if not tags_string:
            print("[Generator] 语义搜索无结果，回退到纯 API 生成")
            if self.is_api_enabled():
                return self._generate_with_api(description)
            return self._generate_default_prompt()

        n_candidates = len([t for t in tags_string.split(",") if t.strip()])
        print(f"[Generator] 语义搜索完成，获得 {n_candidates} 个候选")
        print(f"[Generator] 匹配标签: {tags_string}")

        # ─── 阶段二：LLM 从候选列表中选取 ────────────────────────────
        if self.is_api_enabled():
            print(f"[Generator] LLM 筛选标签...")
            try:
                raw_selected = self.api_client.select_tags(
                    description, tags_string, self.max_tags
                )  # type: ignore[union-attr]

                # 验证：只保留候选列表中存在的标签
                candidate_set = {
                    t.strip().lower() for t in tags_string.split(",") if t.strip()
                }
                selected_tags = []
                for t in raw_selected.split(","):
                    t = t.strip()
                    if t and t.lower() in candidate_set:
                        selected_tags.append(t)

                print(f"[Generator] LLM 筛选完成，选出 {len(selected_tags)} 个标签")

                # ─── 阶段三：验证标签是否符合描述 ────────────────────────────
                print(f"[Generator] 验证标签...")
                try:
                    validated_tags = self.api_client.validate_tags(
                        description, selected_tags
                    )  # type: ignore[union-attr]
                    if validated_tags != selected_tags:
                        print(
                            f"[Generator] 验证后删除 {len(selected_tags) - len(validated_tags)} 个无关标签"
                        )
                    selected_tags = validated_tags
                except InterruptedError:
                    raise
                except Exception:
                    pass

                if selected_tags:
                    display_tags = [self._tag_to_display(tag) for tag in selected_tags]
                    return ", ".join(display_tags)

            except InterruptedError:
                raise
            except Exception as e:
                print(f"[Generator] LLM 筛选失败，使用语义搜索结果")

        # 回退：无 API 或 LLM 选择失败，直接取语义搜索 top N
        top_tags = [r["tag"] for r in results[: self.max_tags]]
        print(f"[Generator] 使用语义搜索 Top {len(top_tags)} 标签")
        display_tags = [self._tag_to_display(tag) for tag in top_tags]
        return ", ".join(display_tags)

    def _generate_with_api(self, description: str) -> str:
        """纯 API 生成（无语义搜索时回退）"""
        try:
            api_result = self.api_client.generate(description)  # type: ignore[union-attr]

            tags = [t.strip() for t in api_result.split(",") if t.strip()]
            tags.sort()
            if len(tags) > self.max_tags:
                core = [t for t in self.CORE_TAGS if t in tags]
                others = [t for t in tags if t not in self.CORE_TAGS]
                tags = core + others[: self.max_tags - len(core)]

            display_tags = [self._tag_to_display(tag) for tag in tags]
            return ", ".join(display_tags)

        except InterruptedError:
            raise
        except Exception as e:
            print(f"API调用失败: {e}")
            return self._generate_default_prompt()

    def _generate_with_api_auto(self) -> str:
        """无描述时让API生成描述，然后用语义搜索"""
        try:
            from .config import get_config

            gen_cfg = get_config("generator") or {}
            user_prompt = gen_cfg.get("auto_generate_prompt", "")

            # 替换占位符
            max_chinese = gen_cfg.get("auto_generate_max_chinese_chars", 40)
            user_prompt = user_prompt.replace("{max_chinese_chars}", str(max_chinese))
            description = self.api_client.generate("", user_prompt).strip()

            # 使用两阶段流程处理生成的描述
            if self.semantic_tagger:
                return self._generate_semantic_then_select(description)

            # 没有语义搜索时回退
            return self._generate_with_api(description)
        except InterruptedError:
            raise
        except Exception as e:
            print(f"[Generator] API调用失败: {e}")
            return self._generate_default_prompt()

    def _generate_default_prompt(self) -> str:
        """生成默认提示词"""
        tags = ["1girl", "solo", "simple_background", "anime_style"]
        display_tags = [self._tag_to_display(tag) for tag in tags]
        return ", ".join(display_tags)

    def batch_generate(self, descriptions: List[str]) -> List[str]:
        """批量生成提示词"""
        return [self.generate(desc) for desc in descriptions]


def create_generator(
    db_path: str,
    max_tags: Optional[int] = None,
    use_space_separator: bool = True,
    api_client: Optional[APIClient] = None,
    semantic_tagger: Optional[Any] = None,
) -> PromptGenerator:
    """创建生成器实例（max_tags 默认从 config.json generator.max_tags 读取）"""
    if max_tags is None:
        from .config import get_config

        max_tags = (get_config("generator") or {}).get("max_tags", 40)
    return PromptGenerator(
        db_path=db_path,
        max_tags=max_tags,  # type: ignore
        use_space_separator=use_space_separator,
        api_client=api_client,
        semantic_tagger=semantic_tagger,
    )
