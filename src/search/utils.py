"""Utility functions for semantic search."""

from __future__ import annotations

import re
from typing import List, Set

import numpy as np


def build_stop_words() -> Set[str]:
    """Build Chinese stop words set."""
    return {
        # Punctuation
        ",",
        ".",
        ":",
        ";",
        "?",
        "!",
        '"',
        "'",
        "`",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "-",
        "_",
        "=",
        "+",
        "/",
        "\\",
        "|",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "~",
        "，",
        "。",
        "：",
        "；",
        "？",
        "！",
        """, """,
        "'",
        "'",
        "（",
        "）",
        "【",
        "】",
        "《",
        "》",
        "、",
        "…",
        "—",
        "·",
        " ",
        "\t",
        "\n",
        "\r",
        # Particles and function words
        "的",
        "地",
        "得",
        "了",
        "着",
        "过",
        "是",
        "为",
        "被",
        "给",
        "把",
        "让",
        "由",
        "在",
        "从",
        "自",
        "向",
        "往",
        "对",
        "于",
        "和",
        "与",
        "及",
        "或",
        "且",
        "而",
        "但",
        "并",
        "即",
        "又",
        "也",
        # Interjections
        "啊",
        "吗",
        "吧",
        "呢",
        "噢",
        "哦",
        "哈",
        "呀",
        "哇",
        # Pronouns
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "这",
        "那",
        "此",
        "其",
        "谁",
        "啥",
        "某",
        "每",
        "这个",
        "那个",
        "这些",
        "那些",
        "这里",
        "那里",
        # Quantifiers
        "个",
        "位",
        "只",
        "条",
        "张",
        "幅",
        "件",
        "套",
        "双",
        "对",
        "副",
        "种",
        "类",
        "群",
        "些",
        "点",
        "份",
        "部",
        "名",
        # Adverbs
        "很",
        "太",
        "更",
        "最",
        "挺",
        "特",
        "好",
        "真",
        # Numerals
        "一",
        "一个",
        "一种",
        "一下",
        "一点",
        "一些",
        # Verbs
        "有",
        "无",
        "非",
        "没",
        "不",
    }


def smart_split(text: str, stop_words: Set[str] | None = None) -> List[str]:
    """Smart tokenization supporting Chinese and English.

    Args:
        text: Input text
        stop_words: Set of stop words to filter (optional)

    Returns:
        List of tokens
    """
    if stop_words is None:
        stop_words = set()

    import jieba

    tokens = []

    # Normalize punctuation to spaces
    text = re.sub(r"[，。！？；：''" "【】、·…—～]", " ", text)
    text = re.sub(r'[,\(\)\[\]{}:;!?\'".\\~\-]', " ", text)

    parts = text.split()

    for part in parts:
        if not part.strip() or part.isdigit():
            continue

        # Pure Chinese - use jieba
        if re.match(r"^[\u4e00-\u9fa5]+$", part):
            sub_parts = list(jieba.cut(part, cut_all=False))
            tokens.extend([p for p in sub_parts if p.strip()])
        else:
            # English/mixed - keep as is
            if re.search(r"[a-zA-Z]", part):
                tokens.append(part)

    # Filter stop words and pure symbols
    tokens = [
        t.strip()
        for t in tokens
        if t.strip() and re.search(r"[\u4e00-\u9fa5a-zA-Z]", t) and t not in stop_words
    ]

    return tokens


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two matrices.

    Args:
        a: Matrix of shape (n, d)
        b: Matrix of shape (m, d)

    Returns:
        Similarity matrix of shape (n, m)
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T) / (a_norm * b_norm.T + 1e-8)


def extract_queries(query: str) -> List[str]:
    """Extract multiple query variants from description.

    Keeps full sentence + Chinese part + English part.

    Args:
        query: User query text

    Returns:
        List of query variants
    """
    queries = [query.strip()]
    if " / " in query:
        parts = [p.strip() for p in query.split(" / ") if p.strip()]
        for p in parts:
            if p not in queries:
                queries.append(p)
    return queries
