#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Danbooru Tag Generator - Browser-first Flet GUI"""

import json
import os
import sys
import threading
import traceback
import contextlib
import time
from pathlib import Path
from typing import Any, Dict, Optional

import flet as ft

from src.config import get_config, reload_config
from src.api_client import APIClient
from src.semantic_search import SemanticTagger
from src.generator import PromptGenerator

CONFIG_PATH = Path(__file__).parent / "config.json"
GUI_STATE_PATH = Path(__file__).parent / "gui_state.json"


def _colors():
    return ft.Colors if hasattr(ft, "Colors") else ft.colors


def _icons():
    if hasattr(ft, "Icons"):
        return ft.Icons
    if hasattr(ft, "icons") and hasattr(ft.icons, "Icons"):
        return ft.icons.Icons
    return ft.icons


def _button(*args, **kwargs):
    if hasattr(ft, "Button"):
        return ft.Button(*args, **kwargs)
    return ft.ElevatedButton(*args, **kwargs)


C = _colors()
I = _icons()


ZH_LABELS = {
    "database": "数据库",
    "database.path": "数据库.路径",
    "database.encoding": "数据库.编码",
    "generator": "生成器",
    "generator.max_tags": "生成器.最大标签数",
    "generator.auto_tag": "生成器.自动标签",
    "generator.auto_generate_max_chinese_chars": "生成器.自动生成中文字符上限",
    "generator.auto_generate_prompt": "生成器.自动生成提示词",
    "semantic_search": "语义搜索",
    "semantic_search.enabled": "语义搜索.启用",
    "semantic_search.embedding": "语义搜索.嵌入",
    "semantic_search.embedding.provider": "语义搜索.嵌入.服务商",
    "semantic_search.embedding.api_key": "语义搜索.嵌入.API密钥",
    "semantic_search.embedding.api_url": "语义搜索.嵌入.API地址",
    "semantic_search.embedding.model": "语义搜索.嵌入.模型",
    "semantic_search.reranker": "语义搜索.重排序",
    "semantic_search.reranker.enabled": "语义搜索.重排序.启用",
    "semantic_search.reranker.provider": "语义搜索.重排序.服务商",
    "semantic_search.reranker.api_key": "语义搜索.重排序.API密钥",
    "semantic_search.reranker.api_url": "语义搜索.重排序.API地址",
    "semantic_search.reranker.model": "语义搜索.重排序.模型",
    "semantic_search.top_k": "语义搜索.Top K",
    "semantic_search.max_encode_tags": "语义搜索.最大编码标签数",
    "semantic_search.timeout": "语义搜索.超时时间",
    "semantic_search.limit": "语义搜索.候选上限",
    "semantic_search.popularity_weight": "语义搜索.热度权重",
    "semantic_search.similarity_threshold": "语义搜索.相似度阈值",
    "semantic_search.max_retries": "语义搜索.最大重试次数",
    "semantic_search.backoff_factor": "语义搜索.退避因子",
    "semantic_search.retry_on_status": "语义搜索.重试状态码",
    "llm": "大语言模型",
    "llm.enabled": "大语言模型.启用",
    "llm.provider": "大语言模型.服务商",
    "llm.api_key": "大语言模型.API密钥",
    "llm.base_url": "大语言模型.基础地址",
    "llm.model": "大语言模型.模型",
    "llm.select_tags_max": "大语言模型.筛选标签最大数",
    "llm.system_prompt": "大语言模型.系统提示词",
    "llm.select_tags": "大语言模型.筛选参数",
    "llm.select_tags.temperature": "大语言模型.筛选参数.温度",
    "llm.select_tags.top_p": "大语言模型.筛选参数.Top P",
    "llm.select_tags.max_tokens": "大语言模型.筛选参数.最大Tokens",
    "llm.generate": "大语言模型.生成参数",
    "llm.generate.temperature": "大语言模型.生成参数.温度",
    "llm.generate.top_p": "大语言模型.生成参数.Top P",
    "llm.generate.max_tokens": "大语言模型.生成参数.最大Tokens",
    "llm.validate": "大语言模型.验证参数",
    "llm.validate.temperature": "大语言模型.验证参数.温度",
    "llm.validate.top_p": "大语言模型.验证参数.Top P",
    "llm.validate.max_tokens": "大语言模型.验证参数.最大Tokens",
    "llm.thinking": "大语言模型.思考模式",
    "llm.thinking.select_tags": "大语言模型.思考模式.筛选标签",
    "llm.thinking.generate": "大语言模型.思考模式.生成描述",
    "llm.thinking.validate": "大语言模型.思考模式.验证标签",
    "llm.timeout": "大语言模型.超时时间",
    "llm.validate_prompt": "大语言模型.验证提示词",
    "llm.remove_tags_prompt": "大语言模型.移除标签提示词",
    "llm.max_retries": "大语言模型.最大重试次数",
    "llm.backoff_factor": "大语言模型.退避因子",
    "llm.retry_on_status": "大语言模型.重试状态码",
}


def _zh_label(path: str) -> str:
    return ZH_LABELS.get(path, path)


class ConfigManager:
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self):
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def save(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        reload_config()

    def get_value(self, key: str, default: Any = None) -> Any:
        value: Any = self.config
        for section in key.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(section, default)
        return value

    def set_value(self, key: str, value: Any):
        parts = key.split(".")
        node = self.config
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value


class AppState:
    def __init__(self):
        self.config = ConfigManager()
        self.generating = False
        self.status: Optional[ft.Text] = None
        self.progress: Optional[ft.ProgressBar] = None
        self.generate_btn = None
        self.generate_desc_btn = None
        self.generate_desc_and_tags_btn = None
        self.log_view: Optional[ft.ListView] = None
        self.log_text_control: Optional[ft.Text] = None
        self.log_text: str = ""

    def set_status(self, page: ft.Page, text: str, color):
        if self.status is not None:
            self.status.value = text
            self.status.color = color
            page.update()

    def append_log(self, page: ft.Page, text: str):
        if self.log_text_control is None:
            return
        self.log_text += text + "\n"
        self.log_text_control.value = self.log_text
        if self.log_view is not None:
            try:
                self.log_view.scroll_to(offset=-1)
            except Exception:
                pass
        page.update()


def make_text_field(state: AppState, key: str, label: str, width=320, password=False):
    field = ft.TextField(
        label=label,
        value=str(state.config.get_value(key, "")),
        width=width,
        password=password,
        border_color=C.BLUE_400,
    )

    def on_change(e):
        state.config.set_value(key, e.control.value)

    field.on_change = on_change
    return field


def make_number_field(state: AppState, key: str, label: str, width=140):
    field = ft.TextField(
        label=label,
        value=str(state.config.get_value(key, 0)),
        width=width,
        border_color=C.BLUE_400,
    )

    def on_change(e):
        raw = e.control.value.strip()
        if raw == "":
            return
        try:
            number = int(raw) if raw.isdigit() else float(raw)
            state.config.set_value(key, number)
        except ValueError:
            return

    field.on_change = on_change
    return field


def make_switch(state: AppState, key: str, label: str):
    sw = ft.Switch(label=label, value=bool(state.config.get_value(key, False)))

    def on_change(e):
        state.config.set_value(key, bool(e.control.value))

    sw.on_change = on_change
    return sw


def make_dropdown(state: AppState, key: str, label: str, options, width=220):
    current = str(state.config.get_value(key, options[0]))
    dd = ft.Dropdown(
        label=label,
        value=current if current in options else options[0],
        options=[ft.dropdown.Option(opt) for opt in options],
        width=width,
        border_color=C.BLUE_400,
    )

    def on_change(e):
        state.config.set_value(key, e.control.value)

    dd.on_change = on_change
    return dd


def _is_secret_key(path: str) -> bool:
    key = path.lower()
    return "api_key" in key or key.endswith("token") or "secret" in key


def _make_leaf_editor(state: AppState, key_path: str, value: Any):
    label = _zh_label(key_path)

    if isinstance(value, bool):
        control = ft.Switch(label=label, value=value)

        def on_change(e):
            state.config.set_value(key_path, bool(e.control.value))
            state.config.save()

        control.on_change = on_change
        return control

    if isinstance(value, int) and not isinstance(value, bool):
        control = ft.TextField(label=label, value=str(value), width=320, border_color=C.BLUE_400)

        def on_change(e):
            raw = (e.control.value or "").strip()
            if raw in {"", "-"}:
                return
            try:
                state.config.set_value(key_path, int(raw))
            except ValueError:
                return

        def on_blur(_):
            state.config.save()

        control.on_change = on_change
        control.on_blur = on_blur
        return control

    if isinstance(value, float):
        control = ft.TextField(label=label, value=str(value), width=320, border_color=C.BLUE_400)

        def on_change(e):
            raw = (e.control.value or "").strip()
            if raw in {"", "-", ".", "-."}:
                return
            try:
                state.config.set_value(key_path, float(raw))
            except ValueError:
                return

        def on_blur(_):
            state.config.save()

        control.on_change = on_change
        control.on_blur = on_blur
        return control

    if value is None:
        control = ft.TextField(label=label, value="", width=760, border_color=C.BLUE_400, hint_text="null")

        def on_change(e):
            raw = e.control.value
            state.config.set_value(key_path, None if raw == "" else raw)

        def on_blur(_):
            state.config.save()

        control.on_change = on_change
        control.on_blur = on_blur
        return control

    if isinstance(value, (list, dict)):
        control = ft.TextField(
            label=label,
            value=json.dumps(value, ensure_ascii=False),
            width=760,
            border_color=C.BLUE_400,
            multiline=True,
            min_lines=2,
            max_lines=4,
        )

        def on_change(e):
            raw = (e.control.value or "").strip()
            if raw == "":
                return
            try:
                parsed = json.loads(raw)
                state.config.set_value(key_path, parsed)
                e.control.border_color = C.BLUE_400
            except Exception:
                e.control.border_color = C.RED_400
            try:
                e.control.update()
            except Exception:
                pass

        def on_blur(e):
            raw = (e.control.value or "").strip()
            try:
                json.loads(raw)
                state.config.save()
            except Exception:
                pass

        control.on_change = on_change
        control.on_blur = on_blur
        return control

    control = ft.TextField(
        label=label,
        value=str(value),
        width=760,
        border_color=C.BLUE_400,
        password=_is_secret_key(key_path),
        can_reveal_password=_is_secret_key(key_path),
    )

    def on_change(e):
        state.config.set_value(key_path, e.control.value)

    def on_blur(_):
        state.config.save()

    control.on_change = on_change
    control.on_blur = on_blur
    return control


def _build_config_controls(state: AppState, node: Any, prefix: str = ""):
    controls = []
    if isinstance(node, dict):
        items = list(node.items())
        for key, value in items:
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                controls.append(ft.Text(_zh_label(path), size=15, weight=ft.FontWeight.BOLD))
                controls.extend(_build_config_controls(state, value, path))
                controls.append(ft.Divider())
            else:
                controls.append(_make_leaf_editor(state, path, value))
    return controls


def _generate_impl(description: Optional[str], use_semantic: bool, use_llm: bool, use_space: bool) -> str:
    cfg = get_config()
    db_path = cfg.get("database", {}).get("path", "danbooruTags/tags_enhanced.csv")
    db_path = Path(db_path)
    if not db_path.is_absolute():
        db_path = Path(__file__).parent / db_path

    generator = PromptGenerator(
        db_path=str(db_path),
        max_tags=cfg.get("generator", {}).get("max_tags", 100),
        use_space_separator=use_space,
    )

    if use_semantic and cfg.get("semantic_search", {}).get("enabled", True):
        try:
            generator.set_semantic_tagger(SemanticTagger(str(db_path), cfg.get("semantic_search", {})))
        except Exception:
            pass

    if use_llm and cfg.get("llm", {}).get("enabled", True):
        try:
            llm_cfg = cfg.get("llm", {})
            client = APIClient(
                provider=llm_cfg.get("provider", "SiliconFlow"),
                api_key=llm_cfg.get("api_key", ""),
                base_url=llm_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
                model=llm_cfg.get("model", "deepseek-ai/DeepSeek-V3.2"),
            )
            if client.is_available():
                generator.set_api_client(client)
        except Exception:
            pass

    return generator.generate(description, use_semantic=use_semantic)


def _generate_natural_language_description() -> str:
    print(f"\n[Generator] LLM 自动生成模式")
    print(f"[Generator]   正在生成场景描述...")

    gen_cfg = get_config("generator") or {}
    user_prompt = gen_cfg.get("auto_generate_prompt", "")
    max_chinese = gen_cfg.get("auto_generate_max_chinese_chars", 40)
    user_prompt = user_prompt.replace("{max_chinese_chars}", str(max_chinese))

    llm_cfg = get_config("llm") or {}
    client = APIClient(
        provider=llm_cfg.get("provider", "SiliconFlow"),
        api_key=llm_cfg.get("api_key", ""),
        base_url=llm_cfg.get("base_url", "https://api.siliconflow.cn/v1"),
        model=llm_cfg.get("model", "deepseek-ai/DeepSeek-V3.2"),
    )

    if not client.is_available():
        raise RuntimeError("LLM 客户端不可用，请检查 API Key 与配置")

    description = client.generate("", user_prompt).strip()
    print(f"[Generator]   LLM生成的场景: {description}")
    return description


def load_window_size():
    try:
        if GUI_STATE_PATH.exists():
            with open(GUI_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get("width", 1100)), int(data.get("height", 820))
    except Exception:
        pass
    return 1100, 820


def save_window_size(width, height):
    try:
        with open(GUI_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"width": int(width), "height": int(height)}, f)
    except Exception:
        pass


def _section_card(title: str, controls, icon=None):
    """带标题的配置卡片区块"""
    header_row = ft.Row(
        [ft.Icon(icon, size=16, color=C.BLUE_400)] + [ft.Text(title, size=13, weight=ft.FontWeight.W_600, color=C.BLUE_GREY_700)]
        if icon else [ft.Text(title, size=13, weight=ft.FontWeight.W_600, color=C.BLUE_GREY_700)],
        spacing=6,
    )
    return ft.Container(
        content=ft.Column([header_row, *controls], spacing=8),
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        border=ft.border.all(1, C.BLUE_GREY_200),
        border_radius=8,
    )


def build_generator_page(page: ft.Page, state: AppState):
    # ── 描述输入框 ──────────────────────────────────────────────────────────
    desc_input = ft.TextField(
        hint_text="输入画面描述，例如：一位白发少女站在樱花树下",
        multiline=True,
        min_lines=3,
        max_lines=8,
        border_color=C.BLUE_400,
        expand=True,
    )

    # ── 日志区 ──────────────────────────────────────────────────────────────
    state.log_text_control = ft.Text("", selectable=True, size=12)
    state.log_view = ft.ListView(
        controls=[state.log_text_control],
        auto_scroll=True,
        spacing=2,
        expand=True,
    )
    state.status = ft.Text("", size=12)
    state.progress = ft.ProgressBar(visible=False)

    # ── 快捷设置控件（持久化到 config） ─────────────────────────────────────
    def _cfg_switch(key: str, label: str):
        sw = ft.Switch(label=label, value=bool(state.config.get_value(key, False)), label_style=ft.TextStyle(size=13))
        def _on(e):
            state.config.set_value(key, bool(e.control.value))
            state.config.save()
        sw.on_change = _on
        return sw

    def _cfg_int_field(key: str, label: str, width=100):
        tf = ft.TextField(label=label, value=str(state.config.get_value(key, 0)), width=width,
                          border_color=C.BLUE_400, text_size=13, label_style=ft.TextStyle(size=12))
        def _on_change(e):
            raw = (e.control.value or "").strip()
            if raw and raw.lstrip("-").isdigit():
                state.config.set_value(key, int(raw))
        def _on_blur(_):
            state.config.save()
        tf.on_change = _on_change
        tf.on_blur = _on_blur
        return tf

    def _cfg_text_field(key: str, label: str, width=340):
        tf = ft.TextField(label=label, value=str(state.config.get_value(key, "")), width=width,
                          border_color=C.BLUE_400, text_size=13, label_style=ft.TextStyle(size=12))
        def _on_change(e):
            state.config.set_value(key, e.control.value)
        def _on_blur(_):
            state.config.save()
        tf.on_change = _on_change
        tf.on_blur = _on_blur
        return tf

    sw_semantic   = _cfg_switch("semantic_search.enabled", "语义搜索")
    sw_llm        = _cfg_switch("llm.enabled", "大语言模型")
    sw_space      = ft.Switch(label="标签用空格分隔", value=True, label_style=ft.TextStyle(size=13))
    tf_max_tags   = _cfg_int_field("generator.max_tags",    "最大标签数", width=110)
    tf_sel_max    = _cfg_int_field("llm.select_tags_max",   "LLM 筛选上限", width=110)
    tf_llm_model  = _cfg_text_field("llm.model", "LLM 模型", width=340)
    tf_emb_model  = _cfg_text_field("semantic_search.embedding.model", "Embedding 模型", width=260)

    sw_think_sel  = _cfg_switch("llm.thinking.select_tags", "筛选标签")
    sw_think_gen  = _cfg_switch("llm.thinking.generate",    "生成描述")
    sw_think_val  = _cfg_switch("llm.thinking.validate",    "验证标签")

    # ── 按钮组 ───────────────────────────────────────────────────────────────
    state.generate_desc_btn         = None
    state.generate_desc_and_tags_btn = None

    def _all_btns():
        return [b for b in [state.generate_btn, state.generate_desc_btn, state.generate_desc_and_tags_btn] if b is not None]

    def run_with_live_logs(task_func, on_success, success_text):
        if state.generating:
            return
        state.generating = True
        for b in _all_btns():
            b.disabled = True
        state.progress.visible = True
        state.log_text = ""
        if state.log_text_control is not None:
            state.log_text_control.value = ""
        state.set_status(page, "正在执行…", C.BLUE)
        page.update()

        def worker():
            line_buffer = []
            current_line = ""
            buffer_lock = threading.Lock()
            last_ui_push = 0.0

            def push_to_ui(force=False):
                nonlocal last_ui_push
                if state.log_text_control is None:
                    return
                now = time.monotonic()
                if not force and (now - last_ui_push) < 0.08:
                    return
                with buffer_lock:
                    parts = list(line_buffer)
                    if current_line:
                        parts.append(current_line)
                    text = "\n".join(parts)
                state.log_text = text
                state.log_text_control.value = text
                last_ui_push = now
                if state.log_view is not None:
                    try:
                        state.log_view.scroll_to(offset=-1)
                    except Exception:
                        pass
                try:
                    page.update()
                except Exception:
                    pass

            class TeeWriter:
                def __init__(self, original_stream):
                    self.original_stream = original_stream

                def write(self, text):
                    if not text:
                        return 0
                    try:
                        self.original_stream.write(text)
                    except Exception:
                        pass
                    nonlocal current_line
                    with buffer_lock:
                        for ch in text:
                            if ch == "\r":
                                current_line = ""
                            elif ch == "\n":
                                line_buffer.append(current_line)
                                current_line = ""
                            else:
                                current_line += ch
                    push_to_ui(force=False)
                    return len(text)

                def flush(self):
                    try:
                        self.original_stream.flush()
                    except Exception:
                        pass

            tee_out = TeeWriter(sys.stdout)
            tee_err = TeeWriter(sys.stderr)

            try:
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    result = task_func()
                on_success(result)
                state.set_status(page, success_text, C.GREEN)
            except Exception as ex:
                with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    traceback.print_exc()
                state.set_status(page, f"错误: {ex}", C.RED)
            finally:
                push_to_ui(force=True)
                state.generating = False
                for b in _all_btns():
                    b.disabled = False
                state.progress.visible = False
                page.update()

        threading.Thread(target=worker, daemon=True).start()

    def run_generate(_):
        description = (desc_input.value or "").strip()
        if not description:
            state.set_status(page, "请输入描述", C.ORANGE)
            return
        def task_func():
            result = _generate_impl(description, sw_semantic.value, sw_llm.value, sw_space.value)
            print("=" * 50)
            print("最终标签:")
            print(result)
            print("=" * 50)
            return result
        run_with_live_logs(task_func, lambda _r: None, "生成完成 ✓")

    def run_generate_description(_):
        def task_func():
            return _generate_natural_language_description()
        def on_success(text):
            desc_input.value = text
            try:
                desc_input.update()
            except Exception:
                pass
        run_with_live_logs(task_func, on_success, "描述生成完成 ✓")

    def run_generate_description_and_tags(_):
        def task_func():
            generated = _generate_natural_language_description()
            print("[Generator] 使用生成的描述继续生成标签…")
            result = _generate_impl(generated, sw_semantic.value, sw_llm.value, sw_space.value)
            print("=" * 50)
            print("最终标签:")
            print(result)
            print("=" * 50)
            return generated
        def on_success(text):
            desc_input.value = text
            try:
                desc_input.update()
            except Exception:
                pass
        run_with_live_logs(task_func, on_success, "随机创作完成 ✓")

    def clear_log(_):
        state.log_text = ""
        if state.log_text_control is not None:
            state.log_text_control.value = ""
        state.set_status(page, "", C.GREEN)
        page.update()

    def copy_log(_):
        text = state.log_text
        if text:
            page.set_clipboard(text)
            state.set_status(page, "已复制到剪贴板 ✓", C.GREEN)

    # ── 主操作按钮 ────────────────────────────────────────────────────────────
    state.generate_btn = _button(
        content=ft.Row([ft.Icon(I.PLAY_CIRCLE, size=18), ft.Text("生成标签", size=14)], spacing=6),
        on_click=run_generate,
        bgcolor=C.BLUE_700,
        color=C.WHITE,
        height=42,
    )
    state.generate_desc_btn = _button(
        content=ft.Row([ft.Icon(I.AUTO_AWESOME, size=16), ft.Text("随机生成描述", size=13)], spacing=5),
        on_click=run_generate_description,
        height=36,
    )
    state.generate_desc_and_tags_btn = _button(
        content=ft.Row([ft.Icon(I.AUTO_FIX_HIGH, size=18), ft.Text("随机创作并生成标签", size=14)], spacing=6),
        on_click=run_generate_description_and_tags,
        bgcolor=C.TEAL_600,
        color=C.WHITE,
        height=42,
    )

    # ── 布局 ───────────────────────────────────────────────────────────────────
    quick_settings = _section_card(
        "快捷设置",
        [
            ft.Row([sw_semantic, sw_llm, sw_space], wrap=True, spacing=4),
            ft.Row([tf_max_tags, tf_sel_max, tf_llm_model], wrap=True, spacing=12),
            ft.Row([tf_emb_model], wrap=True, spacing=12),
            ft.Row(
                [ft.Text("思考模式：", size=13, color=C.BLUE_GREY_600), sw_think_sel, sw_think_gen, sw_think_val],
                wrap=True, spacing=4,
            ),
        ],
        icon=I.TUNE,
    )

    return ft.Column(
        [
            # ── 标题行 ──
            ft.Row(
                [ft.Text("标签生成", size=22, weight=ft.FontWeight.BOLD),
                 ft.Container(expand=True),
                 state.generate_desc_btn],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Divider(height=1),
            # ── 描述输入 ──
            desc_input,
            # ── 主按钮行 ──
            ft.Row(
                [state.generate_btn, state.generate_desc_and_tags_btn, ft.Container(expand=True), state.status],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                wrap=False,
            ),
            state.progress,
            # ── 快捷设置 ──
            quick_settings,
            ft.Divider(height=1),
            # ── 日志区 ──
            ft.Row(
                [ft.Text("运行日志", size=14, weight=ft.FontWeight.W_600),
                 ft.Container(expand=True),
                 _button(content=ft.Row([ft.Icon(I.COPY, size=16), ft.Text("复制", size=13)], spacing=4), on_click=copy_log, height=32),
                 _button(content=ft.Row([ft.Icon(I.DELETE_SWEEP, size=16), ft.Text("清空", size=13)], spacing=4), on_click=clear_log, height=32)],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
            ft.Container(
                content=state.log_view,
                border=ft.border.all(1, C.BLUE_GREY_300),
                border_radius=6,
                padding=8,
                expand=True,
                height=300,
            ),
        ],
        spacing=10,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )


def build_config_page(state: AppState):
    controls = _build_config_controls(state, state.config.config)
    if not controls:
        controls = [ft.Text("配置为空", color=C.ORANGE)]

    return ft.Column(
        [
            ft.Text("高级配置", size=22, weight=ft.FontWeight.BOLD),
            ft.Divider(height=1),
            ft.Row(
                [ft.Icon(I.INFO_OUTLINE, size=14, color=C.BLUE_GREY_400),
                 ft.Text("所有修改将自动保存到 config.json", size=12, color=C.BLUE_GREY_400)],
                spacing=4,
            ),
            *controls,
        ],
        spacing=10,
        scroll=ft.ScrollMode.AUTO,
    )


def main(page: ft.Page):
    page.title = "Danbooru Tag Generator"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0

    win_w, win_h = load_window_size()
    page.window_width = win_w
    page.window_height = win_h
    page.window_min_width = 960
    page.window_min_height = 700

    def on_resize(_):
        save_window_size(page.window_width, page.window_height)

    page.on_resize = on_resize

    state = AppState()

    generator_page = build_generator_page(page, state)
    config_page = build_config_page(state)

    page_tabs = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="标签生成", content=ft.Container(generator_page, padding=20)),
            ft.Tab(text="配置", content=ft.Container(config_page, padding=20)),
        ],
        expand=1,
    )

    page.add(
        ft.Column(
            [
                ft.Container(
                    ft.Row([ft.Icon(I.TAG, size=32, color=C.BLUE_400), ft.Text("Danbooru Tag Generator", size=24, weight=ft.FontWeight.BOLD)]),
                    padding=16,
                ),
                ft.Divider(height=1),
                page_tabs,
            ],
            expand=1,
            spacing=0,
        )
    )


def run_browser_app():
    exe_dir = str(Path(sys.executable).parent)
    scripts_dir = str(Path(sys.executable).parent / "Scripts")
    path_items = os.environ.get("PATH", "").split(os.pathsep)
    path_items = [item for item in path_items if item and item.lower() not in {exe_dir.lower(), scripts_dir.lower()}]
    os.environ["PATH"] = os.pathsep.join([exe_dir, scripts_dir] + path_items)

    app_view = None
    if hasattr(ft, "AppView") and hasattr(ft.AppView, "WEB_BROWSER"):
        app_view = ft.AppView.WEB_BROWSER

    kwargs = {"target": main, "host": "127.0.0.1", "port": 8550}
    if app_view is not None:
        kwargs["view"] = app_view

    runner = ft.run if hasattr(ft, "run") else ft.app
    try:
        runner(**kwargs)
    except TypeError:
        runner(target=main)


if __name__ == "__main__":
    run_browser_app()
