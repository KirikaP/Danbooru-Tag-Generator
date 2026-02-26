#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Danbooru Tag Generator - Flet GUI 应用"""

# 必须在所有其他 import 前设置，否则 requests 先于过滤器被导入
import warnings
warnings.filterwarnings('ignore', message=".*doesn't match a supported version.*")
warnings.filterwarnings('ignore', message=".*chardet.*")
warnings.filterwarnings('ignore', message=".*urllib3.*")

import flet as ft
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config import reload_config

CONFIG_PATH = Path(__file__).parent / 'config.json'
GUI_STATE_PATH = Path(__file__).parent / 'gui_state.json'

# 控制台输出收集器
class ConsoleOutput:
    def __init__(self):
        self._buf = []       # 已完成的行
        self._cur = ""       # 当前未完成的行（\r 会覆盖它）
        self.text_field = None
        self.page = None
        self._lock = threading.Lock()
    
    def write(self, text):
        with self._lock:
            # 按\n分割输入内容
            while text:
                nl = text.find('\n')
                cr = text.find('\r')
                if nl == -1 and cr == -1:
                    self._cur += text
                    break
                elif nl != -1 and (cr == -1 or nl < cr):
                    # 普通换行
                    self._buf.append(self._cur + text[:nl] + '\n')
                    self._cur = ""
                    text = text[nl+1:]
                else:
                    # \r：覆盖当前行（不换行）
                    self._cur = ""
                    text = text[cr+1:]
        self._refresh()
    
    def _refresh(self):
        if self.text_field is None or self.page is None:
            return
        value = ''.join(self._buf) + self._cur
        self.text_field.value = value
        try:
            self.page.update()
        except Exception:
            pass
    
    def flush(self):
        pass
    
    def clear(self):
        with self._lock:
            self._buf = []
            self._cur = ""
        self._refresh()
    
    def get_value(self):
        with self._lock:
            return ''.join(self._buf) + self._cur

console_output = ConsoleOutput()
sys.stdout = console_output


def create_button(text: str, icon=None, on_click=None, **kwargs):
    """创建按钮的辅助函数 - 兼容新旧 Flet API"""
    return ft.Button(
        content=ft.Row([
            ft.Icon(icon, size=18) if icon else ft.Container(),
            ft.Text(text),
        ], spacing=5, alignment=ft.MainAxisAlignment.CENTER),
        on_click=on_click,
        **kwargs
    )


class ConfigManager:
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self):
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
    
    def save(self):
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        reload_config()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set_value(self, key: str, value: Any):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value


class AppState:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.generating = False
        self.status_text = None


def create_text_field(key: str, label: str, state: AppState, width=350, password=False):
    field = ft.TextField(
        label=label,
        value=str(state.config_manager.get_value(key, "")),
        password=password,
        border_color=ft.Colors.BLUE_400,
        width=width
    )
    field.key = key
    field.on_change = lambda e: state.config_manager.set_value(e.control.key, e.control.value)
    return field


def create_number_field(key: str, label: str, state: AppState, width=100):
    field = ft.TextField(
        label=label,
        value=str(state.config_manager.get_value(key, 0)),
        border_color=ft.Colors.BLUE_400,
        width=width
    )
    field.key = key
    field.on_change = lambda e: state.config_manager.set_value(e.control.key, int(e.control.value) if e.control.value.isdigit() else float(e.control.value))
    return field


def create_switch(key: str, label: str, state: AppState):
    switch = ft.Switch(
        label=label,
        value=state.config_manager.get_value(key, False)
    )
    switch.key = key
    switch.on_change = lambda e: state.config_manager.set_value(e.control.key, e.control.value)
    return switch


def create_dropdown(key: str, label: str, options: list, state: AppState, width=200):
    current_val = str(state.config_manager.get_value(key, ""))
    dropdown = ft.Dropdown(
        label=label,
        options=[ft.dropdown.Option(opt) for opt in options],
        value=current_val if current_val in options else options[0],
        border_color=ft.Colors.BLUE_400,
        width=width
    )
    dropdown.key = key
    dropdown.on_change = lambda e: state.config_manager.set_value(e.control.key, e.control.value)
    return dropdown


def build_generator_page(state: AppState, page: ft.Page):
    description_input = ft.TextField(
        label="输入图片描述",
        hint_text="例如：一位白发少女站在樱花树下",
        multiline=True,
        min_lines=3,
        border_color=ft.Colors.BLUE_400,
        width=550,
    )
    
    use_semantic = ft.Switch(label="使用语义搜索", value=True)
    use_llm = ft.Switch(label="使用LLM", value=True)
    use_space = ft.Switch(label="使用空格分隔", value=True)
    auto_generate = ft.Switch(label="自动生成描述", value=False)
    
    generate_btn = create_button(
        "生成标签",
        icon=ft.icons.Icons.PLAY_CIRCLE,
        on_click=lambda e: generate_tags(page, state, description_input, use_semantic, use_llm, use_space, auto_generate, result_output, status_text, progress, generate_btn),
        bgcolor=ft.Colors.BLUE_700,
        color=ft.Colors.WHITE,
        width=150,
        height=45,
    )
    
    result_output = ft.TextField(
        label="控制台输出",
        multiline=True,
        min_lines=12,
        read_only=True,
        border_color=ft.Colors.BLUE_GREY_400,
        width=550,
    )
    console_output.text_field = result_output
    console_output.page = page
    
    status_text = ft.Text("", size=12)
    progress = ft.ProgressBar(visible=False, width=550)
    state.status_text = status_text
    
    return ft.Column([
        ft.Text("标签生成", size=24, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        description_input,
        ft.Row([use_semantic, use_llm]),
        ft.Row([use_space, auto_generate]),
        progress,
        ft.Row([generate_btn, status_text]),
        ft.Divider(),
        result_output,
        ft.Row([
            create_button("复制", icon=ft.icons.Icons.COPY, on_click=lambda e: copy_result(page, result_output, status_text)),
            create_button("清空控制台", icon=ft.icons.Icons.DELETE_SWEEP, on_click=lambda e: clear_result(result_output, status_text)),
        ]),
    ], spacing=10, scroll=ft.ScrollMode.AUTO)


def build_database_page(state: AppState):
    return ft.Column([
        ft.Text("数据库配置", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        create_text_field("database.path", "数据库路径", state),
        create_text_field("database.encoding", "编码", state),
    ], spacing=10)


def build_generator_config_page(state: AppState):
    return ft.Column([
        ft.Text("生成器配置", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        create_number_field("generator.max_tags", "最大标签数", state),
        create_switch("generator.auto_tag", "自动生成标签", state),
        create_number_field("generator.auto_generate_max_chinese_chars", "自动生成中文字符数", state),
    ], spacing=10)


def build_semantic_page(state: AppState):
    return ft.Column([
        ft.Text("语义搜索配置", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        create_switch("semantic_search.enabled", "启用语义搜索", state),
        ft.Text("嵌入配置", size=16, weight=ft.FontWeight.BOLD),
        create_dropdown("semantic_search.embedding.provider", "Provider", ["SiliconFlow", "OpenAI", "DeepSeek"], state),
        create_text_field("semantic_search.embedding.api_key", "API Key", state, password=True),
        create_text_field("semantic_search.embedding.api_url", "API URL", state),
        create_text_field("semantic_search.embedding.model", "模型", state),
        ft.Divider(),
        ft.Text("重排序配置", size=16, weight=ft.FontWeight.BOLD),
        create_switch("semantic_search.reranker.enabled", "启用重排序", state),
        create_dropdown("semantic_search.reranker.provider", "Provider", ["SiliconFlow", "OpenAI", "DeepSeek"], state),
        create_text_field("semantic_search.reranker.api_key", "API Key", state, password=True),
        create_text_field("semantic_search.reranker.api_url", "API URL", state),
        create_text_field("semantic_search.reranker.model", "模型", state),
        ft.Divider(),
        ft.Text("搜索参数", size=16, weight=ft.FontWeight.BOLD),
        ft.Row([
            create_number_field("semantic_search.top_k", "Top K", state),
            create_number_field("semantic_search.max_encode_tags", "编码标签数", state),
            create_number_field("semantic_search.timeout", "超时(秒)", state),
        ]),
        ft.Row([
            create_number_field("semantic_search.limit", "限制", state),
            create_number_field("semantic_search.popularity_weight", "热门权重", state),
            create_number_field("semantic_search.similarity_threshold", "相似度阈值", state),
        ]),
    ], spacing=10, scroll=ft.ScrollMode.AUTO)


def build_llm_page(state: AppState):
    return ft.Column([
        ft.Text("大语言模型配置", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        create_switch("llm.enabled", "启用 LLM", state),
        create_dropdown("llm.provider", "Provider", ["SiliconFlow", "OpenAI", "Anthropic", "DeepSeek"], state),
        create_text_field("llm.api_key", "API Key", state, password=True),
        create_text_field("llm.base_url", "Base URL", state),
        create_text_field("llm.model", "模型", state),
        create_number_field("llm.select_tags_max", "最大选择标签数", state),
        create_number_field("llm.timeout", "超时时间(秒)", state),
        ft.Divider(),
        ft.Text("选择标签参数", size=16, weight=ft.FontWeight.BOLD),
        ft.Row([
            create_number_field("llm.select_tags.temperature", "Temp", state),
            create_number_field("llm.select_tags.top_p", "Top P", state),
            create_number_field("llm.select_tags.max_tokens", "Max Tokens", state),
        ]),
        ft.Divider(),
        ft.Text("生成参数", size=16, weight=ft.FontWeight.BOLD),
        ft.Row([
            create_number_field("llm.generate.temperature", "Temp", state),
            create_number_field("llm.generate.top_p", "Top P", state),
            create_number_field("llm.generate.max_tokens", "Max Tokens", state),
        ]),
        ft.Divider(),
        ft.Text("验证参数", size=16, weight=ft.FontWeight.BOLD),
        ft.Row([
            create_number_field("llm.validate.temperature", "Temp", state),
            create_number_field("llm.validate.top_p", "Top P", state),
            create_number_field("llm.validate.max_tokens", "Max Tokens", state),
        ]),
        ft.Divider(),
        ft.Text("思考模式", size=16, weight=ft.FontWeight.BOLD),
        ft.Row([
            create_switch("llm.thinking.select_tags", "选择标签", state),
            create_switch("llm.thinking.generate", "生成", state),
            create_switch("llm.thinking.validate", "验证", state),
        ]),
    ], spacing=10, scroll=ft.ScrollMode.AUTO)


def generate_tags(page, state, description_input, use_semantic, use_llm, use_space, auto_generate, result_output, status_text, progress, generate_btn):
    if state.generating:
        return
    
    description = description_input.value.strip()
    if auto_generate.value and not description:
        description = None
    
    if not description and not auto_generate.value:
        status_text.value = "请输入描述或启用自动生成"
        status_text.color = ft.Colors.ORANGE
        page.update()
        return
    
    state.generating = True
    generate_btn.disabled = True
    progress.visible = True
    status_text.value = "正在生成..."
    status_text.color = ft.Colors.BLUE
    page.update()
    
    def do_generate():
        try:
            result = _generate_impl(description, use_semantic.value, use_llm.value, use_space.value)
            # 将最终标签打印到控制台
            print(f"\n{'='*50}")
            print("最终标签:")
            print(result)
            print('='*50)
            
            def update_ui():
                status_text.value = "生成完成!"
                status_text.color = ft.Colors.GREEN
                progress.visible = False
                generate_btn.disabled = False
                state.generating = False
                page.update()
            page.run_thread(update_ui)
        except Exception as ex:
            def update_error():
                status_text.value = f"错误: {str(ex)}"
                status_text.color = ft.Colors.RED
                progress.visible = False
                generate_btn.disabled = False
                state.generating = False
                page.update()
            page.run_thread(update_error)
    
    thread = threading.Thread(target=do_generate, daemon=True)
    thread.start()


def _generate_impl(description: Optional[str], use_semantic: bool, use_llm: bool, use_space: bool) -> str:
    from src.config import get_config
    from src.api_client import APIClient
    from src.semantic_search import SemanticTagger
    from src.generator import PromptGenerator
    
    config = get_config()
    db_path = config.get('database', {}).get('path', 'danbooruTags/tags_enhanced.csv')
    
    if not Path(db_path).is_absolute():
        db_path = Path(__file__).parent / db_path
    
    generator = PromptGenerator(
        db_path=str(db_path),
        max_tags=config.get('generator', {}).get('max_tags', 100),
        use_space_separator=use_space
    )
    
    if use_semantic and config.get('semantic_search', {}).get('enabled', True):
        try:
            semantic_tagger = SemanticTagger(str(db_path), config)
            generator.set_semantic_tagger(semantic_tagger)
        except Exception as ex:
            print(f"语义搜索初始化失败: {ex}")
    
    if use_llm and config.get('llm', {}).get('enabled', True):
        try:
            llm_config = config.get('llm', {})
            api_client = APIClient(
                provider=llm_config.get('provider', 'SiliconFlow'),
                api_key=llm_config.get('api_key', ''),
                base_url=llm_config.get('base_url', 'https://api.siliconflow.cn/v1'),
                model=llm_config.get('model', 'deepseek-ai/DeepSeek-V3.2'),
            )
            if api_client.is_available():
                generator.set_api_client(api_client)
        except Exception as ex:
            print(f"LLM初始化失败: {ex}")
    
    return generator.generate(description, use_semantic=use_semantic)


def copy_result(page, result_output, status_text):
    value = console_output.get_value()
    if value:
        page.set_clipboard(value)
        status_text.value = "已复制到剪贴板"
        status_text.color = ft.Colors.GREEN
        page.update()


def clear_result(result_output, status_text):
    console_output.clear()
    status_text.value = ""


def save_config(page, state):
    try:
        state.config_manager.save()
        state.status_text.value = "✓ 配置已保存"
        state.status_text.color = ft.Colors.GREEN
        page.update()
    except Exception as ex:
        state.status_text.value = f"✗ 保存失败: {str(ex)}"
        state.status_text.color = ft.Colors.RED
        page.update()


def load_window_size():
    try:
        if GUI_STATE_PATH.exists():
            with open(GUI_STATE_PATH, 'r', encoding='utf-8') as f:
                state = json.load(f)
                return state.get('width', 900), state.get('height', 800)
    except:
        pass
    return 900, 800

def save_window_size(width, height):
    try:
        with open(GUI_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump({'width': width, 'height': height}, f)
    except:
        pass

def main(page: ft.Page):
    page.title = "Danbooru Tag Generator"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    
    # 加载上次窗口大小
    win_width, win_height = load_window_size()
    page.window_width = win_width
    page.window_height = win_height
    page.window_resizable = True
    page.window_min_width = 800
    page.window_min_height = 600
    
    # 保存窗口大小变化
    def on_resize(e):
        save_window_size(page.window_width, page.window_height)
    page.on_resize = on_resize
    
    state = AppState()
    
    # 构建各页面
    generator_page = build_generator_page(state, page)
    database_page = build_database_page(state)
    generator_config_page = build_generator_config_page(state)
    semantic_page = build_semantic_page(state)
    llm_page = build_llm_page(state)
    
    # 使用 Container 切换显示
    pages = [
        generator_page,
        database_page,
        generator_config_page,
        semantic_page,
        llm_page,
    ]
    
    nav = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=200,
        destinations=[
            ft.NavigationRailDestination(icon=ft.Icon(ft.icons.Icons.PLAY_CIRCLE_OUTLINED), selected_icon=ft.Icon(ft.icons.Icons.PLAY_CIRCLE), label="标签生成"),
            ft.NavigationRailDestination(icon=ft.Icon(ft.icons.Icons.DATASET_OUTLINED), selected_icon=ft.Icon(ft.icons.Icons.DATASET), label="数据库"),
            ft.NavigationRailDestination(icon=ft.Icon(ft.icons.Icons.BUILD_OUTLINED), selected_icon=ft.Icon(ft.icons.Icons.BUILD), label="生成器"),
            ft.NavigationRailDestination(icon=ft.Icon(ft.icons.Icons.FIND_IN_PAGE_OUTLINED), selected_icon=ft.Icon(ft.icons.Icons.FIND_IN_PAGE), label="语义搜索"),
            ft.NavigationRailDestination(icon=ft.Icon(ft.icons.Icons.PSYCHOLOGY_OUTLINED), selected_icon=ft.Icon(ft.icons.Icons.PSYCHOLOGY), label="LLM"),
        ],
        on_change=lambda e: switch_page(e.control.selected_index, pages, page_content),
    )
    
    page_content = ft.Container(content=pages[0], padding=20, expand=1)
    
    def switch_page(index, pages, container):
        container.content = pages[index]
        page.update()
    
    status_text = ft.Text("", size=12, color=ft.Colors.GREEN)
    state.status_text = status_text
    
    page.add(ft.Column([
        ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.Icons.TAG, size=40, color=ft.Colors.BLUE_400),
                ft.Text("Danbooru Tag Generator", size=28, weight=ft.FontWeight.BOLD)
            ]),
            padding=15,
            alignment=ft.alignment.Alignment(0, 0)
        ),
        ft.Divider(),
        ft.Row([
            nav,
            ft.VerticalDivider(width=1),
            page_content,
        ], expand=1, spacing=0),
        ft.Divider(),
        ft.Row([
            status_text,
            ft.Row([
                create_button("重置", icon=ft.icons.Icons.RESTART_ALT, on_click=lambda e: reset_config(page, state)),
                create_button("保存配置", icon=ft.icons.Icons.SAVE_OUTLINED, on_click=lambda e: save_config(page, state), bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
            ], spacing=10)
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ft.Container(height=15)
    ], expand=1))


def reset_config(page, state):
    state.config_manager.load()
    state.status_text.value = "✓ 配置已重置"
    state.status_text.color = ft.Colors.BLUE
    page.update()


if __name__ == "__main__":
    ft.run(main=main)