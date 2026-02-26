#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""API客户端模块 - 支持多种大模型API"""

import os
import json
import time
import sys
import threading
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .config import get_config


class APIClient:
    """通用API客户端"""
    
    def __init__(self, 
                 provider: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 system_prompt: Optional[str] = None):
        """初始化API客户端
        
        Args:
            provider: API提供商 (openai/anthropic/azure/custom)
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            system_prompt: 系统提示词
        """
        # 从配置加载参数（优先使用llm配置，兼容api配置）
        config = get_config('llm') or get_config('api') or {}
        
        self.provider = config.get('provider')
        self.api_key = config.get('api_key') or os.environ.get('LLM_API_KEY') or os.environ.get('API_KEY')
        self.base_url = config.get('base_url')
        self.model = config.get('model')
        self.enable_thinking = config.get('enable_thinking', False)
        self.timeout = config.get('timeout', 60)
        # 各步骤独立的 thinking 控制（优先级高于全局 enable_thinking）
        thinking_cfg = config.get('thinking', {})
        self.thinking_select_tags = thinking_cfg.get('select_tags', self.enable_thinking)
        self.thinking_generate   = thinking_cfg.get('generate',     self.enable_thinking)
        self.thinking_validate   = thinking_cfg.get('validate',     False)
        self.system_prompt = config.get('system_prompt')
    
    def is_available(self) -> bool:
        """检查API是否可用"""
        if not REQUESTS_AVAILABLE:
            return False
        if not self.api_key:
            return False
        return True

    def _request_with_retry(self, url: str, headers: dict, data: dict, timeout: int) -> Any:
        """带重试的 POST 请求

        Args:
            url: 请求地址
            headers: 请求头
            data: 请求体（JSON）
            timeout: 单次请求超时秒数

        Returns:
            requests.Response 对象
        """
        cfg = get_config('llm') or get_config('api') or {}
        max_retries = cfg.get('max_retries', 3)
        backoff_factor = cfg.get('backoff_factor', 2.0)
        retry_on_status = cfg.get('retry_on_status', [429, 500, 502, 503, 504])

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
                if response.status_code in retry_on_status and attempt < max_retries:
                    wait = backoff_factor ** attempt
                    print(f"[APIClient] 状态码 {response.status_code}，{wait:.1f}秒后重试 "
                          f"(第{attempt + 1}/{max_retries}次)...")
                    time.sleep(wait)
                    continue
                return response
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff_factor ** attempt
                    print(f"[APIClient] 请求异常: {exc}，{wait:.1f}秒后重试 "
                          f"(第{attempt + 1}/{max_retries}次)...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"API请求在 {max_retries} 次重试后仍失败: {last_exc}")
    
    def generate(self, prompt: str, user_prompt: Optional[str] = None) -> str:
        """调用API生成文本
        
        Args:
            prompt: 用户输入的描述
            user_prompt: 自定义用户提示词（可选）
            
        Returns:
            API返回的文本内容
        """
        if not self.is_available():
            raise RuntimeError("API客户端不可用，请检查是否安装了requests库并配置了API密钥")
        
        return self._call_openai(prompt, user_prompt)
    
    def _print_thinking(self, message: dict, enabled: bool = True) -> None:
        """（保留接口，思考链不再打印到控制台）"""
        pass

    def _spinner(self, label: str, stop_event: threading.Event) -> None:
        """在独立线程中输出动态 spinner，stop_event 置位后退出"""
        frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = 0
        while not stop_event.is_set():
            sys.stdout.write(f'\r  {frames[idx % len(frames)]}  {label} ')
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(label) + 10) + '\r')
        sys.stdout.flush()

    def _call_with_spinner(self, label: str, fn, *args, **kwargs):
        """执行 fn(*args, **kwargs) 的同时显示 spinner"""
        stop = threading.Event()
        t = threading.Thread(target=self._spinner, args=(label, stop), daemon=True)
        t.start()
        try:
            result = fn(*args, **kwargs)
        finally:
            stop.set()
            t.join()
        return result
    
    def select_tags(self, description: str, candidate_tags: str, max_tags: int = 25) -> str:
        """从候选标签列表中让LLM选择最匹配描述的标签
        
        Args:
            description: 用户的自然语言描述
            candidate_tags: 逗号分隔的候选标签字符串（由语义搜索返回）
            max_tags: 最多选择的标签数量
            
        Returns:
            逗号分隔的已选标签字符串
        """
        if not self.is_available():
            raise RuntimeError("API客户端不可用")
        
        n_candidates = len([t for t in candidate_tags.split(',') if t.strip()])
        
        # 从配置读取 select_tags 专用参数（支持新旧两种格式）
        cfg = get_config('llm') or get_config('api') or {}
        select_cfg = cfg.get('select_tags', {})
        select_temperature = select_cfg.get('temperature', cfg.get('select_tags_temperature'))
        select_top_p = select_cfg.get('top_p', cfg.get('select_tags_top_p'))
        select_max_tokens = select_cfg.get('max_tokens', cfg.get('select_tags_max_tokens'))
        select_tags_max = cfg.get('select_tags_max')
        
        # 获取 system prompt 并替换占位符
        base_system = cfg.get('system_prompt')
        
        # 动态替换 system_prompt 中的 {select_tags_max} 占位符
        system = base_system.replace('{select_tags_max}', str(select_tags_max))
        
        user_msg = (
            f"用户描述：{description}\n\n"
            f"候选标签列表（共 {n_candidates} 个，来自Danbooru语义搜索结果）：\n"
            f"{candidate_tags}\n\n"
            f"请从以上候选标签中选择最多 {max_tags} 个最符合该描述的标签，"
            f"按相关度从高到低排列，只输出逗号分隔的英文标签名。"
        )
        
        start = time.time()
        label = '正在思考并筛选标签...' if self.thinking_select_tags else '正在筛选标签...'
        response = self._call_with_spinner(label, self._request_with_retry,
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            data={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": select_temperature,
                "top_p": select_top_p,
                "max_tokens": select_max_tokens,
                "enable_thinking": self.thinking_select_tags
            },
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API调用失败: {response.status_code} - {response.text}")
        
        message = response.json()['choices'][0]['message']
        self._print_thinking(message, self.thinking_select_tags)
        return message['content'].strip()

    def _call_openai(self, prompt: str, user_prompt: Optional[str] = None) -> str:
        """调用OpenAI兼容API"""
        # 从配置读取 generate 专用参数
        cfg = get_config('llm') or get_config('api') or {}
        gen_cfg = cfg.get('generate', {})
        gen_temperature = gen_cfg.get('temperature', 1.0)
        gen_top_p = gen_cfg.get('top_p', 0.95)
        gen_max_tokens = gen_cfg.get('max_tokens', 128000)
        
        start_time = time.time()
        label = '正在思考并生成内容...' if self.thinking_generate else '正在生成内容...'
        response = self._call_with_spinner(label, self._request_with_retry,
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            data={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt or prompt}
                ],
                "temperature": gen_temperature,
                "top_p": gen_top_p,
                "max_tokens": gen_max_tokens,
                "enable_thinking": self.thinking_generate
            },
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API调用失败: {response.status_code} - {response.text}")
        
        message = response.json()['choices'][0]['message']
        self._print_thinking(message, self.thinking_generate)
        return message['content']

    def validate_tags(self, description: str, tags: List[str]) -> List[str]:
        """
        验证生成的标签是否符合原始描述，删除无关标签
        
        Args:
            description: 原始自然语言描述
            tags: 生成的标签列表
            
        Returns:
            验证后的标签列表（删除无关标签）
        """
        if not tags or not description:
            return tags
        
        if not self.is_available():
            return tags
        
        from .config import get_config
        cfg = get_config('llm') or get_config('api') or {}
        
        # 从配置读取 validate 专用参数（支持新旧两种格式）
        val_cfg = cfg.get('validate', {})
        val_temperature = val_cfg.get('temperature', 0.3)
        val_top_p = val_cfg.get('top_p', 0.95)
        val_max_tokens = val_cfg.get('max_tokens', 64)
        
        validate_system = cfg.get('validate_prompt', 
            "你是标签质量审核员。你的任务是判断给定的标签列表是否符合用户描述。"
            "判断标准：标签是否描述了用户描述中提到的内容。"
            "只输出 '符合' 或 '不符合'，不要输出其他内容。"
        )
        
        response = self._request_with_retry(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            data={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": validate_system},
                    {"role": "user", "content": f"用户描述：{description}\n\n生成的标签列表：{', '.join(tags)}\n\n请判断这些标签是否符合用户描述。只输出 '符合' 或 '不符合'。"}
                ],
                "temperature": val_temperature,
                "top_p": val_top_p,
                "max_tokens": val_max_tokens
            },
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            return tags
        
        result_text = response.json()['choices'][0]['message']['content'].strip()
        
        if '符合' in result_text:
            return tags
        else:
            return self._remove_irrelevant_tags(description, tags)

    def _remove_irrelevant_tags(self, description: str, tags: List[str]) -> List[str]:
        """让LLM识别并返回需要保留的相关标签"""
        from .config import get_config
        cfg = get_config('llm') or get_config('api') or {}
        
        remove_system = cfg.get('remove_tags_prompt',
            "你是标签质量审核员。你的任务是找出与用户描述无关的标签并删除。"
            "输出所有需要保留的标签，用逗号分隔。只输出标签，不要其他文字。"
        )
        
        # 从配置读取 validate 专用参数（与validate_tags使用相同配置）
        val_cfg = cfg.get('validate', {})
        val_temperature = val_cfg.get('temperature', 0.3)
        val_top_p = val_cfg.get('top_p', 0.95)
        val_remove_max_tokens = val_cfg.get('max_tokens', 512)
        
        try:
            response = self._request_with_retry(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                data={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": remove_system},
                        {"role": "user", "content": f"用户描述：{description}\n\n标签列表：{', '.join(tags)}\n\n请找出与上述描述无关的标签，只输出需要保留的相关标签（用逗号分隔）。如果所有标签都无关，则输出 '无'。"}
                    ],
                    "temperature": val_temperature,
                    "top_p": val_top_p,
                    "max_tokens": val_remove_max_tokens
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return tags
            
            result_text = response.json()['choices'][0]['message']['content'].strip()
            
            if result_text == '无' or not result_text.strip():
                return tags
            
            kept_tags = [t.strip() for t in result_text.split(',') if t.strip()]
            original_set = {t.lower() for t in tags}
            return [t for t in kept_tags if t.lower() in original_set]
            
        except Exception:
            return tags


def create_api_client(config: Optional[Dict[str, Any]] = None) -> Optional[APIClient]: # type: ignore
    """从配置创建API客户端
    
    Args:
        config: API配置，如果为None则从config.json加载
        
    Returns:
        APIClient实例，如果未启用API则返回None
    """
    if config is None:
        cfg: Dict[str, Any] = get_config('api') or {}
    else:
        cfg = config
    
    if not cfg.get('enabled', False):
        return None
    
    return APIClient(
        provider=cfg.get('provider'),
        api_key=cfg.get('api_key'),
        base_url=cfg.get('base_url'),
        model=cfg.get('model'),
        temperature=cfg.get('temperature', 1.0),
        max_tokens=cfg.get('max_tokens', 128000),
        system_prompt=cfg.get('system_prompt')
    )