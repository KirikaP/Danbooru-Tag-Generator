"""
命令行界面模块
提供单次生成和批量生成功能
"""

import sys
from pathlib import Path
from typing import Optional
from .generator import PromptGenerator, create_generator
from .api_client import create_api_client, APIClient
from .config import get_config


class SimpleCLI:
    """简化的命令行界面 - 适合单次使用"""
    
    def __init__(self, db_path: str, use_space_separator: bool = True, api_client: Optional[APIClient] = None, use_semantic: bool = False, semantic_config: Optional[dict] = None):
        self.db_path = db_path
        self.api_client = api_client
        self.use_semantic = use_semantic
        self.semantic_tagger = None
        
        # 如果启用语义搜索，尝试加载语义标签器
        if use_semantic:
            try:
                from .semantic_search import create_semantic_tagger
                # 构建语义搜索配置（传递完整配置，支持解耦的embedding和reranker）
                if semantic_config:
                    config = semantic_config  # 传递完整的semantic_search配置
                else:
                    config = {}
                self.semantic_tagger = create_semantic_tagger(db_path, config)
                print("[SimpleCLI] 语义搜索已启用")
            except Exception as e:
                print(f"[SimpleCLI] 无法启用语义搜索: {e}")
                print("[SimpleCLI] 回退到API生成")
                self.use_semantic = False
        
        # 创建生成器（纯API/语义模式）
        self.generator = create_generator(
            db_path, 
            use_space_separator=use_space_separator, 
            api_client=api_client,
            semantic_tagger=self.semantic_tagger
        )
    
    def generate(self, description: Optional[str]) -> str:
        """生成提示词
        
        Args:
            description: 图片描述
        """
        return self.generator.generate(description, use_semantic=self.use_semantic)
    
    def enable_api(self):
        """启用API"""
        return self.generator.enable_api()
    
    def is_api_enabled(self) -> bool:
        """检查API是否启用"""
        return self.generator.is_api_enabled()
