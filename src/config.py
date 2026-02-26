#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配置管理模块"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件"""
    global _config
    
    if _config is not None and config_path is None:
        return _config
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config.json' # type: ignore
    else:
        config_path = Path(config_path) # type: ignore
    
    if not config_path.exists(): # type: ignore
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f: # type: ignore
        _config = json.load(f)
    
    return _config # type: ignore


def get_config(key: Optional[str] = None) -> Any:
    """获取配置项"""
    config = load_config()
    
    if key is None:
        return config
    
    keys = key.split('.')
    value = config
    for k in keys:
        value = value.get(k)
        if value is None:
            return None
    return value


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """重新加载配置"""
    global _config
    _config = None
    return load_config(config_path)