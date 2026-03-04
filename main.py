#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Danbooru Tag Generator - 主程序入口
基于Danbooru标签数据库的文生图提示词生成工具
启动GUI界面
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import gui


if __name__ == '__main__':
    gui.main()