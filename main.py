#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Danbooru Tag Generator - 主程序入口
基于Danbooru标签数据库的文生图提示词生成工具
支持大模型API增强
"""

import sys
import argparse
from typing import Optional
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.cli import SimpleCLI
from src.config import load_config, get_config
from src.api_client import create_api_client


DEFAULT_DB_PATH = Path(__file__).parent / 'danbooruTags' / 'tags_enhanced.csv'
DEFAULT_CONFIG_PATH = Path(__file__).parent / 'config.json'


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Danbooru Tag Generator - 文生图提示词生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                               # 启动交互模式（默认启用语义搜索 + LLM）
  python main.py -d "一位可爱女孩"               # 生成标签（默认启用语义搜索 + LLM）
  python main.py -d "一位可爱女孩" --no-llm      # 只使用语义搜索
  python main.py -d "一位可爱女孩" --no-semantic # 只使用LLM
  python main.py --no-llm --no-semantic        # 纯关键词匹配
  python main.py -d "一位可爱女孩" --underscore  # 使用下划线分隔
  python main.py -f descriptions.txt           # 批量生成
        """
    )
    
    parser.add_argument(
        '-d', '--description',
        type=str,
        help='图片描述文本'
    )
    

    
    parser.add_argument(
        '--db',
        type=str,
        default=str(DEFAULT_DB_PATH),
        help='数据库文件路径 (默认: danbooruTags/tags_enhanced.csv)'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='从文件批量读取描述 (每行一个描述)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出结果到文件'
    )
    
    parser.add_argument(
        '--underscore', '--use-underscore',
        action='store_true',
        help='使用下划线分隔标签（默认使用空格）'
    )
    
    # 语义搜索模式（默认开启）
    parser.add_argument(
        '--no-semantic',
        action='store_true',
        dest='no_semantic',
        help='禁用语义搜索，使用纯关键词匹配'
    )
    
    # LLM相关参数（默认开启）
    parser.add_argument(
        '--no-llm',
        action='store_true',
        dest='no_llm',

        help='禁用LLM，使用本地匹配'
    )
    
    parser.add_argument(
        '--llm-key',
        type=str,
        help='LLM API密钥（也可在config.json或环境变量LLM_API_KEY中设置）'
    )
    
    parser.add_argument(
        '--llm-model',
        type=str,
        help='LLM模型名称'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help='配置文件路径 (默认: config.json)'
    )
    
    parser.add_argument(
        '--escape-parentheses',
        action='store_true',
        default=True,
        help='将括号替换为转义形式 \\(, \\) (默认: 开启)'
    )
    
    parser.add_argument(
        '--no-escape-parentheses',
        action='store_false',
        dest='escape_parentheses',
        help='不替换括号'
    )
    
    args = parser.parse_args()
    
    # 加载配置文件
    if Path(args.config).exists():
        try:
            config = load_config(args.config)
            print(f"已加载配置: {args.config}")
        except Exception as e:
            print(f"配置加载失败: {e}")
            config = {}
    else:
        print(f"配置文件不存在: {args.config}，将使用默认配置")
        config = {}
    
    # 检查数据库文件
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"错误: 数据库文件不存在: {db_path}")
        print("请使用 --db 参数指定正确的数据库路径")
        sys.exit(1)
    
    # 默认使用空格分隔，使用 --underscore 改为下划线分隔
    use_space = not args.underscore
    
    # 确定是否使用LLM（默认开启）
    use_llm = True
    if args.no_llm:
        use_llm = False
    
    # 确定是否使用语义搜索（默认开启）
    use_semantic = True
    if args.no_semantic:
        use_semantic = False
    
    # 创建API客户端（如果需要）
    api_client = None
    if use_llm:
        from src.api_client import APIClient
        llm_config = config.get('llm', {})
        
        # 命令行参数可以覆盖配置文件
        api_client = APIClient(
            provider=llm_config.get('provider', 'openai'),
            api_key=args.llm_key or llm_config.get('api_key', ''),
            base_url=llm_config.get('base_url', 'https://api.openai.com/v1'),
            model=args.llm_model or llm_config.get('model', 'gpt-4o-mini'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 1000),
            system_prompt=llm_config.get('system_prompt')
        )
        
        if not api_client.is_available():
            from src.api_client import REQUESTS_AVAILABLE
            if not REQUESTS_AVAILABLE:
                print("LLM客户端不可用：未安装 requests 库")
                print("   运行: pip install requests")
            else:
                print("LLM客户端不可用：未配置 API 密钥，请在 config.json 中设置 llm.api_key")
            api_client = None
            use_llm = False
        else:
            print(f"LLM已启用 (Provider: {api_client.provider}, Model: {api_client.model})")
    
    # 执行生成
    try:
        from src.cli import SimpleCLI
        
        # 语义搜索配置
        semantic_config = config.get('semantic_search', {})
        # 传递llm配置用于向后兼容（embedding/reranker的api_key默认值的来源）
        semantic_config['llm'] = config.get('llm', config.get('api', {}))
        
        cli = SimpleCLI(str(db_path), use_space_separator=use_space, api_client=api_client, use_semantic=use_semantic, semantic_config=semantic_config)
        
        def escape_parentheses(text: str) -> str:
            """将括号替换为转义形式 \\(, \\)"""
            return text.replace('(', '\\(').replace(')', '\\)')
        
        if args.file:
            # 批量从文件生成
            with open(args.file, 'r', encoding='utf-8') as f:
                descriptions = [line.strip() for line in f if line.strip()]
            
            results = [cli.generate(desc) for desc in descriptions]
            
            # 可选：转义括号
            if args.escape_parentheses:
                results = [escape_parentheses(r) for r in results]
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result}")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(results))
                print(f"\n已保存到: {args.output}")
                
        elif args.description or use_llm:
            # 单次生成（或无API描述时使用API自动生成）
            description_to_use: Optional[str] = args.description if args.description else None
            result = cli.generate(description_to_use)
            
            # 可选：转义括号
            if args.escape_parentheses:
                result = escape_parentheses(result)
            
            print(f"\n生成的提示词:\n{result}")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"\n已保存到: {args.output}")
                
        else:
            # 无参数时显示帮助
            print("Usage: python main.py -d '描述内容' [--semantic] [--llm]")
            print("示例: python main.py -d '一位蓝眼少女' --semantic")
            return
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()