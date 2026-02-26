# Danbooru Tag Generator

将中文描述转换为 Danbooru 英文标签的 AI 绘画提示词生成工具。

## 功能

- **语义搜索**：基于 BGE-M3 向量嵌入 + BGE-Reranker 重排序，从 47,000+ 标签中召回最相关的候选
- **LLM 筛选**：大模型智能挑选最符合描述的标签
- **标签验证**：可选的验证阶段，自动剔除无关标签
- **图形界面**：基于 Flet 的现代化 GUI 应用

## 安装

```bash
pip install -r requirements.txt
```

## 配置

编辑 `config.json`，填写你的 API 密钥。支持 SiliconFlow、DeepSeek 等 OpenAI 兼容接口。

> 密钥也可通过环境变量 `LLM_API_KEY` 设置，避免明文配置。

## 使用

### 图形界面

```bash
# 启动 GUI 应用
python gui.py
```

### 命令行

```bash
# 基本用法
python main.py -d "一位白发少女站在樱花树下"

# 禁用 LLM（仅语义搜索）
python main.py -d "一位白发少女" --no-llm

# 禁用语义搜索（仅 LLM）
python main.py -d "一位白发少女" --no-semantic

# 交互模式（LLM 自动生成场景描述）
python main.py

# 批量生成
python main.py -f descriptions.txt -o output.txt
```

> 首次运行会自动构建嵌入缓存，之后会直接使用缓存。

## 配置选项

| 配置项 | 说明 |
|--------|------|
| `llm.model` | LLM 模型名称 |
| `llm.select_tags.temperature` | 标签筛选温度 |
| `llm.select_tags.top_p` | 标签筛选 top_p |
| `llm.select_tags.max_tokens` | 标签筛选最大 token 数 |
| `semantic_search.embedding.model` | 嵌入模型 |
| `semantic_search.reranker.model` | 重排序模型 |

## 许可证

MIT License
