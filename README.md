# Danbooru Tag Generator

将自然语言描述（中文/英文）转换为 Danbooru 标签的 AI 绘画提示词生成工具。

## 功能

- **语义搜索**：基于 BGE-M3 向量嵌入 + BGE-Reranker 重排序，从 47,000+ 标签中召回最相关的候选
- **LLM 筛选**：大模型智能挑选最符合描述的标签
- **标签验证**：可选的验证阶段，自动剔除无关标签
- **图形界面**：基于 Flet 的现代化 GUI 应用
- **单语自动补全**：当输入仅中文或仅英文时，自动翻译补全为双语后再匹配标签
- **实时日志同步**：终端输出实时同步到网页日志窗格，并可一键复制

## 安装

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

> 已包含 `flet-web==0.26.0`，用于浏览器模式 GUI 运行。

## 配置

编辑 `config.json`，填写你的 API 密钥。支持 SiliconFlow、DeepSeek 等 OpenAI 兼容接口。

> 密钥也可通过环境变量 `LLM_API_KEY` 设置，避免明文配置。

## 使用

### 图形界面

```bash
# 方式1：直接启动
uv run --python .venv/bin/python gui.py

# 方式2：推荐（Windows）
# 自动创建虚拟环境、安装依赖（使用 uv + 中科大镜像），然后启动 GUI
start_gui.bat
```

#### GUI 主要操作

- **生成自然语言描述**：调用 LLM 先生成场景描述并回填输入框
- **生成标签**：使用当前输入描述直接生成标签
- **生成自然语言描述和标签**：先生成描述，再自动继续生成标签
- **大语言模型思考模式**：可分别控制“筛选标签 / 生成描述 / 验证标签”
- **运行日志**：固定窗格、自动滚动到最新行、支持复制完整日志

### 命令行

```bash
# 基本用法
uv run --python .venv/bin/python main.py -d "一位白发少女站在樱花树下"

# 禁用 LLM（仅语义搜索）
uv run --python .venv/bin/python main.py -d "一位白发少女" --no-llm

# 禁用语义搜索（仅 LLM）
uv run --python .venv/bin/python main.py -d "一位白发少女" --no-semantic

# 交互模式（LLM 自动生成场景描述）
uv run --python .venv/bin/python main.py

# 批量生成
uv run --python .venv/bin/python main.py -f descriptions.txt -o output.txt
```

> Windows 下将 `.venv/bin/python` 替换为 `.venv\\Scripts\\python.exe`。

> 首次运行会自动构建嵌入缓存，之后会直接使用缓存。

## 配置

- 配置文件：`config.json`
- 模板文件：`config.example.json`
- GUI 的“配置”页支持编辑 `config.json` 的所有条目（按类型渲染：字符串输入框、布尔开关、数字输入等）

常用配置项：

| 配置项 | 说明 |
|--------|------|
| `llm.model` | LLM 模型名称 |
| `llm.select_tags.temperature` | 标签筛选温度 |
| `llm.select_tags.top_p` | 标签筛选 top_p |
| `llm.select_tags.max_tokens` | 标签筛选最大 token 数 |
| `llm.thinking.select_tags` | 是否在标签筛选阶段启用 thinking |
| `llm.thinking.generate` | 是否在生成描述阶段启用 thinking |
| `llm.thinking.validate` | 是否在标签验证阶段启用 thinking |
| `semantic_search.embedding.model` | 嵌入模型 |
| `semantic_search.reranker.model` | 重排序模型 |
| `generator.auto_generate_prompt` | 自动生成自然语言描述的提示词模板 |
| `generator.auto_generate_max_chinese_chars` | 自动生成中文描述长度上限 |

## 许可证

MIT License
