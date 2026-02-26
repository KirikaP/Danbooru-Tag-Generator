# Danbooru Tag Generator

基于 Danbooru 标签数据库的文生图提示词生成工具。输入中文描述，通过**语义搜索 + LLM 筛选**两阶段流程，自动输出适用于 AI 绘画的 Danbooru 英文标签组合。

## 工作原理

```
中文描述
   │
   ▼ 阶段一：语义搜索
   │  对标签库进行向量嵌入 + 重排序，召回最相关的 80-100 个候选标签
   │  支持中英文双语查询，自动缓存嵌入向量
   │
   ▼ 阶段二：LLM 筛选
   │  将候选标签列表交给大模型，从中挑选最符合描述的标签
   │  支持 thinking 模式提升筛选质量
   │
   ▼ 阶段三：标签验证（可选）
   │  验证并移除与描述无关的标签
   │
   ▼
英文 Danbooru 标签列表
```

## 目录结构

```
DanbooruTagGenerator/
├── main.py                       # 主程序入口
├── config.json                   # 配置文件（复制 config.example.json 修改）
├── config.example.json           # 配置文件模板
├── requirements.txt              # Python 依赖
├── .gitignore                    # Git 忽略文件
├── danbooruTags/
│   └── tags_enhanced.csv         # Danbooru 标签数据库
├── cache/                        # 嵌入向量缓存（运行后自动生成）
└── src/
    ├── api_client.py             # 大模型 API 客户端（带重试和 spinner）
    ├── build_embedding_cache.py  # 预构建嵌入缓存脚本
    ├── cli.py                    # 命令行界面
    ├── config.py                 # 配置加载模块
    ├── generator.py              # 提示词生成器核心
    └── semantic_search.py        # 语义搜索模块（支持双语 + 重排序）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install requests numpy pandas
```

### 2. 配置

**方式一：复制配置模板（推荐）**

```bash
# Windows
copy config.example.json config.json

# Linux/Mac
cp config.example.json config.json
```

然后编辑 `config.json`，填写你的 API 信息（支持任何 OpenAI 兼容接口，如 SiliconFlow、DeepSeek 等）。

**方式二：直接使用 config.json**

确保配置文件包含必要字段（参考 config.example.json）。

> **安全提示**：`api_key` 也可通过环境变量 `LLM_API_KEY` 或 `API_KEY` 设置，避免将密钥提交到 Git。

### 3. 生成提示词（首次会自动构建缓存）

```bash
# 单次生成（默认同时启用语义搜索 + LLM）
python main.py -d "一位白发少女，站在樱花树下"

# 禁用 LLM，只使用语义搜索
python main.py -d "一位白发少女，站在樱花树下" --no-llm

# 禁用语义搜索，只使用 LLM
python main.py -d "一位白发少女，站在樱花树下" --no-semantic

# 让 LLM 自动生成场景描述再生成标签
python main.py

# 批量生成（每行一个描述）
python main.py -f descriptions.txt -o output.txt

# 使用下划线分隔标签（默认空格）
python main.py -d "猫耳少女" --underscore
```

> **注意**：首次运行时会自动构建嵌入缓存，可能需要一些时间。后续运行会直接使用缓存。

### 4. （可选）预构建嵌入缓存

如果想提前构建缓存，可以运行：

```bash
python src/build_embedding_cache.py
```

支持断点续传，中断后重新运行会从上次进度继续。

## 命令行参数

| 参数 | 说明 |
|------|------|
| `-d`, `--description` | 图片描述文本 |
| `-f`, `--file` | 从文件批量读取描述（每行一条） |
| `-o`, `--output` | 将结果输出到文件 |
| `--no-semantic` | 禁用语义搜索，使用纯关键词匹配 |
| `--no-llm` | 禁用 LLM，使用本地匹配 |
| `--llm-key` | 命令行指定 API 密钥（覆盖 config.json） |
| `--llm-model` | 命令行指定模型名（覆盖 config.json） |
| `--underscore`, `--use-underscore` | 使用下划线分隔标签（默认空格） |
| `--db` | 指定数据库文件路径 (默认: danbooruTags/tags_enhanced.csv) |
| `--config` | 指定配置文件路径 (默认: config.json) |
| `--escape-parentheses` | 将括号替换为转义形式 \(, \) (默认: 开启) |
| `--no-escape-parentheses` | 不替换括号 |

## 配置说明

所有参数均在 `config.json` 中管理：

### 基础配置

| 配置项 | 说明 |
|--------|------|
| `database.path` | 标签数据库文件路径 |
| `database.encoding` | 数据库文件编码 |

### Generator 配置

| 配置项 | 说明 |
|--------|------|
| `generator.max_tags` | 最终输出的最大标签数 |
| `generator.auto_tag` | 是否启用自动标签 |
| `generator.auto_generate_max_chinese_chars` | 自动生成描述的中文字符限制 |
| `generator.auto_generate_prompt` | 自动生成场景描述的提示词 |

### 语义搜索配置

| 配置项 | 说明 |
|--------|------|
| `semantic_search.enabled` | 是否启用语义搜索 |
| `semantic_search.embedding` | 嵌入模型配置（provider、api_key、api_url、model） |
| `semantic_search.reranker` | 重排序模型配置（provider、api_key、api_url、model、enabled） |
| `semantic_search.top_k` | 每次查询取 top-k 个结果 |
| `semantic_search.limit` | 候选标签总数上限 |
| `semantic_search.popularity_weight` | 标签热度权重 (0.0-1.0) |
| `semantic_search.similarity_threshold` | 相似度阈值 |
| `semantic_search.timeout` | API 请求超时（秒） |
| `semantic_search.max_retries` | 最大重试次数 |
| `semantic_search.backoff_factor` | 重试退避因子 |

### LLM 配置

| 配置项 | 说明 |
|--------|------|
| `llm.enabled` | 是否启用 LLM 筛选 |
| `llm.provider` | API 提供商 |
| `llm.api_key` | API 密钥 |
| `llm.base_url` | API 基础 URL |
| `llm.model` | 模型名称 |
| `llm.temperature` | LLM 生成温度 |
| `llm.max_tokens` | 最大 token 数 |
| `llm.select_tags_max` | LLM 筛选阶段的最大标签数 |
| `llm.system_prompt` | LLM 标签筛选的系统提示词，支持 `{select_tags_max}` 占位符 |
| `llm.select_tags_temperature` | LLM 标签筛选阶段的温度 |
| `llm.select_tags_max_tokens` | LLM 标签筛选阶段的 token 上限 |
| `llm.thinking.select_tags` | 标签筛选时是否启用 thinking 模式 |
| `llm.thinking.generate` | 生成描述时是否启用 thinking 模式 |
| `llm.thinking.validate` | 验证标签时是否启用 thinking 模式 |
| `llm.timeout` | LLM API 请求超时（秒） |
| `llm.validate_prompt` | 标签验证的系统提示词 |
| `llm.remove_tags_prompt` | 删除无关标签的系统提示词 |

## 特性说明

### 两阶段生成流程

1. **语义搜索阶段**
   - 支持中英文双语查询
   - BGE-M3 嵌入模型
   - BGE-Reranker 重排序
   - 自动构建和加载嵌入缓存

2. **LLM 筛选阶段**
   - 从候选标签中智能选择
   - 支持 thinking 模式提升质量
   - 动态 spinner 显示进度
   - 自动重试机制

3. **标签验证（可选）**
   - 验证标签是否符合描述
   - 自动移除无关标签

## 安全建议

1. **API 密钥安全**
   - 使用环境变量 `LLM_API_KEY` 设置密钥
   - 不要将 `config.json` 提交到 Git
   - 使用 `config.example.json` 作为模板

2. **缓存管理**
   - `cache/` 目录会自动生成
   - 可以随时删除缓存重新构建

## 故障排除

- **首次运行很慢**：正常，首次需要构建嵌入缓存
- **API 调用失败**：检查网络连接和 API 密钥
- **语义搜索无结果**：检查 `semantic_search.enabled` 配置
- **LLM 不工作**：检查 `llm.enabled` 配置和 API 密钥

## 许可证

本项目仅供学习和研究使用。
