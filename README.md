# Agentic RAG · 电商智能助手

一个覆盖 **RAG、Router、Planner、Tool Calling、Memory、MCP、可观测性、评测** 八大能力的
端到端 Agentic RAG 项目，业务场景为「电商智能客服 / 运营分析」。

> 功能蓝图参考腾讯开源的 [WeKnora](https://github.com/Tencent/WeKnora)，基于主流 RAG/Agent 技术栈独立实现。

## 一、能力总览

| 能力 | 实现 | 关键文件 |
|------|------|---------|
| **RAG** | 文档切分 → 向量(Chroma) + BM25 混合检索 → LLM 重排 | `ingest/` `retrieval/` |
| **Router** | 把查询路由到 `rag / data / analytics / chitchat` 四条路径 | `agent/router.py` |
| **Planner** | 复杂分析问题拆解为多步可执行计划 | `agent/planner.py` |
| **Tool Calling** | 原生 OpenAI function-calling 工具循环（订单/商品/客户/退款/聚合） | `agent/tools.py` |
| **Memory** | 短期（对话历史）+ 长期（用户画像 JSON 持久化 + LLM 抽取） | `agent/memory.py` |
| **MCP** | 客户端（Agent 调外部工具）+ 服务端（知识库暴露给 Claude/Cursor） | `mcp/` |
| **可观测性** | 自研 SQLite Span 追踪（路由/规划/LLM/工具 每步的耗时与 Token） | `observability/` |
| **评测** | 自定义检索指标（Hit/MRR/Recall）+ RAGAS（LLM-as-judge） | `eval/` |

## 二、架构

```
用户查询
   │
   ▼
load_memory ──► router ──(analytics)──► planner
   │                │                      │
   │           (rag/data/chitchat)         │
   │                └──────────┬───────────┘
   │                           ▼
   │                     agent (LLM 决策：调工具 or 回答)
   │                           │  tool_calls
   │                           ▼
   │                      tools (RAG检索 / 数据查询 / MCP工具)
   │                           │
   │                           └────► 回到 agent（循环，直至给出答案）
   ▼
finalize（保存长期记忆）──► 答案 + 可观测 trace
```

Agent 用 LangGraph 组装，工具循环走原生 OpenAI function-calling，不依赖 LangChain 的
Agent 封装——底层细节透明。

## 三、目录结构

```
agentic-rag/
├── data/
│   ├── seed/           # 数据：152 条法律条文 + 2.8 万天猫订单 + 商品/客户/退款
│   └── eval/           # 评测数据集（检索 + QA）
├── src/agentic_rag/
│   ├── config.py       # 配置（pydantic-settings，读 .env）
│   ├── llm.py          # OpenAI 兼容 LLM / Embedding 封装
│   ├── data_store.py   # 内存数据仓库（结构化查询 + 聚合）
│   ├── schema.py       # Document / Chunk / RetrievalResult
│   ├── ingest/         # 加载 / 分块 / 建索引
│   ├── retrieval/      # BM25 / 向量 / 混合检索 / LLM 重排
│   ├── agent/          # Router / Planner / Tools / Memory / 状态图
│   ├── mcp/            # MCP 客户端 / 服务端 / 外部演示服务
│   ├── observability/  # 轻量追踪
│   ├── eval/           # 检索评测 + RAGAS
│   ├── api.py          # FastAPI（REST + SSE）
│   └── cli.py          # 命令行入口
└── tests/              # 无需 API 的单元测试
```

## 四、快速开始

```bash
# 1. 安装依赖（建议 Python 3.11+）
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e .                  # 可编辑安装，同时注册 agentic-rag 命令

# 2. 配置
cp .env.example .env              # 填入 LLM 与 Embedding 的 API Key

# 3. 建索引（需先配好 Embedding）
agentic-rag ingest                # 或 python -m agentic_rag.cli ingest

# 4. 对话
agentic-rag chat                  # 交互式
agentic-rag chat "退货政策是什么"   # 单次
```

### 配置说明（重要）

**DeepSeek 不提供 embedding 接口**，所以 LLM 与 Embedding 需要分开配：

| 用途 | 示例 | 说明 |
|------|------|------|
| LLM | DeepSeek `deepseek-chat` | 走 `LLM_BASE_URL` |
| Embedding | 硅基流动 `BAAI/bge-large-zh-v1.5` | 走 `EMBEDDING_BASE_URL` |

其它可选的 Embedding：阿里云 DashScope `text-embedding-v3`、智谱 `embedding-3`、OpenAI
`text-embedding-3-small`，都在 `.env.example` 里有注释。

## 五、各能力使用示例

```bash
# 检索：直接看混合检索 + 重排效果
agentic-rag search "网购商品几天内可以无理由退货"

# 评测：检索指标（无需 LLM）
agentic-rag eval
# 评测：再加 RAGAS（需 pip install -r requirements-eval.txt）
agentic-rag eval --ragas

# 可观测：查看最近 trace
agentic-rag traces
agentic-rag traces --trace-id <trace_id>

# API 服务
agentic-rag serve        # http://localhost:8000  (POST /chat, /chat/stream)
```

### MCP

```bash
# 服务端：把知识库暴露给 Claude / Cursor
agentic-rag mcp-server                       # stdio
agentic-rag mcp-server --transport streamable-http --port 9000

# 客户端：Agent 调用外部 MCP 工具（内置物流跟踪演示）
agentic-rag chat --mcp "查一下订单 ORD-2024-1008 的物流"
```

外部演示 MCP 服务在 `mcp/demo_server.py`，提供 `track_shipment` / `check_inventory` 两个工具，
Agent 会在回答物流/库存问题时通过 MCP 调用它。

## 六、电商业务逻辑

- **结构化数据**（2.8 万条真实天猫订单 + 商品 / 客户 / 退款）→ Agent 走 **工具调用** 查询。
- **非结构化文档**（《消费者权益保护法》《电子商务法》152 条条文）→ Agent 走 **RAG 检索** 回答。
- **Router** 把两类问题自动分流，**Planner** 处理「哪些商品退款率最高、原因是什么」这类
  需要多步查询 + 检索 + 分析的综合问题。

典型问题举例：

| 问题 | 路由 |
|------|------|
| 网购商品几天内可以无理由退货？ | rag |
| 电商平台能删除消费者评价吗？ | rag |
| 查一下 C001 的订单 | data |
| 你好 | chitchat |

## 七、技术栈

`openai`（原生 SDK）· `langgraph`（状态机）· `chromadb`（向量）· `rank-bm25`（关键词）·
`mcp`（客户端/服务端）· `fastapi` + `sse-starlette`（API）· `ragas`（评测）·
`pydantic` + `pydantic-settings`（配置/模型）

## 八、设计要点

1. **为什么 LLM 和 Embedding 要分开配**：DeepSeek 等模型只有 chat 接口，无 embedding。
2. **混合检索怎么融合**：向量 + BM25 分数归一化加权，`alpha` 控制权重；也可换 RRF。
3. **中文 BM25 分词**：无空格，用字符二元组；生产可换 jieba/IK。
4. **重排为什么用 LLM**：免额外模型、走同一接口；生产可换 bge-reranker。
5. **Router/Planner 的代价**：每步都是额外 LLM 调用，延迟与成本换能力。
6. **何时停止工具循环**：`max_iterations` 兜底 + 模型自行判断。
7. **长期记忆的抽取与更新**：LLM 从对话抽取画像，规则约束「只记稳定信息」。
8. **可观测性**：Span 树记录每一步的耗时/Token，可换 Langfuse。

## 九、评测结果

自建 16 条评测集（152 条条文粒度）：

| 指标 | 纯混合检索 | 加 LLM 重排 |
|------|-----------|------------|
| Hit@3 | 87.5% | 100% |
| MRR@3 | 0.84 | 0.93 |
| Recall@3 | 87.5% | 100% |

生成层用 RAGAS 评测，faithfulness ≈ 0.99（无幻觉）、context_recall = 1.0。
