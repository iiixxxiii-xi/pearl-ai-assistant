# 🦪 珍珠 AI 助手（Pearl AI Assistant）

> 为家庭珍珠生意定制的 Agentic RAG 系统。客服回复 + 库存推荐 + 内容生成，一个知识库。
>
> **真实商业场景 × 库存管理 × 混合检索 × LangChain × ReAct Agent × Redis 缓存。**

---

## 📱 产品

手机网页，打开即用。三个功能模块：

| 模块 | 用户 | 场景 |
|------|------|------|
| 💬 客服回复 | 妈妈 | 填客户画像 → AI 结合知识库+库存自动推荐珠子+成本 → 审核 → 复制发送 |
| ✍️ 内容生成 | 姐姐 | 输入话题 → AI 生成小红书标题 + 正文 + 标签 |
| 📦 库存管理 | 妈妈 | 查看/出入库/新增珠子品种，无需编辑 JSON |

**AI 不直接面对客户。** 人是决策者，AI 是工具。

---

## 🏗️ 系统架构

```
                    ┌──────────────────────┐
                    │  纯 HTML/CSS/JS 前端   │
                    │  三 Tab · 手机适配     │
                    └──────────┬───────────┘
                               │ POST /api/reply/stream
                               │ POST /api/content
                               │ GET/POST /api/inventory/*
                               ▼
                    ┌──────────────────────┐
                    │   FastAPI 服务         │
                    │                       │
                    │  ┌─ 客户画像策略系统 ─┐│
                    │  │ 4维度各配策略       ││
                    │  ├─ ReAct 工具调度引擎 ─┤│
                    │  │ Thought→Action→Observe ││
                    │  ├─ 库存管理系统 ────┤│
                    │  │ JSON持久化·出入库    ││
                    │  ├─ 四层幻觉治理 ────┤│
                    │  │ 短路·自检·验证·改写 ││
                    │  ├─ LangChain 检索链 ─┤│
                    │  │ LCEL·Chroma封装    ││
                    │  ├─ 语义对话记忆 ────┤│
                    │  └─ 全链路指标追踪 ──┘│
                    └──┬──────┬──────┬─────┘
                       │      │      │
              ┌────────┘      │      └────────┐
              ▼               ▼               ▼
     ┌────────────┐  ┌──────────────┐  ┌──────────┐
     │ ChromaDB   │  │ PostgreSQL   │  │  Redis   │
     │ 向量检索    │  │ 对话+反馈    │  │ 检索缓存  │
     └────────────┘  └──────────────┘  └──────────┘
```

---

## 🔧 技术栈

| 组件 | 选择 | 理由 |
|------|------|------|
| 后端 | FastAPI | 现代异步 |
| 向量库 | ChromaDB | 轻量嵌入，零运维 |
| 关系库 | PostgreSQL | 对话记忆 + 反馈日志持久化 |
| 缓存 | Redis | 检索结果缓存，自动降级内存 |
| LLM | DeepSeek API | 国内直连，成本低 |
| Embedding | sentence-transformers | 免费本地，中文效果好 |
| AI 框架 | LangChain（薄封装） | Chroma + LCEL 检索链；Agent ReAct 循环自己写 |
| 容器 | Docker Compose | 一键部署 |

---

## ✨ 技术点

### 检索链路
| # | 技术点 | 说明 |
|---|--------|------|
| 1 | **Q&A 按空行切块** | 知识库按 Q&A 对用 `\n\n` 分隔，每块独立完整，无需 overlap |
| 2 | **混合检索** | BM25 关键词 + 向量语义双路召回 |
| 3 | **RRF 融合** | Reciprocal Rank Fusion 合并两路结果 |
| 4 | **DeepSeek Rerank** | LLM 精排 Top-3；103 条小知识库下候选少，复用 DeepSeek API 避免额外部署模型。规模增长后切 BGE-reranker |
| 5 | **Query 改写+扩展** | LLM 口语转书面 + 年龄自动映射年龄段关键词 + 预算数字映射知识库范围关键词（边界值双匹配） |

### Agentic RAG
| # | 技术点 | 说明 |
|---|--------|------|
| 6 | **ReAct 工具调度** | LLM 自主决定调哪些工具、调几次、什么顺序，替换硬编码调用 |
| 7 | **多轮检索 + 硬终止** | 最多 3 轮工具调用，超限后 force reply；每轮 LLM 决策 + 工具执行在 1-2 秒内完成 |
| 8 | **置信度短路** | 知识库无匹配时 LLM 诚实告知，不编造，自动跳过低置信度场景 |

### 幻觉治理
| # | 技术点 | 说明 |
|---|--------|------|
| 9 | **结构化输入** | 表单 4 维度画像降低用户输入模糊性 |
| 10 | **归因约束** | Prompt 要求标注引用的知识库条目 |
| 11 | **归因交叉验证** | 代码级解析归因行，检测引用了不存在的条目 |
| 12 | **Negative Rejection** | 知识库无相关内容正确拒答 |

### 库存与推荐
| # | 技术点 | 说明 |
|---|--------|------|
| 13 | **库存管理系统** | JSON 持久化 + 跨进程文件锁 + 原子写入（tmp→rename），多 worker 并发不坏数据 |
| 14 | **珠子推荐** | LLM 结合库存自动推荐珠子，实时计算珠子成本 |
| 15 | **前端库存管理** | 新增品种/出入库/数量输入，无需编辑 JSON 或重启服务 |

### 工程化
| # | 技术点 | 说明 |
|---|--------|------|
| 16 | **流式输出 SSE** | DeepSeek stream=True → 线程转异步 → SSE 推送 → 前端打字机 |
| 17 | **Redis 检索缓存** | 检索结果缓存 5 分钟，命中直接返回跳过全链路 |
| 18 | **LangChain 检索链** | LCEL 构建 Chroma + ChatOpenAI 链，Agent 部分自己写 |
| 19 | **ReAct 工具调度层** | 原生 Python 实现 Thought→Action→Observation 循环，LLM 自主决策 |
| 20 | **语义对话记忆** | jieba 关键词重合度 + 时间衰减，非简单滑动窗口 |
| 21 | **全链路指标** | `/api/metrics` 查检索缓存命中率 + Agentic RAG + 采纳率 |
| 22 | **检索日志持久化** | JSONL 结构化日志，支持 bad case 回溯 |
| 23 | **RAGAS 风格评估** | LLM-as-Judge 四指标自动评分 |
| 24 | **采纳率追踪** | copy/regenerate 人机协作反馈闭环 |
| 25 | **双角色 Prompt** | 妈妈/姐姐两套 System Prompt 切换 |
| 26 | **客户画像策略** | 4 维度（性格/用途/品质/了解程度）各配策略 |
| 27 | **单元测试** | 84 个测试：预算提取 27 个（自然语言正则全覆盖）+ 库存过滤 4 个 + 年龄段映射 16 个 + 预算范围扩展 12 个 + Query 类别提取 5 个 + 知识库关键词 4 个 + 检索/API/库存/RRF/记忆 16 个 |
| 28 | **Docker Compose** | FastAPI + PostgreSQL + Redis 一键编排 |
| 29 | **自动降级** | Redis/PG 不可用时自动回退内存/JSON，不加 Docker 也能跑 |

---

## 📂 项目结构

```
pearl_ai_assistant/
├── app.py                       # FastAPI 主程序（路由·ReAct Agent·库存 API）
├── rag.py                       # 检索链路（BM25·向量·RRF·Rerank·改写·缓存）
├── prompts.py                   # System Prompt + 客户画像策略
├── models.py                    # Pydantic 请求/响应模型
├── inventory.py                 # 库存管理系统（JSON持久化·出入库）
├── cache.py                     # Redis 缓存（自动降级内存）
├── db.py                        # PostgreSQL 持久化（自动降级 JSON）
├── feedback.py                  # 采纳率追踪
├── langchain_chain.py           # LangChain LCEL 检索链封装
├── evaluate.py                  # RAG 全链路评估脚本（LLM-as-Judge）
├── build_knowledge.py           # 知识库向量化构建
├── test_app.py                  # 测试（17个·检索·API·库存·RRF·记忆）
├── templates/index.html         # 手机网页前端（单文件·三Tab）
├── data/
│   ├── pearl_knowledge.txt      # 知识库源文件
│   ├── inventory.json           # 珠子库存（15种·实时更新）
├── chroma_db/                   # 向量数据库（自动生成）
├── Dockerfile                   # 应用容器化
├── docker-compose.yml           # 多服务编排（FastAPI+PG+Redis）
├── entrypoint.sh                # 容器启动脚本
├── requirements.txt             # Python 依赖
└── .env                         # API Key 配置
```

---

## 🚀 快速启动

### 本地开发（不需要 Docker）

```bash
pip install -r requirements.txt
cp .env.example .env          # 编辑填入 DEEPSEEK_API_KEY
python build_knowledge.py     # 首次运行 / 知识库更新后
python app.py                 # http://localhost:8000
```

### Docker 部署

```bash
docker compose up -d          # 一键启动 FastAPI + Redis + PostgreSQL
```

### 运行测试

```bash
python -m pytest test_app.py -v    # 17 个测试
```

### 运行评估

```bash
python evaluate.py            # 全链路评估
python evaluate.py --verbose  # 打印每条详情
```

---

## 🔍 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/reply` | POST | 客服回复（非流式） |
| `/api/reply/stream` | POST | 客服回复（SSE 流式） |
| `/api/content` | POST | 内容生成 |
| `/api/inventory` | GET | 查看全部库存 |
| `/api/inventory/update` | POST | 新增/更新珠子品种 |
| `/api/inventory/stock-in` | POST | 入库 |
| `/api/inventory/stock-out` | POST | 出库/售出 |
| `/api/feedback` | POST | 记录采纳/重新生成 |
| `/api/stats` | GET | 采纳率统计 |
| `/api/metrics` | GET | 全链路指标（检索 + Agentic RAG + 采纳） |

---

## 👤 关于

杭州师范大学 计算机科学与技术

独立开发，从需求分析到部署上线全流程负责。家庭珍珠生意真实场景——知识库里是妈妈 30 年的珍珠经验，不是公开数据集。

*"别人的 RAG 项目跑在公开数据上，我的跑在我家的生意上。"*
