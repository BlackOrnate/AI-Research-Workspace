# 科研助手多智能体

[English](README.md) | **简体中文**

一个用 [LangGraph](https://github.com/langchain-ai/langgraph) 构建**Supervisor 模式多智能体系统**的科研场景学习项目。Supervisor 对用户问题进行分类，路由到对应的专业 Agent 处理，最后把 Agent 的结果汇总成最终回复。

项目按版本逐步迭代，每个版本只引入一个新想法，并在 GitHub 上打 tag，方便观察设计是怎样一步步演进的。详见 [CHANGELOG.md](CHANGELOG.md) 和 [docs/](docs/)。

每个专业 Agent 演示一种不同的技术：

| Agent | 负责内容 | 技术 |
| --- | --- | --- |
| `search` | 在线查找论文 | **工具调用**：调用公开的 OpenAlex API |
| `library` | 本地论文库中论文的细节 | 基于 Markdown 论文库的 **RAG**，使用 **FAISS** 向量库 |
| `notes` | 用户自己的阅读列表和笔记 | 工具调用 + **MCP** Server，`user_id` 通过运行时上下文注入 |
| `concept` | 通用科研概念和术语 | 纯 LLM，不带工具 |
| `reject` | 与科研无关的问题 | 礼貌拒绝 |

所有模型都通过 [Ollama](https://ollama.com) 在本地运行，LangSmith 追踪为可选项。

> 仓库中的论文摘要和阅读笔记均为教学用的简化模拟数据，具体细节请以论文原文为准。
>
> 项目中的代码、提示词和数据均为英文，向模型提问时也建议使用英文。

## 架构

```text
START -> supervisor_node（分类）
           ├─ search  -> search_node   (工具: OpenAlex)   ─┐
           ├─ library -> library_node  (RAG + FAISS)      ─┤
           ├─ notes   -> notes_node    (工具 + MCP)       ─┼-> supervisor_node（汇总） -> END
           ├─ concept -> concept_node  (纯 LLM)           ─┤
           └─ other   -> reject_node   (礼貌拒绝)          ─┘
```

Supervisor 节点会经过两次。第一次通过结构化输出（`RouteDecision`）对问题分类；Agent 处理完返回后，State 中已有 `agent_result`，Supervisor 就对结果进行汇总，然后路由到 `END`。

## 目录结构

```text
├── config.py            模型名称、文件路径、OpenAlex 和 MCP 配置
├── state.py             全局 State
├── mcp_server.py        阅读笔记 MCP Server（端口 8001）
├── agents/
│   ├── common.py        公共方法：创建模型、调用 Agent、节点返回值
│   ├── supervisor.py    分类 + 汇总 + 路由函数
│   ├── search.py        论文检索 Agent（OpenAlex API）
│   ├── library.py       本地论文库 Agent（RAG）
│   ├── notes.py         阅读笔记 Agent（通过 MCP 查询）
│   ├── concept.py       概念解释 Agent
│   └── reject.py        礼貌拒绝
├── graph.py             组装 StateGraph
├── main.py              命令行运行
├── app.py               Gradio 演示界面
└── data/
    └── papers.md        本地论文库（模拟摘要）
```

## 快速开始

### 前提条件

- Python 3.11+
- 本地已启动 [Ollama](https://ollama.com)，并已下载以下模型：

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text-v2-moe
```

### 安装

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env            # 可选：LangSmith 追踪
```

### 运行

先启动 MCP Server（`notes` Agent 依赖它），再在另一个终端运行：

```bash
python mcp_server.py      # 终端 1
python main.py            # 终端 2：运行 5 个测试问题
python main.py -i         # 或者交互式提问
python app.py             # 或者打开 Gradio 界面 http://127.0.0.1:7860
```

首次运行时会根据 `data/papers.md` 构建 FAISS 索引。修改论文库后，删除 `faiss_index_papers/` 即可重建。

## 示例问题

| 问题 | 路由 |
| --- | --- |
| Find recent papers about foundation models for cell segmentation | `search` |
| Which datasets and metrics did CellViT use? | `library` |
| What did I note about HoVer-Net?（用户 1） | `notes` |
| What is a Vision Transformer? | `concept` |
| Will the stock market go up today? | `other` |

`notes` 的模拟用户：用户 `1` 和用户 `2` 有阅读笔记，用户 `99` 没有。

## 已知局限（v0.1）

- 每个问题只会交给一个 Agent。需要多个 Agent 配合的问题（例如"把 CellViT 和我对 HoVer-Net 的笔记做个对比"）目前处理不好。
- 没有对话记忆，每个问题都独立回答。
- 检索结果按相关性排序，会偏向高引用的老论文，问题里的 "recent" 会被忽略。
- 检索 API 失败时，Agent 会如实告诉用户，不做重试。
