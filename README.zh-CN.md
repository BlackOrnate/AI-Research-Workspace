# 科研助手多智能体

[English](README.md) | **简体中文**

一个用 [LangGraph](https://github.com/langchain-ai/langgraph) 构建**Supervisor 模式多智能体系统**的科研场景学习项目。Supervisor 把用户问题拆分成若干任务，交给多个专业 Agent 并行处理，最后由汇总节点把各 Agent 的结果合并成最终回复。

项目按版本逐步迭代，每个版本只引入一个新想法，并在 GitHub 上打 tag，方便观察设计是怎样一步步演进的。详见 [CHANGELOG.md](CHANGELOG.md) 和 [docs/](docs/)。

每个专业 Agent 演示一种不同的技术：

| Agent       | 负责内容                 | 技术                                                             |
| ----------- | ------------------------ | ---------------------------------------------------------------- |
| `search`  | 在线查找论文             | **工具调用**：调用公开的 OpenAlex API                      |
| `library` | 某篇具体论文的细节       | **子图**，确定性路由：本地元数据里有的论文走按 `paper_id` 过滤的 **FAISS** **RAG**，其他论文走在线检索 |
| `notes`   | 用户自己的阅读列表和笔记 | 工具调用 +**MCP** Server，`user_id` 通过运行时上下文注入 |
| `concept` | 通用科研概念和术语       | 纯 LLM，不带工具                                                 |
| `reject`  | 与科研无关的问题         | 礼貌拒绝                                                         |

所有模型都通过 [Ollama](https://ollama.com) 在本地运行，LangSmith 追踪为可选项。

> 仓库中的论文摘要和阅读笔记均为教学用的简化模拟数据，具体细节请以论文原文为准。
>
> 项目中的代码、提示词和数据均为英文，向模型提问时也建议使用英文。

## 架构

```text
START -> supervisor_node（拆分任务）
           └─ Send × N（1~4 个，并行）
                ├─ search_node   (工具: OpenAlex)   ─┐
                ├─ library_node  (RAG + FAISS)      ─┤
                ├─ notes_node    (工具 + MCP)       ─┼-> summary_node（合并汇总） -> END
                ├─ concept_node  (纯 LLM)           ─┤
                └─ reject_node   (礼貌拒绝)          ─┘
```

Supervisor 通过结构化输出（`RouteDecision`）把问题拆成 1~4 个任务，每个任务包含类别和子问题，每个类别最多一个。`dispatch_tasks` 把这些任务变成 `Send`，各 Agent 在同一步并行运行，并且只收到自己的任务。结果通过 `operator.add` reducer 追加到 `agent_results`，最后由 `summary_node` 合并成一个回答。

## 目录结构

```text
├── config.py            模型名称、文件路径、OpenAlex 和 MCP 配置
├── state.py             全局 State
├── mcp_server.py        阅读笔记 MCP Server（端口 8001）
├── agents/
│   ├── common.py        公共方法：创建模型、调用 Agent、节点返回值
│   ├── supervisor.py    拆分任务 + 用 Send 分派
│   ├── summary.py       把各 Agent 的结果合并成最终回复
│   ├── search.py        论文检索 Agent（OpenAlex API）
│   ├── library.py       论文详情 Agent：解析 -> 本地 RAG 或在线检索 -> 回答
│   ├── paper_resolver.py  这篇论文在不在本地库里？（标题 / 别名匹配）
│   ├── notes.py         阅读笔记 Agent（通过 MCP 查询）
│   ├── concept.py       概念解释 Agent
│   └── reject.py        礼貌拒绝
├── graph.py             组装 StateGraph
├── main.py              命令行运行
├── app.py               Gradio 演示界面
├── evals/
│   ├── router_cases.jsonl  带标注的问题集：每个问题应该分到哪些类别
│   ├── eval_router.py      路由评测：准确率、各类别的 precision/recall
│   ├── library_cases.jsonl 带标注的问题集：每个问题应该用哪个来源
│   └── eval_library.py     library Agent 评测：有没有用对来源
└── data/
    ├── papers.md        本地论文库（模拟摘要）
    └── papers.json      论文元数据：paper_id、标题、别名
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
python main.py            # 终端 2：运行 7 个测试问题
python main.py -i         # 或者交互式提问
python app.py             # 或者打开 Gradio 界面 http://127.0.0.1:7860
python -m evals.eval_router --runs 3   # 路由评测（不需要 MCP Server）
python -m evals.eval_library          # library Agent 评测（在线题需要联网）
```

首次运行时会根据 `data/papers.md` 构建 FAISS 索引。修改论文库后，删除 `faiss_index_papers/` 即可重建。`papers.md` 里的每篇论文都要在 `papers.json` 中有一条记录，其中的别名决定了用哪些名字能找到它。

## 示例问题

| 问题                                                             | 路由                    |
| ---------------------------------------------------------------- | ----------------------- |
| Find recent papers about foundation models for cell segmentation | `search`              |
| Which datasets and metrics did CellViT use?                      | `library`             |
| What did I note about HoVer-Net?（用户 1）                       | `notes`               |
| What is a Vision Transformer?                                    | `concept`             |
| Will the stock market go up today?                               | `other`               |
| Compare CellViT with my notes on HoVer-Net                       | `library` + `notes` |
| What is a Vision Transformer, and what did I note about CellViT? | `concept` + `notes` |

`notes` 的模拟用户：用户 `1` 和用户 `2` 有阅读笔记，用户 `99` 没有。

## 已知局限（v0.3）

- 只描述、不点名的问题（例如 "Which paper uses star-convex polygons?"）找不到本地论文，会转去在线检索。
- 在线检索直接使用整句问题，有时搜不到目标论文。
- "Compare CellViT and Mask R-CNN" 这种本地和未知论文混在一起的问题，只会走本地，未知论文那部分会丢失。
- Agent 之间只能并行，不能使用彼此的结果。例如"根据我的待读列表找相关新论文"需要先查 `notes` 再 `search`，目前还不支持。
- 合并结果时，汇总仍可能加入一些不在任何 Agent 结果里的小推断。
- 没有对话记忆，每个问题都独立回答。
- 检索结果按相关性排序，会偏向高引用的老论文，问题里的 "recent" 会被忽略。
- 检索 API 失败时，Agent 会如实告诉用户，不做重试。
