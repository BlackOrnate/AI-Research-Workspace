# Research Assistant Multi-Agent

**English** | [简体中文](README.zh-CN.md)

A learning project that builds a **Supervisor-pattern multi-agent system** for research work with [LangGraph](https://github.com/langchain-ai/langgraph). The Supervisor splits the user's question into tasks, sends them to specialist Agents that run in parallel, and a summary node merges their results into the final reply.

The project grows one version at a time. Each version adds one idea and is tagged on GitHub, so you can follow how the design evolves. See [CHANGELOG.md](CHANGELOG.md) and [docs/](docs/).

Each specialist Agent demonstrates a different technique:

| Agent | Handles | Technique |
| --- | --- | --- |
| `search` | Finding papers online | **Tool calling** against the public OpenAlex API |
| `library` | Details of papers in the local library | **RAG** over a Markdown paper library with **FAISS** |
| `notes` | The user's own reading list and notes | Tool calling + **MCP** Server, `user_id` injected through runtime context |
| `concept` | General research concepts and terms | Plain LLM, no tools |
| `reject` | Questions unrelated to research | Polite refusal |

All models run locally through [Ollama](https://ollama.com). LangSmith tracing is optional.

> The paper summaries and reading notes in this repo are simplified mock data for learning. Always check the original papers.

## Architecture

```text
START -> supervisor_node (split into tasks)
           └─ Send × N (1-4, in parallel)
                ├─ search_node   (Tool: OpenAlex)   ─┐
                ├─ library_node  (RAG + FAISS)      ─┤
                ├─ notes_node    (Tools + MCP)      ─┼-> summary_node (merge) -> END
                ├─ concept_node  (LLM only)         ─┤
                └─ reject_node   (polite refusal)   ─┘
```

The Supervisor uses structured output (`RouteDecision`) to split the question into 1-4 tasks, each with a category and a sub-question, at most one task per category. `dispatch_tasks` turns them into `Send` calls, so the Agents run in the same step and each receives only its own task. Their results are appended to `agent_results` through an `operator.add` reducer, and `summary_node` merges them into one reply.

## Project Structure

```text
├── config.py            Model names, file paths, OpenAlex and MCP settings
├── state.py             Shared graph State
├── mcp_server.py        Reading notes MCP Server (port 8001)
├── agents/
│   ├── common.py        Helpers: create model, call Agent, node return value
│   ├── supervisor.py    Split the question into tasks + dispatch them with Send
│   ├── summary.py       Merge the Agents' results into the final reply
│   ├── search.py        Paper search Agent (OpenAlex API)
│   ├── library.py       Paper library Agent (RAG)
│   ├── notes.py         Reading notes Agent (via MCP)
│   ├── concept.py       Concept explanation Agent
│   └── reject.py        Polite refusal
├── graph.py             Builds the StateGraph
├── main.py              Command-line runner
├── app.py               Gradio demo UI
└── data/
    └── papers.md        Local paper library (mock summaries)
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally with these models:

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text-v2-moe
```

### Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env            # optional: LangSmith tracing
```

### Run

Start the MCP Server first (the `notes` Agent needs it), then in another terminal:

```bash
python mcp_server.py      # terminal 1
python main.py            # terminal 2: run the seven test questions
python main.py -i         # or ask questions interactively
python app.py             # or open the Gradio UI at http://127.0.0.1:7860
```

The FAISS index is built from `data/papers.md` on first run. Delete `faiss_index_papers/` after editing the paper library to rebuild it.

## Example Questions

| Question | Route |
| --- | --- |
| Find recent papers about foundation models for cell segmentation | `search` |
| Which datasets and metrics did CellViT use? | `library` |
| What did I note about HoVer-Net? (user 1) | `notes` |
| What is a Vision Transformer? | `concept` |
| Will the stock market go up today? | `other` |
| Compare CellViT with my notes on HoVer-Net | `library` + `notes` |
| What is a Vision Transformer, and what did I note about CellViT? | `concept` + `notes` |

Mock users for `notes`: user `1` and user `2` have reading notes, user `99` has none.

## Known Limitations (v0.2)

- Agents run in parallel but can't use each other's results. A question like "Find new papers related to what's on my to-read list" needs `notes` first and then `search`, which isn't supported yet.
- When merging results, the summary can still add small inferences that are not in any Agent's result.
- No conversation memory: every question is answered on its own.
- Search results are ranked by relevance, which favors highly cited older papers, so "recent" in a question is ignored.
- If the search API fails, the Agent tells the user honestly instead of retrying.
