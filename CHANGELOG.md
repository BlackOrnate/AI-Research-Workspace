# Changelog

## v0.3 - Evals and local-first paper lookup

Every change is now measured with an eval, and whether a paper is in the local library is decided in code instead of by the model. See docs/v0.3.md.

- Router eval: 34 labeled questions in `evals/router_cases.jsonl`, run with `python -m evals.eval_router [--runs N]`
  - Reports exact match of the category set (single-task and multi-task separately), per-category precision/recall, stability across runs, and the wrong cases with their sub-questions
  - Baseline with `gemma4:e4b`, 3 runs: 98% exact match; "Tell me about the Mask R-CNN paper" (not in the library) goes to `library` in 2 of 3 runs
- `library` vs `search` is now split by intent, not by data source
  - `library`: details of any named paper; `search`: finding papers on a topic or similar to a paper
  - The router prompt no longer lists the library titles; no prompt does, so it doesn't grow with the library
  - Router eval after the change (Mask R-CNN relabeled as `library`, one case added): 35/35 in all 3 runs
- Library Agent: local library first, online search as the fallback (version 1 of the local-first routing plan)
  - Paper metadata in `data/papers.json` (`paper_id`, title, aliases) is kept apart from the chunks in FAISS:
    the metadata decides *which paper*, the vector store decides *which passages*
  - `agents/paper_resolver.py`: `resolve_local_paper(query)` matches titles and aliases in code, ignoring case.
    A match must be a whole name, so "SAM" doesn't match "MedSAM" or "Cellpose-SAM", and "CellViT" doesn't match "CellViT++"
  - The library Agent is now a LangGraph subgraph with a conditional edge instead of a tool-calling Agent:
    `resolve_local -> local_rag | external_retrieval -> answer`. The model only writes the answer
  - Local RAG retrieves only from the resolved papers (FAISS filter on `paper_id`), so other papers can't crowd them out.
    Chunks carry their `paper_id`; an older index without it is rebuilt on start
  - The online search reuses the search Agent's `search_papers` tool
- Library Agent eval: 18 labeled questions in `evals/library_cases.jsonl`, run with `python -m evals.eval_library`
  - Checks the source from the subgraph state: which local papers were found, and whether the online search was used
  - 16/18. The two misses describe a library paper without naming it, which title / alias matching can't find (planned for version 2)
- `message_text()` also handles a single content block given as a dict; the summary no longer fails when `tasks` is missing
- Fix: OpenAlex reads `?` and `*` as wildcards and answered 400 for any query that is a question; they are now removed from the query

## v0.2 - Parallel multi-Agent collaboration

One question can now be split across several Agents that run in parallel. See docs/v0.2.md.

- Supervisor splits the question into 1-4 tasks, each with its own sub-question, at most one per category
- Tasks are dispatched with LangGraph `Send` and run in the same step; results are merged with an `operator.add` reducer
- New `summary_node` merges all results; the Supervisor no longer runs twice
- Agent nodes receive only their own task (`TaskState`) and no longer write to `messages`
- A failed branch no longer hides the others: the summary says which part could not be answered
- `main.py` shows the task split; two multi-Agent test questions added
- Gradio UI shows each step as a live box: analyzing (with the router's reason), one box per task updating on its own, then the summary
- Each step box shows its time and token usage, and the summary box shows the total tokens
- Step boxes are full width from the start instead of only after being expanded
- Paper search falls back to arXiv when OpenAlex fails (e.g. its daily limit is reached)

## v0.1 - Supervisor + 4 specialist Agents

First version, modeled on a medical consultation multi-agent learning project.

- Supervisor classifies each question with structured output and summarizes the Agent's result
- `search`: finds papers through the OpenAlex API (switched from arXiv, see docs/v0.1.md), merges versions of the same paper
- `library`: RAG over a local Markdown paper library with FAISS
- `notes`: reads the user's reading notes through an MCP Server, `user_id` injected via runtime context
- `concept`: explains general research concepts without tools
- `reject`: politely declines unrelated questions
- Command-line runner (`main.py`) and Gradio UI (`app.py`)
