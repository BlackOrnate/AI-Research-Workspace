# Changelog

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
