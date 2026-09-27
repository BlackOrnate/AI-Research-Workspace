# Changelog

## v0.1 - Supervisor + 4 specialist Agents

First version, modeled on a medical consultation multi-agent learning project.

- Supervisor classifies each question with structured output and summarizes the Agent's result
- `search`: finds papers through the OpenAlex API (switched from arXiv, see docs/v0.1.md), merges versions of the same paper
- `library`: RAG over a local Markdown paper library with FAISS
- `notes`: reads the user's reading notes through an MCP Server, `user_id` injected via runtime context
- `concept`: explains general research concepts without tools
- `reject`: politely declines unrelated questions
- Command-line runner (`main.py`) and Gradio UI (`app.py`)
