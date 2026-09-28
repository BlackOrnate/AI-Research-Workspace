from typing import Any

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from config import CHAT_MODEL
from state import Task


def create_model(temperature: float = 0) -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, temperature=temperature)


def message_text(content: Any) -> str:
    """Message content can be a string or a list of content blocks; return it as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
    return str(content)


def count_tokens(messages: list) -> int:
    """Total tokens (input + output) of every model call in an Agent's conversation."""
    return sum((m.usage_metadata or {}).get("total_tokens", 0) for m in messages if isinstance(m, AIMessage))


async def ask_agent(agent, question: str, fail_text: str, context=None) -> tuple[str, int]:
    """Ask an Agent a question and return its last reply and the tokens it used.

    Return a fallback message (and 0 tokens) if it still fails after retries.

    Catching every exception matters even more with parallel Agents: if one node raises,
    the whole parallel step fails and the other branches' results are lost too.
    """
    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            context=context,
        )
        # The Agent returns the whole conversation; the last message is the model's reply
        return message_text(response["messages"][-1].content), count_tokens(response["messages"])
    except Exception as e:
        print(f"Agent call failed, using fallback message: {e}")
        return fail_text, 0


def agent_update(task: Task, answer: str, tokens: int) -> dict:
    """Return value of an Agent node: append the result to agent_results.

    Agent nodes don't write messages; only the summary node adds the final reply,
    so parallel answers don't get interleaved into the conversation.
    """
    print(f"[{task['category']}] {answer}")
    return {"agent_results": [{**task, "answer": answer, "tokens": tokens}]}
