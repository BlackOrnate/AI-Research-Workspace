"""Supervisor: classifies the question on the first pass, then summarizes once an Agent returns."""
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langgraph.graph import END
from pydantic import BaseModel, Field

from agents.common import ask_agent, create_model
from agents.library import library_titles
from state import State


# Category -> next node name
CATEGORY_TO_NODE = {
    "search": "search_node",
    "library": "library_node",
    "notes": "notes_node",
    "concept": "concept_node",
    "other": "reject_node",
}


# Structured output: the model can only return one of these categories
class RouteDecision(BaseModel):
    category: Literal["search", "library", "notes", "concept", "other"] = Field(
        description="Category of the user's question"
    )
    reason: str = Field(description="Brief reason for the classification")


# Classification needs stable output, so temperature is 0.
# The library titles are listed in the prompt, otherwise the router can't tell "library" from "search"
router_agent = create_agent(
    model=create_model(),
    system_prompt=f"""
        You are the router of a research assistant. Classify the user's question into one of these categories:

        - search: find new papers online on a topic, e.g. "find recent papers about X"
        - library: details of a specific paper in the user's local library (method, datasets, metrics, limitations).
          Papers in the local library: {", ".join(library_titles())}
        - notes: the user's own reading list, reading progress, or notes they wrote
        - concept: explain a general concept or term, not tied to a specific paper, e.g. "what is a Vision Transformer"
        - other: questions unrelated to research
    """,
    # Structured output; the result is stored in response["structured_response"]
    response_format=RouteDecision,
)

# Summarizing only reorganizes existing content and needs no creativity, so temperature is 0
summary_agent = create_agent(
    model=create_model(),
    system_prompt="""
        You are the reply-writing assistant of a research workspace.
        You will receive the user's question and the result from a specialist assistant. Turn that result into the final reply:

        1. Respond directly to the user's question in a clear, friendly tone
        2. Keep it well structured; use bullet points when there is a lot of content
        3. Only reorganize information already in the result; do not add, remove or change any facts
           (paper titles, authors, dates, links, datasets, numbers, etc.)
        4. Keep every link from the result unchanged
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


def route_by_category(state: State):
    # Summary is done, end the flow
    if state.get("final_answer"):
        return END

    category = state.get("category")
    if category is None:
        return "reject_node"

    return CATEGORY_TO_NODE.get(category, "reject_node")


def _extract_message_text(content: Any) -> str:
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


async def classify(question: str | list[str | dict[str, Any]]) -> str:
    normalized_question = _extract_message_text(question)
    try:
        response = await router_agent.ainvoke(
            {"messages": [{"role": "user", "content": normalized_question}]}
        )
        decision = response["structured_response"]
        print(f"Category: {decision.category}, reason: {decision.reason}")
        return decision.category
    except Exception as e:
        # Fall back to "other" if the Agent call or structured output fails
        print(f"Classification failed, defaulting to other: {e}")
        return "other"


async def summarize(state: State) -> str:
    question = _extract_message_text(state["messages"][0].content)
    agent_result = state.get("agent_result", "")

    # The polite refusal is already short, no need to summarize it
    if state.get("category") == "other":
        return agent_result

    # If summarizing fails, use the Agent's result directly
    return await ask_agent(
        summary_agent,
        f"User question: {question}\n\nResult:\n{agent_result}",
        fail_text=agent_result,
    )


async def supervisor_node(state: State):
    print("~~~~~~This is Supervisor~~~~~~")

    # Back at the supervisor after an Agent finished: summarize the result
    if state.get("agent_result"):
        final_answer = await summarize(state)
        print(final_answer)
        return {"final_answer": final_answer, "messages": [{"role": "ai", "content": final_answer}]}
    else:
        # First pass through the supervisor: classify the user's question
        category = await classify(state["messages"][0].content)
        return {"category": category}
