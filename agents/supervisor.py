"""Supervisor: splits the question into tasks and sends each task to its Agent, all in parallel."""
from typing import Literal

from langchain.agents import create_agent
from langgraph.types import Send
from pydantic import BaseModel, Field

from agents.common import count_tokens, create_model, message_text
from agents.library import library_titles
from state import State, Task


# Category -> Agent node name
CATEGORY_TO_NODE = {
    "search": "search_node",
    "library": "library_node",
    "notes": "notes_node",
    "concept": "concept_node",
    "other": "reject_node",
}


class TaskItem(BaseModel):
    category: Literal["search", "library", "notes", "concept", "other"] = Field(
        description="Which assistant handles this part of the question"
    )
    sub_question: str = Field(
        description="The part of the user's question for this assistant, rewritten as a standalone English question"
    )


# Structured output: a list of tasks instead of a single category
class RouteDecision(BaseModel):
    tasks: list[TaskItem] = Field(
        min_length=1, max_length=4, description="One task per assistant that is needed"
    )
    reason: str = Field(description="Brief reason for the split")


# Classification needs stable output, so temperature is 0.
# The library titles are listed in the prompt, otherwise the router can't tell "library" from "search"
router_agent = create_agent(
    model=create_model(),
    system_prompt=f"""
        You are the router of a research assistant. Decide which assistants are needed to answer the user's question:

        - search: find new papers online on a topic, e.g. "find recent papers about X"
        - library: details of a specific paper in the user's local library (method, datasets, metrics, limitations).
          Papers in the local library: {", ".join(library_titles())}
        - notes: the user's own reading list, reading progress, or notes they wrote
        - concept: explain a general concept or term, not tied to a specific paper, e.g. "what is a Vision Transformer"
        - other: questions unrelated to research

        Rules:
        1. Most questions need only ONE assistant. Only split when the question clearly asks about
           several different things, e.g. "Compare CellViT with my notes on HoVer-Net" needs library and notes
        2. Use each category at most once. The library assistant can look up several papers at once,
           so "Compare CellViT and HoVer-Net" is a single library task
        3. For each task, write a sub_question that contains only the part this assistant should answer,
           as a standalone English question
    """,
    # Structured output; the result is stored in response["structured_response"]
    response_format=RouteDecision,
)


def normalize_tasks(items: list[TaskItem], question: str) -> list[Task]:
    """Keep the first task of each category, and fall back to the whole question if a sub-question is empty.

    "One task per category" is a fixed rule, so it is enforced in code instead of trusting the prompt alone.
    """
    tasks: list[Task] = []
    seen: set[str] = set()
    for item in items:
        if item.category in seen:
            continue
        seen.add(item.category)
        tasks.append({"category": item.category, "sub_question": item.sub_question.strip() or question})
    return tasks or [{"category": "other", "sub_question": question}]


async def split_question(question: str) -> tuple[list[Task], str, int]:
    """Return the tasks, the router's reason for splitting the question this way, and the tokens used."""
    try:
        response = await router_agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        decision = response["structured_response"]
        tasks = normalize_tasks(decision.tasks, question)
        print(f"Tasks: {[t['category'] for t in tasks]}, reason: {decision.reason}")
        return tasks, decision.reason, count_tokens(response["messages"])
    except Exception as e:
        # Fall back to "other" if the Agent call or structured output fails
        print(f"Routing failed, defaulting to other: {e}")
        return [{"category": "other", "sub_question": question}], "Routing failed, so the question was treated as unrelated.", 0


async def supervisor_node(state: State):
    print("~~~~~~This is Supervisor~~~~~~")
    tasks, reason, tokens = await split_question(message_text(state["messages"][0].content))
    return {"tasks": tasks, "route_reason": reason, "route_tokens": tokens}


def dispatch_tasks(state: State) -> list[Send]:
    """Send every task to its Agent node; nodes sent in the same step run in parallel.

    Each node gets only its own task and the user_id, not the whole State.
    """
    tasks = state.get("tasks", [])
    return [
        Send(CATEGORY_TO_NODE[task["category"]], {"task": task, "user_id": state.get("user_id")})
        for task in tasks
    ]
