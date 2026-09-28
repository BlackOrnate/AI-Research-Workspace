import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Task(TypedDict):
    category: str       # Which Agent handles it: search / library / notes / concept / other
    sub_question: str   # The part of the user's question this Agent should answer


class AgentResult(TypedDict):
    category: str
    sub_question: str
    answer: str
    tokens: int         # Tokens (input + output) this Agent used, shown in the UI


class State(TypedDict):
    # Conversation messages; add_messages appends messages and converts dicts to message objects
    messages: Annotated[list[AnyMessage], add_messages]
    # The fields below are written step by step during the flow, so they are all optional
    user_id: NotRequired[int]        # Current user ID, used when querying reading notes
    tasks: NotRequired[list[Task]]   # Supervisor's split of the question, one task per Agent
    route_reason: NotRequired[str]   # Why the supervisor split the question this way, shown in the UI
    route_tokens: NotRequired[int]   # Tokens the supervisor used, shown in the UI
    # Agents run in parallel and write in the same step, so results are appended with a reducer;
    # without it LangGraph raises InvalidUpdateError when two branches write this key at once
    agent_results: Annotated[list[AgentResult], operator.add]
    final_answer: NotRequired[str]   # Final answer after the summary node merges the results
    summary_tokens: NotRequired[int] # Tokens the summary node used, shown in the UI


class TaskState(TypedDict):
    """Input of an Agent node, sent by Send: only its own task, not the whole State."""
    task: Task
    user_id: NotRequired[int]
