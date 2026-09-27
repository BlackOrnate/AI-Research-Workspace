from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Conversation messages; add_messages appends messages and converts dicts to message objects
    messages: Annotated[list[AnyMessage], add_messages]
    # The fields below are written step by step during the flow, so they are all optional
    user_id: NotRequired[int]        # Current user ID, used when querying reading notes
    category: NotRequired[str]       # Supervisor's classification result
    agent_result: NotRequired[str]   # Result produced by the Agent node
    final_answer: NotRequired[str]   # Final answer after the supervisor summarizes
