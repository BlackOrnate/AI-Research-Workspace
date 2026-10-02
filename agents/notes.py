"""Reading notes Agent: queries the current user's reading list and notes through the MCP Server."""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.tools import tool, ToolRuntime
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.common import agent_update, ask_agent, create_model
from config import MCP_URL
from state import TaskState

mcp_client = MultiServerMCPClient(
    {"workspace": {"url": MCP_URL, "transport": "streamable_http"}}
)
_mcp_query_notes = None


async def get_mcp_query_notes():
    """Connect to the MCP Server and fetch the tool on first use, then reuse it."""
    global _mcp_query_notes
    if _mcp_query_notes is None:
        mcp_tools = await mcp_client.get_tools()
        _mcp_query_notes = next(t for t in mcp_tools if t.name == "query_reading_notes")
    return _mcp_query_notes


# Runtime context: user_id is passed in by the program, not filled in by the model
@dataclass
class UserContext:
    user_id: int


@tool
async def query_reading_notes(runtime: ToolRuntime[UserContext]) -> str:
    """Query the current user's reading list and notes, sorted from most recent to oldest."""
    mcp_tool = await get_mcp_query_notes()
    # Call the MCP tool with the user_id passed in by the program, so the model can't read other users' notes
    result = await mcp_tool.ainvoke({"user_id": runtime.context.user_id})
    # The MCP tool returns a list of content blocks: [{"type": "text", "text": "..."}]
    return "\n".join(block["text"] for block in result if block["type"] == "text")


NOTES_FAIL_TEXT = "Sorry, your reading notes are temporarily unavailable. Please make sure the MCP Server is running and try again."

notes_agent = create_agent(
    model=create_model(),
    tools=[query_reading_notes],
    system_prompt="""
        You are the reading notes assistant of a research workspace. You answer questions about the user's
        own reading list and notes, such as which papers they have read, what is still on their to-read list,
        and what they noted about a paper.

        First call query_reading_notes to get the current user's notes, then answer based on the results.
        Notes are sorted from most recent to oldest, so "last" / "most recent" refers to the first record.

        Answer only from the query results and never make up notes; if there are no notes, tell the user honestly.
    """,
    context_schema=UserContext,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def notes_node(state: TaskState):
    print("~~~~~~This is Notes Agent~~~~~~")
    task = state["task"]
    user_id = state.get("user_id")
    if user_id is None:
        raise ValueError("user_id is required to query reading notes")
    answer, tokens = await ask_agent(
        notes_agent,
        task["sub_question"],
        NOTES_FAIL_TEXT,
        # Pass the user_id sent by the supervisor to the tool
        context=UserContext(user_id=user_id),
    )
    return agent_update(task, answer, tokens)
