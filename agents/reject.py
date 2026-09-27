"""Other questions: politely decline."""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from agents.common import agent_update, ask_agent, create_model
from state import State


REJECT_TEXT = (
    "Sorry, I'm a research assistant and can only help with research-related questions, "
    "such as searching for papers, asking about papers in your library, checking your reading notes, "
    "or explaining research concepts. Is there anything research-related I can help you with?"
)

reject_agent = create_agent(
    # Allow a little variation in the refusal so it reads more naturally
    model=create_model(temperature=0.3),
    system_prompt="""
        You are a research assistant and only handle research-related questions,
        including: searching for papers, asking about papers in the user's library,
        checking the user's reading notes, and explaining research concepts.

        The user's question is outside your scope. Politely decline in one or two sentences and guide the user toward a research-related question.
        Do not answer the content of the user's question itself.
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def reject_node(state: State):
    print("~~~~~~This is Reject Agent~~~~~~")
    answer = await ask_agent(reject_agent, state["messages"][0].content, REJECT_TEXT)
    return agent_update(answer)
