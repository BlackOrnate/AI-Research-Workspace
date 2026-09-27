"""Concept Agent: explains general research concepts and terms, without tools."""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from agents.common import agent_update, ask_agent, create_model
from state import State


CONCEPT_FAIL_TEXT = "Sorry, concept explanations are temporarily unavailable. Please try again later."

concept_agent = create_agent(
    model=create_model(),
    system_prompt="""
        You are the concept tutor of a research workspace. You explain general concepts and terms in
        machine learning, computer vision and biomedical imaging, such as "What is a Vision Transformer?"
        or "What does Panoptic Quality measure?".

        When explaining:
        1. Start with a one-sentence definition
        2. Then explain the key idea in plain language, with a simple example if it helps
        3. Keep it short, around 150-250 words

        You have no access to papers or search, so do not quote specific numbers, results or paper details.
        If you are not sure about something, say so instead of guessing.
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def concept_node(state: State):
    print("~~~~~~This is Concept Agent~~~~~~")
    answer = await ask_agent(concept_agent, state["messages"][0].content, CONCEPT_FAIL_TEXT)
    return agent_update(answer)
