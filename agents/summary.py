"""Summary: merges the results of every Agent into one final reply."""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from agents.common import ask_agent, create_model, message_text
from state import AgentResult, State


# Summarizing only reorganizes existing content and needs no creativity, so temperature is 0
summary_agent = create_agent(
    model=create_model(),
    system_prompt="""
        You are the reply-writing assistant of a research workspace.
        You will receive the user's question and the results from one or more specialist assistants,
        each under a "### [category] sub-question" heading. Turn them into one final reply:

        1. Answer the user's original question as a whole, combining the results where needed
           (e.g. if the user asks for a comparison, write a comparison)
        2. Keep it well structured and friendly; use bullet points when there is a lot of content
        3. Only reorganize information already in the results; do not add, remove or change any facts
           (paper titles, authors, dates, links, datasets, numbers, etc.)
        4. Keep every link from the results unchanged
        5. If a result says that part is unavailable, or declines the question, tell the user that part
           could not be answered; do not fill it in with other results or your own knowledge
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


def ordered_results(state: State) -> list[AgentResult]:
    """Parallel branches may finish in any order; sort results back into the order of the tasks."""
    order = {task["category"]: i for i, task in enumerate(state["tasks"])}
    return sorted(state["agent_results"], key=lambda r: order.get(r["category"], len(order)))


def format_results(results: list[AgentResult]) -> str:
    return "\n\n".join(f"### [{r['category']}] {r['sub_question']}\n{r['answer']}" for r in results)


async def summarize(state: State) -> tuple[str, int]:
    """Return the final reply and the tokens used to write it."""
    results = ordered_results(state)

    # The polite refusal is already short, no need to summarize it
    if all(r["category"] == "other" for r in results):
        return "\n\n".join(r["answer"] for r in results), 0

    question = message_text(state["messages"][0].content)
    formatted = format_results(results)
    # If summarizing fails, return every Agent's result under its own heading
    return await ask_agent(
        summary_agent,
        f"User question: {question}\n\nResults:\n{formatted}",
        fail_text=formatted,
    )


async def summary_node(state: State):
    print("~~~~~~This is Summary~~~~~~")
    final_answer, tokens = await summarize(state)
    print(final_answer)
    return {"final_answer": final_answer, "summary_tokens": tokens, "messages": [{"role": "ai", "content": final_answer}]}
