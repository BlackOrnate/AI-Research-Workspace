from langchain_ollama import ChatOllama

from config import CHAT_MODEL


def create_model(temperature: float = 0) -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, temperature=temperature)


async def ask_agent(agent, question: str, fail_text: str, context=None) -> str:
    """Ask an Agent a question and return its last reply; return a fallback message if it still fails after retries."""
    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            context=context,
        )
        # The Agent returns the whole conversation; the last message is the model's reply
        return response["messages"][-1].content
    except Exception as e:
        print(f"Agent call failed, using fallback message: {e}")
        return fail_text


def agent_update(answer: str) -> dict:
    """Return value of an Agent node: store the result and append the answer to the conversation."""
    print(answer)
    return {"agent_result": answer, "messages": [{"role": "ai", "content": answer}]}
