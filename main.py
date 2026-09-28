"""Command-line runner: python main.py        run the seven test questions
                       python main.py -i     ask questions interactively"""
import asyncio
import sys

from graph import graph


TEST_CASES = [
    "Find recent papers about foundation models for cell segmentation",
    "Which datasets and metrics did CellViT use?",
    "What did I note about HoVer-Net?",
    "What is a Vision Transformer?",
    "Will the stock market go up today?",
    # Need several Agents at once
    "Compare CellViT with my notes on HoVer-Net",
    "What is a Vision Transformer, and what did I note about CellViT?",
]


def format_tasks(tasks: list[dict]) -> str:
    return "\n".join(f"  - [{t['category']}] {t['sub_question']}" for t in tasks)


async def ask(question: str, user_id: int = 1) -> dict:
    return await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "user_id": user_id},
        # Run name, tags and metadata shown in LangSmith, for easier filtering
        config={"run_name": "Research Assistant", "tags": ["cli"], "metadata": {"user_id": user_id}},
    )


async def run_tests():
    graph.get_graph().print_ascii()
    for question in TEST_CASES:
        print(f"\n{'=' * 20} Question: {question} {'=' * 20}")
        result = await ask(question)
        print(f"\n>>> Tasks:\n{format_tasks(result['tasks'])}")
        print(f">>> Final answer:\n{result['final_answer']}")


async def interactive():
    user_id = int(input("Enter user ID (1 / 2 / 99): ") or 1)
    while True:
        question = input("\nEnter a question (press Enter to quit): ").strip()
        if not question:
            break
        result = await ask(question, user_id)
        print(f"\n>>> Tasks:\n{format_tasks(result['tasks'])}\n\n{result['final_answer']}")


if __name__ == "__main__":
    asyncio.run(interactive() if "-i" in sys.argv else run_tests())
