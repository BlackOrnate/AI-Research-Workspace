"""Gradio demo UI: run `python app.py`, then open http://127.0.0.1:7860"""
import gradio as gr

from graph import graph


CATEGORY_NAMES = {
    "search": "Paper Search",
    "library": "Paper Library",
    "notes": "Reading Notes",
    "concept": "Concept Explanation",
    "other": "Other",
}


# Use astream to show progress in the UI while the graph is running
async def chat(message, history, user_id):
    category = ""
    answer = ""

    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": message}], "user_id": int(user_id)},
        stream_mode="updates",
        # Run name, tags and metadata shown in LangSmith, for easier filtering
        config={"run_name": "Research Assistant", "tags": ["gradio"], "metadata": {"user_id": int(user_id)}},
    ):
        for update in chunk.values():
            if not update:
                continue
            if "category" in update:
                category = CATEGORY_NAMES.get(update["category"], update["category"])
                yield f"⏳ Question category: **{category}**, processing..."
            if "final_answer" in update:
                answer = update["final_answer"]

    yield f"**[{category}]**\n\n{answer}"


demo = gr.ChatInterface(
    fn=chat,
    title="🔬 Research Assistant Multi-Agent",
    description="Search papers online, ask about papers in your library, check your reading notes, or ask about research concepts. Switch the user ID below to see a different user's notes.",
    additional_inputs=[
        gr.Dropdown(choices=[1, 2, 99], value=1, label="Current user ID"),
    ],
    examples=[
        ["Find recent papers about foundation models for cell segmentation", 1],
        ["Which datasets and metrics did CellViT use?", 1],
        ["What did I note about HoVer-Net?", 1],
        ["What is a Vision Transformer?", 1],
        ["Will the stock market go up today?", 1],
    ],
)


if __name__ == "__main__":
    demo.launch()
