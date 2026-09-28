"""Gradio demo UI: run `python app.py`, then open http://127.0.0.1:7860"""
import time

import gradio as gr

from graph import graph


CATEGORY_NAMES = {
    "search": "🌐 Paper Search",
    "library": "📚 Paper Library",
    "notes": "📝 Reading Notes",
    "concept": "💡 Concept Explanation",
    "other": "🚫 Other",
}


def step(title: str, content: str = "") -> gr.ChatMessage:
    """A collapsible step box; "pending" shows a spinner until the step is marked done."""
    return gr.ChatMessage(content=content, metadata={"title": title, "status": "pending"})


def finish(message: gr.ChatMessage, started: float, content: str, tokens: int) -> None:
    message.content = content
    message.metadata["status"] = "done"
    # Shown next to the title once the step is done. "log" is used instead of "duration",
    # because Gradio's duration can only show seconds; 0 tokens means no model was called (or it failed)
    stats = f"{time.monotonic() - started:.1f}s"
    if tokens:
        stats += f" · {tokens:,} tokens"
    message.metadata["log"] = f"({stats})"


# By default a bubble is only as wide as its content, so collapsed step boxes start out narrow;
# give assistant bubbles the full width from the start
CSS = """
.message-row.bubble.bot-row { width: calc(100% - var(--spacing-xl) * 6); }
"""


# Use astream to show each step in the UI while the graph is running.
# Every yield sends the whole list of messages, so boxes update in place.
async def chat(message, history, user_id):
    started = time.monotonic()
    analyze = step("🔍 Analyzing the question", "Deciding which assistants are needed...")
    messages = [analyze]
    yield messages

    task_boxes: dict[str, tuple[gr.ChatMessage, float]] = {}  # category -> (box, start time)
    summary = None
    total_tokens = 0

    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": message}], "user_id": int(user_id)},
        stream_mode="updates",
        # Run name, tags and metadata shown in LangSmith, for easier filtering
        config={"run_name": "Research Assistant", "tags": ["gradio"], "metadata": {"user_id": int(user_id)}},
    ):
        for update in chunk.values():
            if not update:
                continue

            # Supervisor finished: show the split and its reason, then one box per task, all running at once
            if "tasks" in update:
                names = " · ".join(CATEGORY_NAMES.get(t["category"], t["category"]) for t in update["tasks"])
                total_tokens += update.get("route_tokens", 0)
                finish(
                    analyze, started,
                    f"**Split into:** {names}\n\n**Reason:** {update.get('route_reason', '')}",
                    update.get("route_tokens", 0),
                )
                now = time.monotonic()
                for task in update["tasks"]:
                    box = step(CATEGORY_NAMES.get(task["category"], task["category"]), f"*{task['sub_question']}*")
                    task_boxes[task["category"]] = (box, now)
                    messages.append(box)

            # An Agent finished: update only its own box, the others keep spinning
            for result in update.get("agent_results", []):
                box, box_started = task_boxes[result["category"]]
                total_tokens += result["tokens"]
                finish(box, box_started, f"*{result['sub_question']}*\n\n{result['answer']}", result["tokens"])

            # Every Agent is done: the summary starts
            if summary is None and task_boxes and all(b.metadata["status"] == "done" for b, _ in task_boxes.values()):
                summary = step("✍️ Writing the final answer", "Merging the results...")
                summary_started = time.monotonic()
                messages.append(summary)

            if "final_answer" in update:
                total_tokens += update.get("summary_tokens", 0)
                finish(
                    summary, summary_started,
                    f"Merged the results into the answer below.\n\n**Total:** {total_tokens:,} tokens",
                    update.get("summary_tokens", 0),
                )
                messages.append(gr.ChatMessage(content=update["final_answer"]))

        yield messages


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
        ["Compare CellViT with my notes on HoVer-Net", 1],
        ["What is a Vision Transformer, and what did I note about CellViT?", 1],
    ],
)


if __name__ == "__main__":
    demo.launch(css=CSS)
