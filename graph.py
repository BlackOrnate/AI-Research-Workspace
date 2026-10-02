from langgraph.graph import END, START, StateGraph

from agents.concept import concept_node
from agents.library import library_node
from agents.notes import notes_node
from agents.reject import reject_node
from agents.search import search_node
from agents.summary import summary_node
from agents.supervisor import CATEGORY_TO_NODE, dispatch_tasks, supervisor_node
from state import State


def build_graph():
    builder = StateGraph(State)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("search_node", search_node)
    builder.add_node("library_node", library_node)
    builder.add_node("notes_node", notes_node)
    builder.add_node("concept_node", concept_node)
    builder.add_node("reject_node", reject_node)
    builder.add_node("summary_node", summary_node)

    # START -> supervisor splits the question -> Agents run in parallel -> summary merges -> END
    builder.add_edge(START, "supervisor_node")
    builder.add_conditional_edges(
        "supervisor_node", dispatch_tasks, list(CATEGORY_TO_NODE.values())
    )
    # summary_node runs once, after every Agent sent in the same step has finished
    for node in CATEGORY_TO_NODE.values():
        builder.add_edge(node, "summary_node")
    builder.add_edge("summary_node", END)

    return builder.compile()


graph = build_graph()
graph.get_graph().print_ascii()
