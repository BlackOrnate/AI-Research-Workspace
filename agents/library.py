"""Paper library Agent: answers questions about specific papers, local library first.

A small LangGraph subgraph; the local-vs-online decision is made in code, not by the model:

    resolve_local -> route_source -> local_rag          -> answer
                                  -> external_retrieval -> answer
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.graph import END, START, StateGraph

from agents.common import agent_update, ask_agent, create_model
from agents.paper_resolver import paper_metadata, resolve_local_paper
from agents.search import search_papers
from config import EMBEDDING_MODEL, INDEX_PATH, LIBRARY_TOP_K, PAPERS_PATH
from state import PaperState, TaskState


def split_papers() -> list[Document]:
    """Split the library on "## <paper>" headings, and tag each chunk with its paper_id from the metadata."""
    text = TextLoader(str(PAPERS_PATH), encoding="utf-8").load()[0].page_content
    # strip_headers=False keeps the title in the chunk, which helps retrieval
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "paper")],
        strip_headers=False,
    )
    # Drop the intro section at the top that has no paper heading
    chunks = [c for c in text_splitter.split_text(text) if "paper" in c.metadata]

    paper_ids = {paper["title"]: paper["paper_id"] for paper in paper_metadata}
    for chunk in chunks:
        # Every paper in papers.md needs an entry in papers.json, or it could never be found
        if chunk.metadata["paper"] not in paper_ids:
            raise ValueError(f'"{chunk.metadata["paper"]}" in papers.md has no entry in papers.json')
        chunk.metadata["paper_id"] = paper_ids[chunk.metadata["paper"]]
    return chunks


def load_vector_store() -> FAISS:
    """Load the vector store if it exists; otherwise split the library -> embed -> save."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if INDEX_PATH.exists():
        # The index was generated locally by us, so deserializing it is safe
        vector_store = FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )
        # An index built before chunks had a paper_id can't be filtered by paper; rebuild it
        first_chunk = vector_store.docstore.search(vector_store.index_to_docstore_id[0])
        if isinstance(first_chunk, Document) and "paper_id" in first_chunk.metadata:
            return vector_store
        print("Vector store has no paper_id, rebuilding it")

    chunks = split_papers()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_PATH))
    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store


vector_store = load_vector_store()


def resolve_local(state: PaperState):
    return {"local_paper": resolve_local_paper(state["query"])}


def route_source(state: PaperState) -> str:
    """Deterministic routing: a paper found in the metadata goes to local RAG, anything else online."""
    return "local" if state["local_paper"]["found"] else "external"


def local_rag(state: PaperState):
    """Retrieve chunks only from the papers the resolver found, so other papers can't crowd them out."""
    documents = []
    for paper in state["local_paper"]["papers"]:
        docs = vector_store.similarity_search(state["query"], k=LIBRARY_TOP_K, filter={"paper_id": paper["paper_id"]})
        documents += [doc.page_content for doc in docs]
    print(f"Library: local papers {[p['title'] for p in state['local_paper']['papers']]}")
    return {"source": "local", "documents": documents}


def external_retrieval(state: PaperState):
    print("Library: not in the local library, searching online")
    # Reuses the search Agent's tool; it returns failures as text instead of raising
    return {"source": "external", "documents": [search_papers.invoke({"query": state["query"]})]}


LIBRARY_FAIL_TEXT = "Sorry, paper lookup is temporarily unavailable. Please try again later."

SOURCE_NOTES = {
    "local": "These sources come from the user's local paper library.",
    "external": (
        "The paper is not in the user's local library; these sources come from an online search "
        "and only include abstracts."
    ),
}

# The prompt holds rules only; which papers exist locally is decided by the resolver, not listed here
answer_agent = create_agent(
    model=create_model(),
    system_prompt="""
        You are the paper assistant of a research workspace. You answer questions about specific papers,
        such as their method, backbone, datasets, metrics and limitations.

        You will receive the user's question and the retrieved sources. Rules:
        1. Answer only from the sources, and name which paper each piece of information comes from
        2. Do not add facts, numbers or datasets that are not in the sources
        3. If the sources come from an online search, tell the user the paper is not in their local library
           and that abstracts may not cover details like datasets or metrics
        4. If the sources don't answer the question, say so honestly
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def answer(state: PaperState):
    sources = "\n\n---\n\n".join(state["documents"])
    answer_text, tokens = await ask_agent(
        answer_agent,
        f"Question: {state['query']}\n\n{SOURCE_NOTES[state['source']]}\n\nSources:\n{sources}",
        LIBRARY_FAIL_TEXT,
    )
    return {"answer": answer_text, "tokens": tokens}


def build_library_graph():
    builder = StateGraph(PaperState)
    builder.add_node("resolve_local", resolve_local)
    builder.add_node("local_rag", local_rag)
    builder.add_node("external_retrieval", external_retrieval)
    builder.add_node("answer", answer)

    builder.add_edge(START, "resolve_local")
    builder.add_conditional_edges(
        "resolve_local", route_source, {"local": "local_rag", "external": "external_retrieval"}
    )
    builder.add_edge("local_rag", "answer")
    builder.add_edge("external_retrieval", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


library_graph = build_library_graph()


async def library_node(state: TaskState):
    print("~~~~~~This is Library Agent~~~~~~")
    task = state["task"]
    try:
        result = await library_graph.ainvoke({"query": task["sub_question"]})
        answer_text, tokens = result["answer"], result["tokens"]
    except Exception as e:
        # Same as ask_agent(): a raising branch would fail the whole parallel step
        print(f"Library lookup failed, using fallback message: {e}")
        answer_text, tokens = LIBRARY_FAIL_TEXT, 0
    return agent_update(task, answer_text, tokens)
