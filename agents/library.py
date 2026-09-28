"""Paper library Agent: retrieves from the local paper library with RAG and answers questions about those papers."""
import re

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from agents.common import agent_update, ask_agent, create_model
from config import EMBEDDING_MODEL, INDEX_PATH, PAPERS_PATH
from state import TaskState


def library_titles() -> list[str]:
    """Titles of all papers in the local library (the "## <title>" headings)."""
    text = PAPERS_PATH.read_text(encoding="utf-8")
    return re.findall(r"^## (.+)$", text, flags=re.MULTILINE)


def load_vector_store() -> FAISS:
    """Load the vector store if it exists; otherwise load document -> split by heading -> embed -> save."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if INDEX_PATH.exists():
        # The index was generated locally by us, so deserializing it is safe
        return FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )

    text = TextLoader(str(PAPERS_PATH), encoding="utf-8").load()[0].page_content

    # Split on "## <paper>" headings, one chunk per paper; strip_headers=False keeps the title for retrieval
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "paper")],
        strip_headers=False,
    )
    # Drop the intro section at the top that has no paper heading
    chunks = [c for c in text_splitter.split_text(text) if "paper" in c.metadata]

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_PATH))
    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store


retriever = load_vector_store().as_retriever(search_kwargs={"k": 3})


@tool
def search_library(query: str) -> str:
    """Search the local paper library for content related to a paper name, method, dataset or metric."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant content found in the local library"
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


LIBRARY_FAIL_TEXT = "Sorry, the local paper library is temporarily unavailable. Please try again later."

library_agent = create_agent(
    model=create_model(),
    tools=[search_library],
    system_prompt="""
        You are the paper library assistant of a research workspace. You answer questions about papers
        in the user's local library, such as their method, backbone, datasets, metrics and limitations.

        Follow these steps:
        1. Extract the paper names or key terms from the question and call search_library
           - If the question mentions several papers, you may call search_library once per paper
        2. Answer only from the retrieved content, and name which paper each piece of information comes from
        3. If the library has no relevant content, say so honestly

        Do not add facts, numbers or datasets that are not in the retrieved content.
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def library_node(state: TaskState):
    print("~~~~~~This is Library Agent~~~~~~")
    task = state["task"]
    answer, tokens = await ask_agent(library_agent, task["sub_question"], LIBRARY_FAIL_TEXT)
    return agent_update(task, answer, tokens)
