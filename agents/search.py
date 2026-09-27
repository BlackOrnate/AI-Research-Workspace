"""Paper search Agent: finds papers online through the public OpenAlex API."""
import re

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.tools import tool

from agents.common import agent_update, ask_agent, create_model
from config import OPENALEX_API_URL, OPENALEX_MAX_RESULTS, OPENALEX_TIMEOUT
from state import State


def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put every word back at its position."""
    if not inverted_index:
        return ""
    positions = {pos: word for word, pos_list in inverted_index.items() for pos in pos_list}
    return " ".join(positions[i] for i in sorted(positions))


def work_link(work: dict) -> str:
    # Prefer the DOI link; fall back to the OpenAlex page when there is no DOI
    return work.get("doi") or work["id"]


def merge_versions(works: list[dict]) -> list[list[dict]]:
    """Group versions of the same paper (e.g. preprint and journal version), newest first in each group.

    Two works count as the same paper when their titles match, ignoring case and punctuation.
    Groups keep the order of the search results, so the most relevant paper stays first.
    """
    groups: dict[str, list[dict]] = {}
    for work in works:
        # A work without a title can't be matched, so it gets its own group
        key = re.sub(r"[^a-z0-9]", "", (work.get("title") or "").lower()) or work["id"]
        groups.setdefault(key, []).append(work)
    return [
        sorted(versions, key=lambda w: w.get("publication_date") or "", reverse=True)
        for versions in groups.values()
    ]


@tool
def search_papers(query: str) -> str:
    """Search OpenAlex for papers matching an English keyword query, most relevant first."""
    try:
        response = httpx.get(
            OPENALEX_API_URL,
            params={
                "search": query,
                "per_page": OPENALEX_MAX_RESULTS,
                # Only ask for the fields we use, which keeps the response small
                "select": "title,publication_date,doi,id,authorships,abstract_inverted_index",
            },
            timeout=OPENALEX_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        # Return the error to the model so it can tell the user instead of making up papers
        return f"Paper search failed: {e}"

    works = response.json().get("results", [])
    if not works:
        return "No papers found for this query"

    papers = []
    for versions in merge_versions(works):
        newest = versions[0]
        authors = [a["author"]["display_name"] for a in newest.get("authorships", [])]
        # Journal versions often have no abstract in OpenAlex, so take it from any version that has one
        abstract = next(
            (rebuild_abstract(v["abstract_inverted_index"]) for v in versions if v.get("abstract_inverted_index")),
            "",
        )
        paper = (
            f"Title: {newest.get('title')}\n"
            f"Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}\n"
            f"Published: {newest.get('publication_date')}\n"
            f"Link: {work_link(newest)}\n"
            # Only keep the start of the abstract, enough for the model to judge relevance
            f"Abstract: {abstract[:400] or 'Not available'}"
        )
        if len(versions) > 1:
            older = "\n".join(f"  - {v.get('publication_date')}: {work_link(v)}" for v in versions[1:])
            paper += f"\nOlder versions:\n{older}"
        papers.append(paper)
    return "\n\n---\n\n".join(papers)


SEARCH_FAIL_TEXT = "Sorry, online paper search is temporarily unavailable. Please try again later."

search_agent = create_agent(
    model=create_model(),
    tools=[search_papers],
    system_prompt="""
        You are the paper search assistant of a research workspace. You find papers for the user with the OpenAlex scholarly search.

        Follow these steps:
        1. Turn the user's request into a short English keyword query (3-6 words) and call search_papers
        2. If the results are clearly off-topic, you may try one more query with different keywords
        3. List the relevant papers, each with title, first authors, publish date, link,
           and one sentence on what the paper does based on its abstract
           - Versions of the same paper are already merged: show the newest version as the paper,
             and list every link under "Older versions" below it as older versions

        Only list papers returned by search_papers, and never make up titles or links.
        If the search fails or finds nothing, tell the user honestly.
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def search_node(state: State):
    print("~~~~~~This is Search Agent~~~~~~")
    question = state["messages"][0].content
    if not isinstance(question, str):
        question = " ".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in question
        )
    answer = await ask_agent(search_agent, question, SEARCH_FAIL_TEXT)
    return agent_update(answer)
