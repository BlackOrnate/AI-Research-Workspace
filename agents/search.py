"""Paper search Agent: finds papers online through the public OpenAlex API, with arXiv as a fallback."""

import re
from xml.etree import ElementTree

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.tools import tool

from agents.common import agent_update, ask_agent, create_model
from config import (
    ARXIV_API_URL,
    ARXIV_MAX_RESULTS,
    ARXIV_TIMEOUT,
    OPENALEX_API_URL,
    OPENALEX_MAX_RESULTS,
    OPENALEX_TIMEOUT,
)
from state import TaskState


def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put every word back at its position."""
    if not inverted_index:
        return ""
    positions = {
        pos: word for word, pos_list in inverted_index.items() for pos in pos_list
    }
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


def format_paper(
    title: str | None,
    authors: list[str],
    published: str | None,
    link: str,
    abstract: str,
) -> str:
    """One paper as text for the model; both search sources use the same format."""
    return (
        f"Title: {title}\n"
        f"Authors: {', '.join(authors[:3])}{' et al.' if len(authors) > 3 else ''}\n"
        f"Published: {published}\n"
        f"Link: {link}\n"
        # Only keep the start of the abstract, enough for the model to judge relevance
        f"Abstract: {abstract[:400] or 'Not available'}"
    )


def search_openalex(query: str) -> str:
    """Search OpenAlex; raises httpx.HTTPError if the request fails (e.g. the daily limit is reached)."""
    response = httpx.get(
        OPENALEX_API_URL,
        params={
            # OpenAlex reads "?" and "*" as wildcards and answers 400 for a normal search,
            # so a question like "What is X?" must have them removed
            "search": re.sub(r"[?*]", " ", query),
            "per_page": OPENALEX_MAX_RESULTS,
            # Only ask for the fields we use, which keeps the response small
            "select": "title,publication_date,doi,id,authorships,abstract_inverted_index",
        },
        timeout=OPENALEX_TIMEOUT,
    )
    response.raise_for_status()

    works = response.json().get("results", [])
    if not works:
        return "No papers found for this query"

    papers = []
    for versions in merge_versions(works):
        newest = versions[0]
        authors = [a["author"]["display_name"] for a in newest.get("authorships", [])]
        # Journal versions often have no abstract in OpenAlex, so take it from any version that has one
        abstract = next(
            (
                rebuild_abstract(v["abstract_inverted_index"])
                for v in versions
                if v.get("abstract_inverted_index")
            ),
            "",
        )
        paper = format_paper(
            newest.get("title"),
            authors,
            newest.get("publication_date"),
            work_link(newest),
            abstract,
        )
        if len(versions) > 1:
            older = "\n".join(
                f"  - {v.get('publication_date')}: {work_link(v)}" for v in versions[1:]
            )
            paper += f"\nOlder versions:\n{older}"
        papers.append(paper)
    return "\n\n---\n\n".join(papers)


def search_arxiv(query: str) -> str:
    """Search arXiv; raises httpx.HTTPError if the request fails. Only covers preprints on arXiv."""
    response = httpx.get(
        ARXIV_API_URL,
        params={
            # Every keyword must appear somewhere in the paper, like a normal keyword search
            "search_query": " AND ".join(f"all:{word}" for word in query.split()),
            # Results are sorted by relevance by default; passing sortBy=relevance explicitly
            # makes the API answer 406 Not Acceptable
            "max_results": ARXIV_MAX_RESULTS,
        },
        timeout=ARXIV_TIMEOUT,
    )
    response.raise_for_status()

    # arXiv returns an Atom XML feed, one <entry> per paper
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = ElementTree.fromstring(response.text).findall("atom:entry", ns)
    if not entries:
        return "No papers found for this query"

    def text(element: ElementTree.Element, tag: str) -> str:
        # Titles and abstracts contain line breaks and extra spaces
        return " ".join(element.findtext(tag, "", ns).split())

    return "\n\n---\n\n".join(
        format_paper(
            text(entry, "atom:title"),
            [text(author, "atom:name") for author in entry.findall("atom:author", ns)],
            text(entry, "atom:published")[
                :10
            ],  # "2024-05-01T17:59:59Z" -> "2024-05-01"
            text(
                entry, "atom:id"
            ),  # The abstract page, e.g. http://arxiv.org/abs/2405.00001v1
            text(entry, "atom:summary"),
        )
        for entry in entries
    )


@tool
def search_papers(query: str) -> str:
    """Search for papers matching an English keyword query, most relevant first."""
    try:
        return search_openalex(query)
    except httpx.HTTPError as e:
        # OpenAlex has a daily request limit; when it fails, try arXiv once instead
        print(f"OpenAlex search failed, trying arXiv: {e}")
    try:
        return (
            "(OpenAlex is unavailable, these results come from arXiv)\n\n"
            + search_arxiv(query)
        )
    except (httpx.HTTPError, ElementTree.ParseError) as e:
        # Return the error to the model so it can tell the user instead of making up papers
        return f"Paper search failed: {e}"


SEARCH_FAIL_TEXT = (
    "Sorry, online paper search is temporarily unavailable. Please try again later."
)

search_agent = create_agent(
    model=create_model(),
    tools=[search_papers],
    system_prompt="""
        You are the paper search assistant of a research workspace. You find papers for the user with an
        online scholarly search (OpenAlex, with arXiv as a fallback when OpenAlex is unavailable).

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


async def search_node(state: TaskState):
    print("~~~~~~This is Search Agent~~~~~~")
    task = state["task"]
    answer, tokens = await ask_agent(
        search_agent, task["sub_question"], SEARCH_FAIL_TEXT
    )
    return agent_update(task, answer, tokens)
