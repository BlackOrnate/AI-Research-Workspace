"""Reading notes MCP Server. Run: python mcp_server.py"""

from mcp.server import FastMCP

from config import MCP_HOST, MCP_PORT

mcp = FastMCP("Research Workspace", host=MCP_HOST, port=MCP_PORT)


# Mock data: user ID -> reading list with notes
READING_NOTES = {
    1: [
        {
            "date": "2026-08-02",
            "paper": "HoVer-Net",
            "status": "read",
            "note": "Horizontal/vertical distance maps are a neat trick for splitting touching nuclei. Post-processing looks fragile.",
        },
        {
            "date": "2026-08-20",
            "paper": "CellViT",
            "status": "read",
            "note": "Same maps as HoVer-Net but with a ViT encoder. SAM ViT-H encoder gave the best PQ on PanNuke.",
        },
        {
            "date": "2026-09-10",
            "paper": "MedSAM",
            "status": "to-read",
            "note": "Check whether box prompts are practical for thousands of nuclei per image.",
        },
    ],
    2: [
        {
            "date": "2026-07-15",
            "paper": "StarDist",
            "status": "read",
            "note": "Star-convex polygons work well for round nuclei, not for elongated cells.",
        },
        {
            "date": "2026-09-05",
            "paper": "Cellpose",
            "status": "reading",
            "note": "Flow fields handle irregular shapes. Compare against StarDist on our data.",
        },
    ],
}


@mcp.tool()
def query_reading_notes(user_id: int) -> str:
    """Query the given user's reading list and notes, sorted from most recent to oldest."""
    notes = READING_NOTES.get(user_id)
    if not notes:
        return "No reading notes found for this user"

    notes = sorted(notes, key=lambda n: n["date"], reverse=True)
    return "\n".join(
        f"Date: {n['date']}, Paper: {n['paper']}, Status: {n['status']}, Note: {n['note']}"
        for n in notes
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
