from pathlib import Path

from dotenv import load_dotenv

# Project root; all paths are based on it, so the working directory doesn't matter
BASE_DIR = Path(__file__).parent

# Load LangSmith settings from .env to enable tracing
load_dotenv(BASE_DIR / ".env")

# Models
CHAT_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"

# RAG: local paper library and vector store location
PAPERS_PATH = BASE_DIR / "data" / "papers.md"
INDEX_PATH = BASE_DIR / "faiss_index_papers"

# Paper search: public OpenAlex API, no key needed
OPENALEX_API_URL = "https://api.openalex.org/works"
OPENALEX_MAX_RESULTS = 5
OPENALEX_TIMEOUT = 15  # seconds

# MCP Server: uses port 8001 to avoid clashing with other local servers on 8000
MCP_HOST = "127.0.0.1"
MCP_PORT = 8001
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"
