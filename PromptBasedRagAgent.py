import os
import datetime
from pathlib import Path
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# ── Document loaders ──────────────────────────────────────────────────────────
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Local embeddings + vector store ──────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── Google Drive ──────────────────────────────────────────────────────────────
import gdrive_utils

PROMPT_NAME  = "agent.prompt"
PROMPT_PATH  = os.path.join(os.path.dirname(__file__), "prompts", PROMPT_NAME)
RAG_DIR      = os.path.join(os.path.dirname(__file__), "rag")
OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Folder ID comes from env / Streamlit secrets (set GDRIVE_FOLDER_ID)
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "")

# ── Build RAG index at startup ────────────────────────────────────────────────

_LOADERS = {
    ".txt":  TextLoader,
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
}

def _load_documents():
    """Load all supported files from the rag/ folder."""
    docs = []
    rag_path = Path(RAG_DIR)
    for path in rag_path.iterdir():
        loader_cls = _LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            loader = loader_cls(str(path))
            loaded = loader.load()
            # Tag each chunk with its source filename
            for doc in loaded:
                doc.metadata.setdefault("source", path.name)
            docs.extend(loaded)
            print(f"[RAG] Loaded: {path.name} ({len(loaded)} chunk(s))")
        except Exception as e:
            print(f"[RAG] Warning — could not load {path.name}: {e}")
    return docs


def _build_index():
    """Return a FAISS retriever, or None if the rag/ folder is empty."""
    docs = _load_documents()
    if not docs:
        print("[RAG] No documents found in rag/ — retrieval tool will be disabled.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    print(f"[RAG] Index built: {len(chunks)} chunks from {len(docs)} document(s).")
    return db.as_retriever(search_kwargs={"k": 4})


_retriever = _build_index()

# ── Tools ─────────────────────────────────────────────────────────────────────

def get_current_date() -> str:
    """Get today's date in ISO format."""
    return datetime.date.today().isoformat()


def search_documents(query: str) -> str:
    """Search the internal knowledge base for information relevant to the query.
    Returns the most relevant passages found in the loaded documents."""
    if _retriever is None:
        return "No documents are available in the knowledge base."
    try:
        results = _retriever.invoke(query)
        if not results:
            return "No relevant passages found."
        parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error during document search: {e}"

# ── Google Drive tools ────────────────────────────────────────────────────────
def list_drive_recipes(search: str = "") -> str:
    """List recipe image files available in the Google Drive recipe folder.

    Optionally pass a *search* term (e.g. 'pasta', 'salad') to filter results
    by filename. Returns file names and their Drive IDs so you can call
    get_recipe_image() with the right ID.
    """
    if not GDRIVE_FOLDER_ID:
        return "Google Drive folder is not configured (GDRIVE_FOLDER_ID missing)."
    try:
        files = gdrive_utils.list_image_files(GDRIVE_FOLDER_ID)
    except Exception as e:
        return f"Error accessing Google Drive: {e}"

    if not files:
        return "No recipe images found in the Google Drive folder."

    if search:
        term = search.lower()
        files = [f for f in files if term in f["name"].lower()]
        if not files:
            return f"No recipe images matched '{search}'."

    lines = [f"- {f['name']}  (id: {f['id']})" for f in files]
    return "Available recipe images:\n" + "\n".join(lines)


def get_recipe_image(file_id: str) -> str:
    """Fetch a recipe image from Google Drive.

    CRITICAL: The tool returns a tag like [RECIPE_IMAGE:abc123].
    You MUST paste that tag EXACTLY and VERBATIM into your reply.
    The UI converts it into a visible image — but only if the tag
    is present in your message. Never replace it with a description.
    """
    if not file_id.strip():
        return "No file_id provided."
    return f"[RECIPE_IMAGE:{file_id.strip()}]"

# ── Prompt ────────────────────────────────────────────────────────────────────

def _load_system_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

base_system_prompt = _load_system_prompt()


def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    system_msg = base_system_prompt
    return [{"role": "system", "content": system_msg}] + state["messages"]


# ── Graph ─────────────────────────────────────────────────────────────────────

_tools = [get_current_date, search_documents]

graph = create_react_agent(
    model=f"openai:{OPENAI_MODEL}",
    tools=_tools,
    prompt=prompt,
)
