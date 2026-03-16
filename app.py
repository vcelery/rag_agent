"""
Prompt-Based RAG Agent — Streamlit Community Cloud entry point.

Env vars are loaded from (in priority order):
  1. Streamlit secrets  (st.secrets)
  2. .env file          (python-dotenv)
  3. Real environment variables
"""

import os
import uuid
import streamlit as st

# ── 0. Page config — must be the very first Streamlit call ───────────────────
st.set_page_config(page_title="Prompt-based RAG Agent", page_icon="📚")

# ── 1. Load configuration before importing the agent ────────────────────────

def _bootstrap_env() -> None:
    """Populate os.environ from Streamlit secrets or .env, whichever is present."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass

    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass


_bootstrap_env()

# ── 2. Import agent (env vars must be set first) ─────────────────────────────

from PromptBasedRagAgent import graph  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

# ── 3. Helpers ────────────────────────────────────────────────────────────────

def make_thread_id(seed: str) -> str:
    """Return a deterministic UUID5 from *seed* (session key)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def run_graph(messages: list, thread_id: str) -> str:
    """Invoke the LangGraph agent and return the last AI message content."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": messages}, config=config)
    last = result["messages"][-1]
    if hasattr(last, "content"):
        return last.content
    return str(last.get("content", last))


# ── 4. Streamlit UI ───────────────────────────────────────────────────────────

st.title("📚 Prompt-based RAG Agent")

# ── Session seed (stable per browser session) ────────────────────────────────
if "session_seed" not in st.session_state:
    st.session_state.session_seed = str(uuid.uuid4())

thread_id = make_thread_id(st.session_state.session_seed)

# ── 5. Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Thread ID: `{thread_id}`")
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.pending_b64 = None
        st.session_state.pending_mime = None
        st.session_state.show_camera = False
        st.rerun()
    st.divider()
    st.subheader("📎 Attach Image")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        key="uploader",
    )
    if uploaded:
        b64, mime = file_to_base64(uploaded)
        st.session_state.pending_b64 = b64
        st.session_state.pending_mime = mime

    toggle_label = "📷 Close camera" if st.session_state.show_camera else "📷 Take a photo"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.show_camera = not st.session_state.show_camera
        st.rerun()

    if st.session_state.show_camera:
        camera_snap = st.camera_input("Take a photo", label_visibility="collapsed")
        if camera_snap:
            b64, mime = file_to_base64(camera_snap)
            st.session_state.pending_b64 = b64
            st.session_state.pending_mime = mime
            st.session_state.show_camera = False
            st.rerun()

    if st.session_state.pending_b64:
        st.divider()
        st.caption("📌 Attached — will send with next message")
        render_image(st.session_state.pending_b64, width=220)
        if st.button("✕ Remove image", use_container_width=True):
            st.session_state.pending_b64 = None
            st.session_state.pending_mime = None
            st.rerun()

# ── 6. Title ──────────────────────────────────────────────────────────────────
st.title("Prompt Based Agent")

# ── 7. Chat history ───────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("image_b64"):
            render_image(msg["image_b64"])
        if msg.get("content"):
            if msg["role"] == "assistant":
                render_response(msg["content"])   # ← handles [RECIPE_IMAGE:…]
            else:
                st.markdown(msg["content"])

# ── 8. Chat input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message…")
if user_input or (st.session_state.pending_b64 and user_input is not None):
    text = user_input or ""
    image_b64  = st.session_state.pending_b64
    image_mime = st.session_state.pending_mime

    with st.chat_message("user"):
        if image_b64:
            render_image(image_b64)
        if text:
            st.markdown(text)

    st.session_state.chat_history.append({
        "role": "user",
        "content": text,
        "image_b64": image_b64,
        "image_mime": image_mime,
    })

    st.session_state.pending_b64  = None
    st.session_state.pending_mime = None

    lc_messages = []
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            content = build_lc_content(m["content"], m.get("image_b64"), m.get("image_mime"))
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = run_graph(lc_messages, thread_id)
            except Exception as exc:
                response = f"⚠️ Error: {exc}"
        render_response(response)   # ← handles [RECIPE_IMAGE:…]

    st.session_state.chat_history.append({"role": "assistant", "content": response})
