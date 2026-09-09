import hashlib
import os
import re

import chromadb
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown(
    """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #18223d 0%, #0b1020 38%, #070b14 100%);
        color: #f5f7fb;
    }
    #MainMenu, footer {visibility: hidden;}
    header {background: transparent !important;}

    .hero {
        padding: 2.2rem 2.4rem;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(76, 99, 255, .22), rgba(21, 192, 170, .10));
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 3rem;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero p {
        color: #b9c3d6;
        font-size: 1.05rem;
        margin: .55rem 0 0 0;
    }
    .badge {
        display: inline-block;
        padding: .35rem .75rem;
        border-radius: 999px;
        background: rgba(77, 99, 255, .20);
        border: 1px solid rgba(118, 136, 255, .35);
        color: #cdd5ff;
        font-size: .82rem;
        margin-bottom: .8rem;
    }
    .feature-card {
        padding: 1.2rem;
        min-height: 110px;
        border-radius: 18px;
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.08);
    }
    .feature-card h3 {margin: 0 0 .4rem 0;}
    .feature-card p {color: #aeb8ca; margin: 0;}
    .source-box {
        padding: .75rem 1rem;
        margin: .5rem 0;
        border-left: 3px solid #6f83ff;
        background: rgba(111,131,255,.08);
        border-radius: 8px;
        color: #cbd3e4;
        font-size: .9rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11182a, #0b1020);
        border-right: 1px solid rgba(255,255,255,.07);
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model():
    """Load the local embedding model once per Streamlit process."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def initialize_database():
    """
    Create one isolated, in-memory Chroma database for this browser session.

    Client() is Chroma's standard in-memory client. We intentionally keep the
    client in session_state so one user's uploaded documents are not shared
    with another user's session.
    """
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.Client()

    if "collection" not in st.session_state:
        st.session_state.collection = (
            st.session_state.chroma_client.get_or_create_collection(
                name="documents",
                metadata={
                    "description": "DocuMind RAG document collection",
                    "hnsw:space": "cosine",
                },
            )
        )

    if "document_count" not in st.session_state:
        st.session_state.document_count = st.session_state.collection.count()

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def extract_text_from_pdf(uploaded_file):
    """Extract text page-by-page from a PDF uploaded to Streamlit."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=900, overlap=150):
    """Split text into overlapping chunks while preferring sentence boundaries."""
    text = clean_text(text)
    chunks = []

    if not text:
        return chunks

    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            sentence_break = text.rfind(". ", start, end)
            if sentence_break > start + (chunk_size // 2):
                end = sentence_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def process_documents(uploaded_files):
    """Embed uploaded PDFs and store their chunks in Chroma."""
    model = load_embedding_model()
    collection = st.session_state.collection

    total_chunks = 0
    processed_this_run = []

    for uploaded_file in uploaded_files:
        file_key = f"{uploaded_file.name}:{uploaded_file.size}"

        if file_key in st.session_state.processed_files:
            continue

        pages = extract_text_from_pdf(uploaded_file)

        for page_data in pages:
            chunks = chunk_text(page_data["text"])

            if not chunks:
                continue

            embeddings = model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            ids = []
            metadatas = []

            for index, chunk in enumerate(chunks):
                unique_text = (
                    f"{uploaded_file.name}|{uploaded_file.size}|"
                    f"{page_data['page']}|{index}|{chunk}"
                )
                chunk_id = hashlib.sha256(unique_text.encode("utf-8")).hexdigest()

                ids.append(chunk_id)
                metadatas.append(
                    {
                        "source": uploaded_file.name,
                        "page": page_data["page"],
                        "chunk": index,
                    }
                )

            # upsert makes repeated processing safe instead of failing on
            # duplicate Chroma IDs.
            collection.upsert(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)

        if pages:
            st.session_state.processed_files.append(file_key)
            processed_this_run.append(uploaded_file.name)

    st.session_state.document_count = collection.count()

    return total_chunks, processed_this_run


def retrieve_context(question, n_results=4):
    """Retrieve the most relevant document chunks for a question."""
    if st.session_state.document_count == 0:
        return "", []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    result_count = min(n_results, st.session_state.document_count)

    results = st.session_state.collection.query(
        query_embeddings=question_embedding,
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []

    context_parts = []
    sources = []

    for document, metadata in zip(documents, metadatas):
        if not document:
            continue

        context_parts.append(document)

        source = {
            "file": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "?"),
        }

        if source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(context_parts), sources


def get_groq_api_key():
    """Read the Groq key from Streamlit secrets, with an environment fallback."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None

    if not key:
        key = os.environ.get("GROQ_API_KEY")

    if not key or not str(key).strip():
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add GROQ_API_KEY to Streamlit Cloud "
            "Secrets and restart the app."
        )

    return str(key).strip()


def generate_answer(question):
    context, sources = retrieve_context(question)

    if not context:
        return (
            "I couldn't find this information in the uploaded documents.",
            sources,
        )

    client = Groq(api_key=get_groq_api_key())

    system_prompt = """You are DocuMind AI, a helpful document question-answering assistant.

Use ONLY the information provided in the retrieved document context.

Rules:
1. Do not invent information.
2. If the answer is not available in the context, clearly say:
   "I couldn't find this information in the uploaded documents."
3. Give a clear, accurate, and well-structured answer.
4. You may summarize or combine information from multiple context sections.
"""

    user_prompt = f"""RETRIEVED DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Answer the question using only the retrieved document context."""

    response = client.chat.completions.create(
        # This is the current recommended replacement for the retired
        # llama-3.1-8b-instant ID.
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    answer = response.choices[0].message.content or (
        "I couldn't generate an answer from the retrieved documents."
    )

    return answer, sources


def clear_knowledge_base():
    """Reset the current user's in-memory knowledge base."""
    st.session_state.chroma_client = chromadb.Client()
    st.session_state.collection = (
        st.session_state.chroma_client.get_or_create_collection(
            name="documents",
            metadata={
                "description": "DocuMind RAG document collection",
                "hnsw:space": "cosine",
            },
        )
    )
    st.session_state.document_count = 0
    st.session_state.processed_files = []
    st.session_state.chat_history = []


def display_sources(sources):
    if not sources:
        return

    with st.expander("📌 Sources used"):
        for source in sources:
            st.markdown(
                f'<div class="source-box">📄 <b>{source["file"]}</b> '
                f'— Page {source["page"]}</div>',
                unsafe_allow_html=True,
            )


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
initialize_database()


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 DocuMind AI")
    st.caption("Intelligent Document Assistant")
    st.divider()

    st.markdown("### 📁 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files.",
    )

    if uploaded_files and st.button(
        "⚡ Process Documents",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Reading documents and building the knowledge base..."
            ):
                new_chunks, processed_names = process_documents(uploaded_files)

            if new_chunks > 0:
                st.success(f"Processed {new_chunks} new chunks.")
            elif processed_names:
                st.info("The selected PDFs contained no extractable text.")
            else:
                st.info(
                    "These documents were already processed. "
                    "Upload a new PDF or clear the knowledge base."
                )

        except Exception as error:
            st.error(f"Could not process document: {error}")

    st.markdown(
        f"**📄 Documents:** {len(st.session_state.processed_files)}"
    )
    st.markdown(
        f"**🧩 Knowledge chunks:** {st.session_state.document_count}"
    )

    st.divider()

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        clear_knowledge_base()
        st.rerun()

    st.divider()
    st.caption("Powered by Groq • GPT-OSS • ChromaDB")


# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.markdown(
    """
<div class="hero">
    <div class="badge">● RAG POWERED DOCUMENT INTELLIGENCE</div>
    <h1>DocuMind <span style="color:#8b9aff;">AI</span></h1>
    <p>Upload your documents, build a private knowledge base, and ask intelligent questions grounded in your content.</p>
</div>
""",
    unsafe_allow_html=True,
)


if st.session_state.document_count == 0:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <h3>📄 Upload</h3>
            <p>Add one or multiple PDF documents to your knowledge base.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <h3>🧠 Retrieve</h3>
            <p>Semantic search finds the most relevant document sections.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="feature-card">
            <h3>⚡ Ask</h3>
            <p>Groq generates fast answers using your retrieved context.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info(
        "👈 Start by uploading PDF documents from the sidebar, "
        "then click **Process Documents**."
    )

else:
    st.success(
        f"Knowledge base ready — {len(st.session_state.processed_files)} "
        f"document(s), {st.session_state.document_count} chunks."
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                display_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching your documents and generating an answer..."
            ):
                try:
                    answer, sources = generate_answer(question)
                    st.markdown(answer)
                    display_sources(sources)

                except RuntimeError as error:
                    answer = str(error)
                    sources = []
                    st.error(answer)

                except Exception as error:
                    answer = (
                        "An error occurred while generating the answer. "
                        f"Details: {error}"
                    )
                    sources = []
                    st.error(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
