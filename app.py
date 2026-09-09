import os
import re
from io import BytesIO

import numpy as np
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


def get_groq_api_key():
    """Read and validate the Groq API key from Streamlit Secrets."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except KeyError as exc:
        raise RuntimeError(
            "Groq API key not found. Open Streamlit Cloud → Manage app → Settings → Secrets "
            "and add: GROQ_API_KEY = \"your_groq_api_key\""
        ) from exc

    if not api_key or not str(api_key).strip():
        raise RuntimeError("GROQ_API_KEY is empty. Add a valid Groq API key to Streamlit Secrets.")

    return str(api_key).strip()


def display_sources(sources):
    """Display source pages used to answer the question."""
    if not sources:
        return
    with st.expander("📌 Sources used"):
        for source in sources:
            st.markdown(
                f'<div class="source-box">📄 <b>{source["file"]}</b> — Page {source["page"]}</div>',
                unsafe_allow_html=True,
            )


def initialize_database():
    """Initialize a lightweight per-session vector store.

    We intentionally do not start ChromaDB here. Streamlit Community Cloud
    reruns the script frequently, and Chroma's process-global system cache can
    conflict with Streamlit sessions. A small NumPy cosine-similarity store is
    sufficient for this application's document-sized RAG workload.
    """
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = []
    if "metadatas" not in st.session_state:
        st.session_state.metadatas = []
    if "document_count" not in st.session_state:
        st.session_state.document_count = len(st.session_state.documents)
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def extract_text_from_pdf(pdf_bytes, filename):
    """Extract text page-by-page from PDF bytes.

    Streamlit's UploadedFile is a temporary stream that can be exhausted after
    reruns. We therefore copy the bytes once and always give pypdf a fresh
    BytesIO object. This is especially important when users ask questions and
    then add another PDF to the same Streamlit session.
    """
    if not pdf_bytes:
        raise ValueError(f"'{filename}' is empty.")

    reader = PdfReader(BytesIO(pdf_bytes))
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
    """Embed uploaded PDFs and store their chunks in this session.

    Each file is copied to memory before pypdf reads it. A bad PDF is reported
    individually instead of aborting the entire batch, so adding a second PDF
    cannot destroy the knowledge base created from the first PDF.
    """
    model = load_embedding_model()
    total_chunks = 0
    processed_this_run = []
    errors = []

    for uploaded_file in uploaded_files:
        try:
            pdf_bytes = uploaded_file.getvalue()
        except Exception as error:
            errors.append(f"{uploaded_file.name}: could not read the uploaded file ({error}).")
            continue

        file_key = f"{uploaded_file.name}:{hashlib.sha256(pdf_bytes).hexdigest()}"

        if file_key in st.session_state.processed_files:
            continue

        try:
            pages = extract_text_from_pdf(pdf_bytes, uploaded_file.name)
        except Exception as error:
            errors.append(f"{uploaded_file.name}: {error}")
            continue

        if not pages:
            errors.append(
                f"{uploaded_file.name}: no extractable text was found. "
                "If this is a scanned/image-only PDF, OCR is required."
            )
            continue

        file_documents = []
        file_embeddings = []
        file_metadatas = []

        for page_data in pages:
            chunks = chunk_text(page_data["text"])
            if not chunks:
                continue

            embeddings = model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            metadatas = [
                {
                    "source": uploaded_file.name,
                    "page": page_data["page"],
                    "chunk": index,
                }
                for index, _ in enumerate(chunks)
            ]

            file_documents.extend(chunks)
            file_embeddings.extend(embeddings)
            file_metadatas.extend(metadatas)

        if not file_documents:
            errors.append(
                f"{uploaded_file.name}: no extractable text was found. "
                "If this is a scanned/image-only PDF, OCR is required."
            )
            continue

        # Commit this file only after all of its pages have been embedded.
        # That prevents a failed second PDF from leaving a half-indexed file.
        st.session_state.documents.extend(file_documents)
        st.session_state.embeddings.extend(file_embeddings)
        st.session_state.metadatas.extend(file_metadatas)
        total_chunks += len(file_documents)
        st.session_state.processed_files.append(file_key)
        processed_this_run.append(uploaded_file.name)

    st.session_state.document_count = len(st.session_state.documents)
    return total_chunks, processed_this_run, errors


def retrieve_context(question, n_results=4):
    """Retrieve the most similar chunks using cosine similarity."""
    if not st.session_state.documents:
        return "", []

    model = load_embedding_model()
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )[0]

    matrix = np.asarray(st.session_state.embeddings, dtype=np.float32)
    query = np.asarray(question_embedding, dtype=np.float32)

    # Embeddings are normalized, so dot product equals cosine similarity.
    scores = matrix @ query
    top_k = min(n_results, len(st.session_state.documents))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    context_parts = []
    sources = []

    for index in top_indices:
        document = st.session_state.documents[int(index)]
        metadata = st.session_state.metadatas[int(index)]
        context_parts.append(document)

        source = {
            "file": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "?"),
        }
        if source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(context_parts), sources


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
    """Clear only this browser session's uploaded knowledge base."""
    st.session_state.documents = []
    st.session_state.embeddings = []
    st.session_state.metadatas = []
    st.session_state.document_count = 0
    st.session_state.processed_files = []
    st.session_state.chat_history = []


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
                new_chunks, processed_names, processing_errors = process_documents(
                    uploaded_files
                )

            if new_chunks > 0:
                st.success(f"Processed {new_chunks} new chunks.")
            elif not processing_errors and not processed_names:
                st.info(
                    "These documents were already processed. "
                    "Upload a new PDF or clear the knowledge base."
                )

            for processing_error in processing_errors:
                st.warning(f"Could not process PDF: {processing_error}")

        except Exception as error:
            st.error(f"Could not process documents: {error}")

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
    st.caption("Powered by Groq • GPT-OSS • NumPy vector search")


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
