import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib
import re

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
st.markdown("""
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
""", unsafe_allow_html=True)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })
    return pages


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=900, overlap=150):
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


def initialize_database():
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.EphemeralClient()
        st.session_state.collection = st.session_state.chroma_client.create_collection(
            name="documents",
            metadata={"description": "DocuMind RAG document collection"}
        )
        st.session_state.document_count = 0
        st.session_state.processed_files = []


def process_documents(uploaded_files):
    model = load_embedding_model()
    collection = st.session_state.collection

    total_chunks = 0

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_files:
            continue

        pages = extract_text_from_pdf(uploaded_file)

        for page_data in pages:
            chunks = chunk_text(page_data["text"])

            if not chunks:
                continue

            embeddings = model.encode(
                chunks,
                normalize_embeddings=True
            ).tolist()

            ids = []
            metadatas = []

            for index, chunk in enumerate(chunks):
                unique_text = f"{uploaded_file.name}-{page_data['page']}-{index}-{chunk[:100]}"
                chunk_id = hashlib.md5(unique_text.encode()).hexdigest()

                ids.append(chunk_id)
                metadatas.append({
                    "source": uploaded_file.name,
                    "page": page_data["page"],
                    "chunk": index
                })

            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )

            total_chunks += len(chunks)

        st.session_state.processed_files.append(uploaded_file.name)

    st.session_state.document_count = collection.count()
    return total_chunks


def retrieve_context(question, n_results=4):
    model = load_embedding_model()
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = st.session_state.collection.query(
        query_embeddings=question_embedding,
        n_results=min(n_results, max(st.session_state.document_count, 1)),
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []
    sources = []

    for document, metadata in zip(documents, metadatas):
        context_parts.append(document)
        source = {
            "file": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "?")
        }
        if source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(context_parts), sources


def generate_answer(question):
    context, sources = retrieve_context(question)

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=800
    )

    answer = response.choices[0].message.content
    return answer, sources


def clear_knowledge_base():
    st.session_state.chroma_client = chromadb.EphemeralClient()
    st.session_state.collection = st.session_state.chroma_client.create_collection(
        name="documents"
    )
    st.session_state.document_count = 0
    st.session_state.processed_files = []
    st.session_state.chat_history = []


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
initialize_database()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
        help="Upload one or more PDF files."
    )

    if uploaded_files:
        if st.button("⚡ Process Documents", use_container_width=True):
            try:
                with st.spinner("Reading documents and building the knowledge base..."):
                    new_chunks = process_documents(uploaded_files)

                if new_chunks > 0:
                    st.success(f"Processed {new_chunks} new chunks.")
                else:
                    st.info("These documents were already processed or contained no extractable text.")
            except Exception as error:
                st.error(f"Could not process document: {error}")

    st.markdown(f"**📄 Documents:** {len(st.session_state.processed_files)}")
    st.markdown(f"**🧩 Knowledge chunks:** {st.session_state.document_count}")

    st.divider()

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        clear_knowledge_base()
        st.rerun()

    st.divider()
    st.caption("Powered by Groq • Llama • ChromaDB")


# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="badge">● RAG POWERED DOCUMENT INTELLIGENCE</div>
    <h1>DocuMind <span style="color:#8b9aff;">AI</span></h1>
    <p>Upload your documents, build a private knowledge base, and ask intelligent questions grounded in your content.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.document_count == 0:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📄 Upload</h3>
            <p>Add one or multiple PDF documents to your knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Retrieve</h3>
            <p>Semantic search finds the most relevant document sections.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Ask</h3>
            <p>Groq generates fast answers using your retrieved context.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("👈 Start by uploading PDF documents from the sidebar, then click **Process Documents**.")

else:
    st.success(
        f"Knowledge base ready — {len(st.session_state.processed_files)} document(s), "
        f"{st.session_state.document_count} chunks."
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📌 Sources used"):
                    for source in message["sources"]:
                        st.markdown(
                            f'<div class="source-box">📄 <b>{source["file"]}</b> '
                            f'— Page {source["page"]}</div>',
                            unsafe_allow_html=True
                        )

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents and generating an answer..."):
                try:
                    answer, sources = generate_answer(question)
                    st.markdown(answer)

                    if sources:
                        with st.expander("📌 Sources used"):
                            for source in sources:
                                st.markdown(
                                    f'<div class="source-box">📄 <b>{source["file"]}</b> '
                                    f'— Page {source["page"]}</div>',
                                    unsafe_allow_html=True
                                )

                except KeyError:
                    answer = "Groq API key not found. Please add GROQ_API_KEY to Streamlit secrets."
                    sources = []
                    st.error(answer)

                except Exception as error:
                    answer = f"An error occurred: {error}"
                    sources = []
                    st.error(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
