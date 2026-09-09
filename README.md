# 📚 DocuMind AI — RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application built with:

- Streamlit
- Groq API
- Llama 3.1 8B Instant
- ChromaDB
- Sentence Transformers
- PyPDF

## Features

- Upload multiple PDF files
- Extract and chunk document text
- Create local semantic embeddings
- Store vectors in ChromaDB
- Retrieve relevant document sections
- Ask questions using Groq
- Display source pages
- Maintain chat history
- Clear the knowledge base
- Modern custom Streamlit interface

## Project Structure

```text
DocuMind_RAG/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── .streamlit/
    └── config.toml
```

## Local Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate it

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create local secrets

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 5. Run

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Upload these files to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select your GitHub repository.
5. Select `app.py`.
6. Open Advanced Settings.
7. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

8. Deploy.

## Important Note

This version uses an in-memory ChromaDB database (`EphemeralClient`).
That makes deployment simple, but uploaded documents are only available during
the current app session. For permanent multi-user storage, use a managed cloud
vector database in a future version.
