# DocuMind AI — Streamlit deployment

## Files to put in GitHub

```text
app.py
requirements.txt
.streamlit/config.toml
```

Do not upload `__pycache__`.

## Streamlit Cloud

Use Python 3.12 (Advanced settings) and add this secret:

```toml
GROQ_API_KEY = "YOUR_ACTUAL_GROQ_API_KEY"
```

Never commit the API key to GitHub.

## Why this version is stable

- It does not use ChromaDB, avoiding the Chroma SharedSystemClient startup failure.
- PDF uploads are copied to bytes before pypdf reads them. This avoids Streamlit UploadedFile stream exhaustion after reruns/questions and when a second PDF is uploaded.
- Each PDF is indexed atomically: a failed PDF cannot leave a half-indexed document in the session.
- Duplicate detection uses the PDF content hash, so the same file is not indexed twice while different files with the same filename/size can still be indexed.
- Retrieval uses normalized SentenceTransformer embeddings with NumPy cosine similarity.
- The Groq API key is read from Streamlit Secrets.

## Expected workflow

1. Upload one or more text-based PDFs.
2. Click **Process Documents**.
3. Ask questions about the uploaded documents.
4. Add another PDF later and click **Process Documents** again. Existing indexed documents remain in the current session.

Scanned/image-only PDFs are not OCR'd by this application; they will show a clear warning instead of crashing.
