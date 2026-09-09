# DocuMind AI — Streamlit deployment

## Files
- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`

## Streamlit Secrets
In your Streamlit app, open **Manage app → Settings → Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do not commit the API key to GitHub.

## Important
This version intentionally does **not** use ChromaDB. The previous deployment was failing inside ChromaDB's process-global `SharedSystemClient` initialization on Streamlit Cloud. The app now uses the same SentenceTransformer embeddings plus an in-session NumPy cosine-similarity vector store, which removes that failure point and keeps each browser session isolated.

The application is ephemeral: uploaded documents are held in the Streamlit session and are cleared when that session/app instance is restarted. For a production multi-user persistent knowledge base, use a hosted vector database later.

## Deployment
1. Push the files to GitHub.
2. Deploy `app.py` from Streamlit Community Cloud.
3. Add `GROQ_API_KEY` in Streamlit Secrets.
4. Redeploy/reboot the app.
