# DocuMind AI — Streamlit deployment

## Put these files in your GitHub repository

- `app.py` (use the supplied fixed app)
- `requirements.txt` (use the supplied fixed requirements)
- `.streamlit/config.toml` (use the supplied fixed config)

## Streamlit Cloud

1. Deploy `app.py`.
2. In **Advanced settings**, select **Python 3.12**.
3. In **Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Never commit the API key to GitHub.

The Chroma database is intentionally in-memory and isolated per browser session. Uploaded documents are cleared when the session/app restarts.
