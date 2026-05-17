
# NileTel Arabic AI Assistant

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-CPU-2C2D72)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-5.5-0A66C2)
![Groq](https://img.shields.io/badge/Groq-LLM-FF6B00)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-2088FF?logo=github-actions&logoColor=white)

## 🚀 Live Demo

Experience the NileTel Arabic AI Assistant in action!

👉 <b><a href="https://telecom-rag-system.streamlit.app/" style="color: #007bff; text-decoration: none;">Launch the Live Demo Here</a></b>

RAG-based telecom support assistant for Arabic and English with hybrid retrieval, query rewriting, and ticket automation. The UI is built with Streamlit and the FastAPI backend is deployed on Hugging Face. Ticket creation is triggered automatically via n8n when the answer implies an action, and the UI asks for the user name only in those cases.

## Features

- Hybrid retrieval: semantic search + BM25 with RRF fusion
- Query rewriting to improve recall
- Paragraph chunking with overlap
- Action detection to trigger ticket automation
- Streamlit UI with RTL-friendly answer rendering
- Caching of embeddings and FAISS index for fast startup
- CI/CD automation: Automated deployment using GitHub Actions

## Architecture

1. Streamlit UI sends queries to FastAPI
2. FastAPI calls TelecomRAG for retrieval and answer generation
3. If action is required, FastAPI posts to the n8n webhook (Google Sheets)
4. UI requests the user name only when a ticket must be created

## Setup

### 1) Create environment file

Copy the example and set your Groq API key:

```
cp .env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=your_key_here
```

### Installation & Running Options

Choose one of the following methods to run the project. We recommend using **Option A** for significantly faster dependency installation and execution.

#### Option A: Fast Setup with `uv` (Recommended)

**1) Install dependencies:**
```bash
uv pip install -r requirements.txt
```

**2) Run the API:**
```bash
uv run uvicorn main:app --reload --port 8000
```

**3) Run the UI:**
```bash
uv run streamlit run streamlit_app.py
```

---

#### Option B: Standard Setup with `pip`

**1) Install dependencies:**
```bash
pip install -r requirements.txt
```

**2) Run the API:**
```bash
uvicorn main:app --reload --port 8000
```

**3) Run the UI:**
```bash
streamlit run streamlit_app.py
```

## Configuration

- `N8N_WEBHOOK_URL` in `main.py` to point to your n8n workflow
- `API_URL` in `streamlit_app.py` to point to your FastAPI endpoint
- `SHEET_URL` in `streamlit_app.py` to your Google Sheet CSV export

## Data Ingestion

Place Markdown files inside the `data/` directory. The index is cached in `cache/` (ignored by git). If the data changes, the cache is automatically rebuilt.

## Notes

- Ticket creation is triggered only when the assistant response implies an action.
- The UI collects the user name only in those cases, then submits the ticket.

