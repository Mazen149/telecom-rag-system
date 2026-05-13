# ============================================================
# IMPORTS
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from rag_class import TelecomRAG
import requests  # Used to call n8n webhook


# ============================================================
# CONFIGURATION
# ============================================================

# n8n Webhook URL (PUT YOUR REAL LINK HERE)
N8N_WEBHOOK_URL = "https://mazen2488.app.n8n.cloud/webhook/telecom-rag"


# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title="NileTel Arabic AI Assistant",
    description="RAG-based telecom support assistant with ticket automation",
    version="1.0"
)

# Load RAG system once (very important for performance)
rag = TelecomRAG()


# ============================================================
# REQUEST & RESPONSE SCHEMAS
# ============================================================

# What the user sends
class QueryRequest(BaseModel):
    query: str
    name: Optional[str] = None


# What the API returns
class QueryResponse(BaseModel):
    answer: str
    needs_action: str
    sources: list
    displayed_source: str


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {"message": "NileTel AI Assistant API is running successfully!"}


# ============================================================
# MAIN ENDPOINT (/ask)
# ============================================================

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    This endpoint:
    1. Receives user query
    2. Runs RAG pipeline
    3. Checks if action is needed
    4. If YES → calls n8n webhook
    5. Returns response to UI (Streamlit)
    """

    name = (request.name or "").strip()
    print(f"\n[API] Received new query: {request.query}")
    if name:
        print(f"[API] User name: {name}")

    # --------------------------------------------------------
    # 1. RUN RAG PIPELINE
    # --------------------------------------------------------
    response = rag.run_rag_pipeline(request.query)

    print(f"[API] Response ready | Needs Action: {response['needs_action']}")

    # Ensure response matches QueryResponse schema (add displayed_source)
    if "displayed_source" not in response:
        sources = response.get("sources") or []
        response["displayed_source"] = sources[0] if len(sources) > 0 else ""

    # --------------------------------------------------------
    # 2. IF ACTION NEEDED → CALL n8n
    # --------------------------------------------------------
    if response["needs_action"] == "YES":
        print("[API] Action detected → Triggering n8n workflow...")

        try:
            # Send POST request to n8n webhook
            res = requests.post(
                N8N_WEBHOOK_URL,   #  destination (n8n)
                json={             #  data sent to n8n
                    "query": request.query,
                    "name": name,
                    "answer": response["answer"],
                    "sources": response["sources"]
                },
                timeout=5  # prevent hanging if n8n is slow
            )

            print(f"[API] n8n status: {res.status_code}")
            print(f"[API] n8n response: {res.text}")

        except Exception as e:
            print(f"[API] n8n error: {str(e)}")

    else:
        print("[API] No action needed")

    # --------------------------------------------------------
    # 3. RETURN RESPONSE TO STREAMLIT
    # --------------------------------------------------------
    return response