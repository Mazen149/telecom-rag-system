import streamlit as st
import requests
import pandas as pd
import html
import json

# =========================
# 🔗 API & Sheet URLs
# =========================
# The FastAPI endpoint
API_URL = "https://1943-156-218-66-1.ngrok-free.app/ask"

# The Google Sheet CSV export link
SHEET_URL = "https://docs.google.com/spreadsheets/d/1qgwJYpHE6ehE242_CUC9-iH0SLDK-VPKtp1VDIp4Rqs/export?format=csv&gid=0"

st.set_page_config(page_title="NileTel Assistant", layout="wide", page_icon="📡")

if "last_response" not in st.session_state:
    st.session_state.last_response = None
    st.session_state.last_query = ""
    st.session_state.ticket_name = ""

# Injecting Custom CSS for a VERY GOOD UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #0f8b8d, #e76f51);
        --bg-color: #f4f6f8;
        --card-bg: rgba(255, 255, 255, 0.85);
        --text-main: #1c1c1c;
        --text-muted: #5a5a5a;
        --border-color: rgba(0, 0, 0, 0.08);
        --shadow-soft: 0 10px 40px rgba(0, 0, 0, 0.05);
        --shadow-hover: 0 15px 50px rgba(15, 139, 141, 0.15);
        --radius-lg: 24px;
        --radius-md: 16px;
    }

    .stApp {
        font-family: 'Outfit', 'Cairo', sans-serif;
        background: 
            radial-gradient(circle at 15% 50%, rgba(15, 139, 141, 0.08), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(231, 111, 81, 0.08), transparent 25%),
            var(--bg-color);
        color: var(--text-main);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border-color);
        box-shadow: 8px 0 24px rgba(0, 0, 0, 0.04);
    }
    .sidebar-brand {
        padding: 14px 16px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(15, 139, 141, 0.12), rgba(231, 111, 81, 0.12));
        border: 1px solid rgba(15, 139, 141, 0.2);
        margin-bottom: 14px;
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f8b8d;
        margin-bottom: 4px;
    }
    .sidebar-subtitle {
        font-size: 0.9rem;
        color: #5a5a5a;
    }
    .sidebar-card {
        border-radius: 14px;
        padding: 14px;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid var(--border-color);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04);
    }
    .sidebar-card-title {
        font-weight: 700;
        margin-bottom: 8px;
        color: #1c1c1c;
    }
    .sidebar-row {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #5a5a5a;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #0f8b8d;
        box-shadow: 0 0 0 4px rgba(15, 139, 141, 0.15);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', 'Cairo', sans-serif;
        color: var(--text-main);
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
        margin-bottom: 30px;
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        backdrop-filter: blur(12px);
        box-shadow: var(--shadow-soft);
        animation: fadeInDown 0.8s ease-out;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-muted);
        max-width: 600px;
        margin: 0 auto 20px auto;
        line-height: 1.6;
    }

    .badge-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    .custom-badge {
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        background: rgba(15, 139, 141, 0.1);
        color: #0f8b8d;
        border: 1px solid rgba(15, 139, 141, 0.2);
        transition: transform 0.2s ease;
    }
    .custom-badge:hover {
        transform: translateY(-2px);
    }

    /* Main Chat Layout */
    .chat-card {
        background: var(--card-bg);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        padding: 30px;
        box-shadow: var(--shadow-soft);
        backdrop-filter: blur(10px);
        min-height: 500px;
    }

    /* Answer Styling (RTL for Arabic with English mixed) */
    .answer-box {
        background: #ffffff;
        border-right: 5px solid #0f8b8d;
        border-radius: var(--radius-md);
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        animation: slideInUp 0.5s ease-out;
    }

    .answer-text {
        direction: rtl; /* Enforce RTL */
        text-align: right; /* Enforce right alignment */
        font-size: 1.15rem;
        line-height: 1.8;
        font-family: 'Cairo', sans-serif;
        color: #2c3e50;
    }

    /* English words in Arabic text fallback */
    .answer-text * {
        unicode-bidi: embed;
    }

    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 15px;
    }
    
    .status-yes {
        background: rgba(231, 111, 81, 0.15);
        color: #d35400;
        border: 1px solid rgba(231, 111, 81, 0.3);
    }
    
    .status-no {
        background: rgba(15, 139, 141, 0.15);
        color: #0f8b8d;
        border: 1px solid rgba(15, 139, 141, 0.3);
    }

    .source-chip {
        display: inline-block;
        background: #f1f2f6;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #57606f;
        margin-top: 10px;
        margin-right: 8px;
        border: 1px solid #dfe4ea;
    }

    /* Streamlit overrides */
    .stTextInput input {
        border-radius: var(--radius-md) !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        padding: 16px 20px !important;
        font-size: 1.1rem !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput input:focus {
        border-color: #0f8b8d !important;
        box-shadow: 0 0 0 3px rgba(15, 139, 141, 0.2) !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: var(--radius-md);
        border: none;
        padding: 14px 24px;
        background: var(--primary-gradient);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(15, 139, 141, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(15, 139, 141, 0.4);
        color: white;
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Hide default radio buttons visually to make them look like tabs in the sidebar */
    .stRadio > div {
        gap: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# 🧭 SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-title">NileTel Support</div>
            <div class="sidebar-subtitle">AI Operations Console</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("## Navigation")
    page = st.radio(
        "Choose a View",
        ["💬 Chat Assistant", "🎫 Tickets Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">System Status</div>
            <div class="sidebar-row"><span class="status-dot"></span> Live API</div>
            <div class="sidebar-row">Data refresh: 60s</div>
            <div class="sidebar-row">Languages: AR / EN</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# 🌟 MAIN VIEW ROUTING
# =========================
if page == "💬 Chat Assistant":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">NileTel AI Assistant</div>
            <div class="hero-subtitle">Experience next-generation telecom support. Smart routing, instant hybrid-RAG answers, and automated ticket dispatch.</div>
            <div class="badge-container">
                <div class="custom-badge">🤖 Llama 3.1 8B</div>
                <div class="custom-badge">⚡ Hybrid Search</div>
                <div class="custom-badge">🌍 Arabic Native</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💬 Ask the AI")
    st.markdown("<p style='color:#5a5a5a; margin-bottom: 20px;'>Type your inquiry in Arabic or English. The AI will respond in a beautifully formatted RTL format.</p>", unsafe_allow_html=True)
    
    query = st.text_input("Your question...", placeholder="مثال: النت فاصل عندي، ممكن أعمل تذكرة؟", label_visibility="collapsed")
    
    # Send button
    send_clicked = st.button("🚀 Send Message")

    if send_clicked:
        if query.strip():
            st.session_state.ticket_name = ""
            st.session_state.last_response = None
            st.session_state.last_query = query.strip()
            with st.spinner("AI is thinking..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={
                            "query": st.session_state.last_query
                        }
                    )

                    if response.status_code == 200:
                        st.session_state.last_response = response.json()

                    else:
                        st.error(f"API Error {response.status_code}: Could not fetch answer.")
                except Exception as e:
                    st.error(f"Connection Failed: Ensure the FastAPI server is running. Error: {e}")
        else:
            st.warning("Please enter a question to get started.")

    if st.session_state.last_response:
        data = st.session_state.last_response
        answer_text = data.get("answer", "")
        needs_action = data.get("needs_action", "NO")
        sources = data.get("sources", [])

        render_answer = True
        if needs_action == "YES" and not st.session_state.ticket_name:
            render_answer = False
            st.markdown("### Provide your name to create the ticket")
            ticket_name = st.text_input("Name for ticket", key="ticket_name_input", placeholder="Your name")
            submit_name = st.button("Submit Name & Create Ticket", key="submit_ticket_name")

            if submit_name:
                if ticket_name.strip():
                    with st.spinner("Submitting ticket..."):
                        try:
                            response = requests.post(
                                API_URL,
                                json={
                                    "query": st.session_state.last_query,
                                    "name": ticket_name.strip()
                                }
                            )
                            if response.status_code == 200:
                                st.session_state.ticket_name = ticket_name.strip()
                                st.session_state.last_response = response.json()
                                data = st.session_state.last_response
                                answer_text = data.get("answer", "")
                                needs_action = data.get("needs_action", "NO")
                                sources = data.get("sources", [])
                                render_answer = True
                                st.success("Ticket request sent with your name.")
                            else:
                                st.error(f"API Error {response.status_code}: Could not submit ticket.")
                        except Exception as e:
                            st.error(f"Ticket submit failed. Error: {e}")
                else:
                    st.warning("Please enter your name to continue.")

        if render_answer:
            # Convert newlines to HTML breaks
            answer_html = answer_text.replace("\n", "<br>")

            # Render Answer Block
            st.markdown(
                f"""
                <div class="answer-box">
                    <div class="answer-text" dir="rtl">{answer_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Render Action Badge
            if needs_action == "YES":
                st.markdown("<div class='status-badge status-yes'>⚠️ Action Triggered (Ticket/Engineer)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-badge status-no'>✅ No Action Required</div>", unsafe_allow_html=True)

            # Render Sources
            if sources:
                chips_html = "".join([f"<span class='source-chip'>📄 {html.escape(str(s))}</span>" for s in sources])
                st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

elif page == "🎫 Tickets Dashboard":
    st.markdown("### 🎫 Tickets Dashboard")
    st.markdown("<p style='color:#5a5a5a; margin-bottom: 20px;'>View and manage all telecom support tickets fetched live from the central system.</p>", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False, ttl=60)
    def _load_tickets(url):
        return pd.read_csv(url)

    with st.spinner("Loading tickets from live database..."):
        try:
            df = _load_tickets(SHEET_URL)
            if df.empty:
                st.info("No tickets available at the moment.")
            else:
                st.success(f"Successfully loaded {len(df)} tickets.")
                st.dataframe(
                    df, 
                    use_container_width=True, 
                    hide_index=True,
                    height=600
                )
        except Exception as e:
            st.error(f"Failed to load tickets. Error: {e}")
