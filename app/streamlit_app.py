"""
Streamlit app — Recommendation System Management Interface.
Calls Azure Function and displays top-5 recommendations.

Run locally:
    streamlit run app/streamlit_app.py

Environment variables:
    AZURE_FUNCTION_URL : Base URL of the Azure Function
                         (default: http://localhost:7071/api)
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FUNCTION_URL = os.environ.get("AZURE_FUNCTION_URL", "http://localhost:7071/api")

TOP_USERS = [
    5890, 73574, 80350, 15275, 48723,
    15867, 4568, 169, 34541, 57221,
    3391, 25258, 7349, 12897, 4966,
    9373, 9261, 44967, 48448, 59193,
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Recommendation System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f4f6f9;
        --surface: #ffffff;
        --border: #d8dee9;
        --text: #1f2937;
        --text-muted: #6b7280;
        --accent: #1d4ed8;
        --accent-soft: #dbeafe;
        --accent-hover: #1e40af;
        --success: #047857;
        --error: #b91c1c;
        --error-bg: #fef2f2;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    .stApp {
        background-color: var(--bg);
    }

    #MainMenu, footer { visibility: hidden; }

    section[data-testid="stMain"] .block-container {
        padding-top: 4.5rem;
        max-width: 1100px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 4rem;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.25rem;
    }

    /* Header */
    .app-header {
        margin-bottom: 2rem;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* Cards */
    .article-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .article-card:hover {
        border-color: #93c5fd;
    }

    .rank {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-muted);
        min-width: 24px;
        text-align: center;
        background: #f3f4f6;
        border-radius: 6px;
        padding: 4px 0;
    }
    .article-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: var(--text);
        flex: 1;
    }
    .article-badge {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid #bfdbfe;
        padding: 3px 10px;
        border-radius: 999px;
    }

    .meta-line {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    .meta-line strong {
        color: var(--text);
        font-weight: 600;
    }

    .empty-state {
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: 12px;
        padding: 4rem 2rem;
        text-align: center;
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        font-size: 0.875rem;
        color: var(--text-muted);
        line-height: 1.7;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .info-card-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--accent);
        margin-bottom: 0.75rem;
    }
    .info-card strong {
        color: var(--text);
    }

    .error-box {
        background: var(--error-bg);
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 0.875rem;
        color: var(--error);
        line-height: 1.6;
    }

    .sidebar-footer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
        word-break: break-all;
        line-height: 1.5;
    }
    .sidebar-footer span {
        color: var(--text);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Parameters")

    beta = st.slider(
        "Beta (popularity weight)",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="0 = pure content-based · 1 = pure popularity · optimal: 0.8",
    )

    topk = st.selectbox("Top-K", options=[5, 10, 20], index=0)

    st.divider()
    st.markdown("### Select a user")

    selected_user = st.selectbox(
        "User ID",
        options=TOP_USERS,
        format_func=lambda x: f"User #{x}",
        label_visibility="collapsed",
    )

    get_recs = st.button("Get recommendations", use_container_width=True, type="primary")

    st.divider()
    st.markdown(
        f'<div class="sidebar-footer">Function URL<br><span>{FUNCTION_URL}</span></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <p class="app-title">Recommendation System</p>
        <p class="app-subtitle">Content-based filtering + popularity · Globo.com dataset</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    if not get_recs:
        st.markdown(
            """
            <div class="empty-state">
                Select a user in the sidebar, then click <strong>Get recommendations</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        url = f"{FUNCTION_URL}/recommend/{selected_user}"
        params = {"beta": beta, "topk": topk}

        with st.spinner(f"Fetching recommendations for user #{selected_user}..."):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                recommendations = data.get("recommendations", [])

                st.markdown(
                    f"""
                    <div class="meta-line">
                        <strong>User #{selected_user}</strong>
                        · {len(recommendations)} recommendations
                        · beta = {beta}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for i, article_id in enumerate(recommendations):
                    st.markdown(
                        f"""
                        <div class="article-card">
                            <span class="rank">{i + 1}</span>
                            <span class="article-id">{article_id}</span>
                            <span class="article-badge">article</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except requests.Timeout:
                st.markdown(
                    f"""
                    <div class="error-box">
                        Timeout — the Azure Function did not respond within 30s.<br>
                        Is it running at <strong>{FUNCTION_URL}</strong>?
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except requests.HTTPError as e:
                st.markdown(
                    f"""
                    <div class="error-box">
                        HTTP error {e.response.status_code}<br>
                        {e.response.text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except requests.ConnectionError:
                st.markdown(
                    f"""
                    <div class="error-box">
                        Connection refused — Azure Function unreachable.<br>
                        Check that it is running at <strong>{FUNCTION_URL}</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with col2:
    if get_recs:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-title">Model info</div>
                <strong>Strategy:</strong> CB Category<br><br>
                <strong>PCA:</strong> 33 components<br>
                (85% variance retained)<br><br>
                <strong>Catalogue:</strong> 364,047 articles<br>
                <strong>Users:</strong> 64,734 profiles<br><br>
                Higher beta → more popular<br>
                Lower beta → more semantic
            </div>
            """,
            unsafe_allow_html=True,
        )
