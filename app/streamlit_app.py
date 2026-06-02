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
)

st.markdown("""
<style>
    /* Global */
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide Streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Title */
    .app-title {
        font-family: 'DM Mono', monospace;
        font-size: 1.1rem;
        font-weight: 500;
        color: #00e5a0;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .app-subtitle {
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #666;
        margin-bottom: 32px;
    }

    /* Article card */
    .article-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 16px;
        transition: border-color 0.15s;
    }
    .article-card:hover { border-color: #00e5a0; }

    .rank {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: #555;
        min-width: 20px;
        text-align: right;
    }
    .article-id {
        font-family: 'DM Mono', monospace;
        font-size: 1rem;
        color: #e8e8e8;
        flex: 1;
    }
    .article-badge {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        color: #00e5a0;
        background: #00e5a015;
        border: 1px solid #00e5a040;
        padding: 2px 10px;
        border-radius: 4px;
    }

    /* Error */
    .error-box {
        background: #ff4d4d10;
        border: 1px solid #ff4d4d;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #ff4d4d;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">▸ Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Content-Based + Popularity · Globo.com dataset</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Layout : sidebar + main
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Parameters")

    beta = st.slider(
        "Beta (popularity weight)",
        min_value=0.0, max_value=1.0,
        value=0.8, step=0.05,
        help="0 = CB pur · 1 = popularité pure · optimal: 0.8"
    )

    topk = st.selectbox("Top-K", options=[5, 10, 20], index=0)

    st.divider()
    st.markdown("### 👤 Select a user")

    selected_user = st.selectbox(
        "User ID",
        options=TOP_USERS,
        format_func=lambda x: f"User #{x}",
    )

    get_recs = st.button("▸ Get recommendations", use_container_width=True, type="primary")

    st.divider()
    st.markdown(f"""
    <div style="font-family: DM Mono, monospace; font-size: 0.7rem; color: #555;">
    Function URL<br>
    <span style="color: #888">{FUNCTION_URL}</span>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    if not get_recs:
        st.markdown("""
        <div style="
            border: 1px dashed #2a2a2a;
            border-radius: 8px;
            padding: 64px 32px;
            text-align: center;
            font-family: DM Mono, monospace;
            font-size: 0.8rem;
            color: #444;
        ">
            ← Select a user and click "Get recommendations"
        </div>
        """, unsafe_allow_html=True)

    else:
        url = f"{FUNCTION_URL}/recommend/{selected_user}"
        params = {"beta": beta, "topk": topk}

        with st.spinner(f"Calling Azure Function for user #{selected_user}..."):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                recommendations = data.get("recommendations", [])

                st.markdown(f"""
                <div style="
                    font-family: DM Mono, monospace;
                    font-size: 0.75rem;
                    color: #00e5a0;
                    margin-bottom: 16px;
                ">
                    User #{selected_user} · {len(recommendations)} recommendations · beta={beta}
                </div>
                """, unsafe_allow_html=True)

                for i, article_id in enumerate(recommendations):
                    st.markdown(f"""
                    <div class="article-card">
                        <span class="rank">{i + 1}</span>
                        <span class="article-id">{article_id}</span>
                        <span class="article-badge">article</span>
                    </div>
                    """, unsafe_allow_html=True)

            except requests.Timeout:
                st.markdown(f"""
                <div class="error-box">
                    ⚠ Timeout — Azure Function did not respond within 30s.<br>
                    Is it running at {FUNCTION_URL} ?
                </div>
                """, unsafe_allow_html=True)

            except requests.HTTPError as e:
                st.markdown(f"""
                <div class="error-box">
                    ⚠ HTTP Error {e.response.status_code}<br>
                    {e.response.text}
                </div>
                """, unsafe_allow_html=True)

            except requests.ConnectionError:
                st.markdown(f"""
                <div class="error-box">
                    ⚠ Connection refused — Azure Function unreachable.<br>
                    Check that it is running at {FUNCTION_URL}
                </div>
                """, unsafe_allow_html=True)

with col2:
    if get_recs:
        st.markdown("""
        <div style="
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            font-family: DM Mono, monospace;
            font-size: 0.75rem;
            color: #666;
        ">
        <div style="color: #00e5a0; margin-bottom: 12px;">ℹ Model info</div>
        Strategy: CB Category<br><br>
        PCA: 33 components<br>
        (85% variance retained)<br><br>
        Catalogue: 364,047 articles<br>
        Users: 64,734 profiles<br><br>
        <div style="color: #444; margin-top: 12px; font-size: 0.65rem;">
        Higher beta → more popular<br>
        Lower beta → more semantic
        </div>
        </div>
        """, unsafe_allow_html=True)
