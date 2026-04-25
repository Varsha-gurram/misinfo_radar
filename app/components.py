import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }

    /* main background */
    .main { background: #0f172a; }
    .block-container { padding-top: 1.5rem !important; }

    /* metric cards */
    .metric-card {
        background: linear-gradient(135deg,#1e1b4b,#312e81);
        border: 1px solid #4338ca;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(99,102,241,0.15);
    }
    .metric-card h2 { color:#a5b4fc; font-size:2rem; margin:0; font-weight:700; }
    .metric-card p  { color:#94a3b8; font-size:.8rem; margin:0; letter-spacing:.05em; text-transform:uppercase; }

    /* risk gauge colours */
    .risk-low    { color: #22c55e !important; }
    .risk-medium { color: #f59e0b !important; }
    .risk-high   { color: #ef4444 !important; }

    /* section headers */
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #a5b4fc;
        border-bottom: 1px solid #312e81; padding-bottom: .4rem; margin-bottom: 1rem;
    }

    /* badge */
    .badge {
        display:inline-block; padding:.25em .7em; border-radius:999px;
        font-size:.78rem; font-weight:600; letter-spacing:.04em;
    }
    .badge-red    { background:#fef2f2; color:#dc2626; }
    .badge-green  { background:#f0fdf4; color:#16a34a; }
    .badge-yellow { background:#fffbeb; color:#d97706; }

    /* tweet card */
    .tweet-card {
        background:#1e293b; border-left:3px solid #6366f1;
        border-radius:8px; padding:.8rem 1rem; margin-bottom:.5rem;
        font-size:.85rem; color:#cbd5e1;
    }
    </style>
    """, unsafe_allow_html=True)


def render_gauge_chart(risk: float):
    risk_pct = int(risk * 100)
    fill_color = "#22c55e" if risk < 0.4 else ("#f59e0b" if risk < 0.7 else "#ef4444")

    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, np.ones_like(theta) * 0.8, color="#1e293b", lw=18)

    fill_theta = np.linspace(np.pi, np.pi - risk * np.pi, 200)
    ax.plot(fill_theta, np.ones_like(fill_theta) * 0.8, color=fill_color, lw=18, solid_capstyle="round")

    ax.text(0, 0, f"{risk_pct}%", ha="center", va="center",
            fontsize=26, fontweight="bold", color=fill_color, transform=ax.transData)
    ax.set_ylim(0, 1)
    ax.axis("off")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_shap_waterfall(explainer, feats: dict, top_n: int = 12):
    try:
        fig_wf = explainer.plot_local_waterfall(feats, top_n=top_n)
        fig_wf.patch.set_facecolor("#0f172a")
        for ax in fig_wf.get_axes():
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8")
            ax.xaxis.label.set_color("#94a3b8")
            ax.title.set_color("#a5b4fc")
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")


def render_conversation_tree(G: nx.DiGraph, thread: dict):
    if len(G) <= 1:
        st.info("No replies to show in the conversation tree.")
        return

    fig_tree, ax_tree = plt.subplots(figsize=(8, 4))
    fig_tree.patch.set_facecolor("#0f172a")
    ax_tree.set_facecolor("#0f172a")

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    src_id = thread["source"]["id"]
    node_colors = [
        "#6366f1" if n == src_id else "#f59e0b"
        for n in G.nodes()
    ]

    nx.draw(G, pos, ax=ax_tree, with_labels=False,
            node_color=node_colors, node_size=80,
            edge_color="#334155", arrows=True, arrowsize=10)

    ax_tree.set_title("Purple=source, Yellow=replies", color="#94a3b8", fontsize=9)
    st.pyplot(fig_tree, use_container_width=True)
    plt.close(fig_tree)
