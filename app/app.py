"""
app.py  —  Streamlit entrypoint
Early Prediction of Misinformation Before It Goes Viral
"""

import os, sys, pickle, warnings, logging
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import requests

def fetch_tweet_data(tweet_id):
    urls = [
        f"https://react-tweet.vercel.app/api/tweet/{tweet_id}",
        f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}"
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                
                # handle different formats
                if "data" in data:
                    return data["data"]
                return data

        except Exception:
            continue

    return None

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MisinfoRadar — Early Rumour Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components import (
    inject_custom_css,
    render_gauge_chart,
    render_shap_waterfall,
    render_conversation_tree
)

inject_custom_css()

# ── constants ──────────────────────────────────────────────────────────────────
PROCESSED_PATH = "data/pheme_processed.pkl"
MODEL_PATHS = {
    5:  "models/xgb_model_N5.joblib",
    10: "models/xgb_model_N10.joblib",
    20: "models/xgb_model_N20.joblib",
}
CV_RESULTS_PATH = "results/cv_results.pkl"
EVENTS = ["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege"]

# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading dataset…")
def load_threads():
    if not os.path.exists(PROCESSED_PATH):
        return None
    with open(PROCESSED_PATH,"rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading model…")
def load_model(n):
    path = MODEL_PATHS.get(n)
    if path and os.path.exists(path):
        import joblib
        obj = joblib.load(path)
        return obj["model"], obj["feature_cols"]
    return None, None

@st.cache_resource(show_spinner="Loading CV results…")
def load_cv():
    if os.path.exists(CV_RESULTS_PATH):
        with open(CV_RESULTS_PATH,"rb") as f:
            return pickle.load(f)
    return None

# ── sidebar nav ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 MisinfoRadar")
    st.markdown("<p style='color:#6366f1;font-size:.8rem;margin-top:-.5rem'>Early Rumour Detection</p>",
                unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", ["🎯 Predict", "📊 Compare", "🧪 Live Test", "ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.markdown("<p style='color:#475569;font-size:.72rem'>PHEME Dataset · XGBoost · SHAP XAI</p>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if page == "🎯 Predict":
    st.markdown("# 🎯 Predict Rumour Risk")
    st.markdown("Select a thread and early-window size to see the risk prediction and explanation.")

    threads = load_threads()
    data_ready = threads is not None

    if not data_ready:
        st.warning("⚠️ No pre-processed data found at `data/pheme_processed.pkl`. "
                   "Run `python train.py --data <PHEME_PATH>` first.")
        st.info("**Demo mode** — showing synthetic example while you set up the dataset.")

    col_l, col_r = st.columns([1,2])

    with col_l:
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

        if data_ready:
            # filter to rumour threads only (interesting ones)
            rumour_threads = [t for t in threads if t["rumour"]]
            event_filter = st.selectbox("Filter by event", ["All"] + EVENTS)
            if event_filter != "All":
                rumour_threads = [t for t in rumour_threads if t["event"] == event_filter]

            thread_options = {
                f"{t['thread_id']} ({t['event']}, {t['label_str']})": t
                for t in rumour_threads[:200]
            }
            chosen_key = st.selectbox("Select thread", list(thread_options.keys()))
            thread = thread_options[chosen_key]
        else:
            thread = None

        n_replies = st.select_slider(
            "Early window (N replies)", options=[5, 10, 20], value=10
        )

        predict_btn = st.button("🔮 Predict", use_container_width=True, type="primary")

    # ── run prediction ─────────────────────────────────────────────────────
    if predict_btn or (data_ready and thread):
        with col_r:
            if data_ready and thread:
                from src.data_parser import apply_early_window_n, build_tree
                from src.feature_extractor import extract_features, get_feature_cols
                from src.explainer import MisinfoExplainer

                early = apply_early_window_n(thread, n_replies)
                feats = extract_features(thread, early)

                model, feat_cols = load_model(n_replies)

                if model is None:
                    st.warning(f"Model for N={n_replies} not found. Run train.py first.")
                else:
                    from src.model import predict_thread
                    label, risk = predict_thread(model, feat_cols, feats)

                    # ── risk gauge ─────────────────────────────────────────
                    st.markdown('<div class="section-header">Risk Assessment</div>',
                                unsafe_allow_html=True)

                    risk_pct = int(risk * 100)
                    risk_cls = "risk-low" if risk < 0.4 else ("risk-medium" if risk < 0.7 else "risk-high")
                    risk_label = "✅ Likely True" if label == 0 else "🚨 Likely False/Unverified"
                    badge_cls = "badge-green" if label == 0 else "badge-red"

                    render_gauge_chart(risk)

                    c1,c2,c3 = st.columns(3)
                    c1.markdown(f'<div class="metric-card"><h2>{risk_pct}%</h2><p>Risk Score</p></div>',
                                unsafe_allow_html=True)
                    c2.markdown(f'<div class="metric-card"><h2>{n_replies}</h2><p>Replies Used</p></div>',
                                unsafe_allow_html=True)
                    c3.markdown(f'<div class="metric-card"><h2>{len(thread["reactions"])}</h2><p>Total Replies</p></div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<span class="badge {badge_cls}">{risk_label}</span>',
                                unsafe_allow_html=True)
                    st.markdown(f"**Ground truth:** `{thread['label_str']}`")

                    st.divider()

                    # ── SHAP local waterfall ───────────────────────────────
                    st.markdown('<div class="section-header">Explanation — What drove this prediction?</div>',
                                unsafe_allow_html=True)
                    try:
                        explainer = MisinfoExplainer(model, feat_cols)
                        render_shap_waterfall(explainer, feats, top_n=12)

                        # tooltip table
                        from src.explainer import get_tooltip, get_label, FEATURE_TOOLTIPS
                        sv_dict = dict(zip(feat_cols,
                                          explainer.local_explanation(feats)["shap_values"]))
                        top_feats = sorted(sv_dict, key=lambda k:abs(sv_dict[k]), reverse=True)[:5]
                        st.markdown("**Top contributing features:**")
                        for f in top_feats:
                            tip = get_tooltip(f)
                            direction = "🔴 increases risk" if sv_dict[f]>0 else "🟢 reduces risk"
                            val = feats.get(f,0)
                            st.markdown(
                                f"- **{get_label(f)}** = `{val:.2f}` → {direction}"
                                + (f" _{tip}_" if tip else "")
                            )
                    except Exception as e:
                        st.info(f"SHAP explanation unavailable: {e}")

                    st.divider()

                    # ── conversation tree ─────────────────────────────────
                    st.markdown('<div class="section-header">Conversation Tree (early window)</div>',
                                unsafe_allow_html=True)
                    G = build_tree(thread, early)
                    render_conversation_tree(G, thread)

                    # ── early tweets ──────────────────────────────────────
                    with st.expander("📜 View early replies"):
                        src_text = thread["source"]["text"]
                        st.markdown(f'<div class="tweet-card">🐦 <b>Source:</b> {src_text}</div>',
                                    unsafe_allow_html=True)
                        for rxn in early[:10]:
                            st.markdown(
                                f'<div class="tweet-card">↩ {rxn["text"][:140]}</div>',
                                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — COMPARE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Compare":
    st.markdown("# 📊 Performance Comparison")
    st.markdown("Compare how prediction quality changes with more replies available.")

    cv = load_cv()

    if cv is None:
        st.warning("CV results not found. Run `python train.py` first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Window Size vs F1","Per-Event Breakdown","Model Comparison"])

        with tab1:
            st.markdown("### F1 Score vs Early-Window Size (LOEO CV)")
            keys = [5,10,20,"full"]
            labels_x = ["N=5","N=10","N=20","Full thread"]
            model_map = {"lr":"Logistic Reg","rf":"Random Forest","xgboost":"XGBoost"}
            colors = {"lr":"#94a3b8","rf":"#f59e0b","xgboost":"#6366f1"}

            fig, axes = plt.subplots(1,2,figsize=(12,4.5))
            fig.patch.set_facecolor("#0f172a")
            for ax in axes:
                ax.set_facecolor("#1e293b")
                ax.tick_params(colors="#94a3b8")
                ax.xaxis.label.set_color("#94a3b8")
                ax.yaxis.label.set_color("#94a3b8")
                ax.title.set_color("#a5b4fc")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#334155")
                ax.grid(True, alpha=0.15, color="#334155")

            for mt, (name, color) in [(m,(_n,_c)) for m,_n,_c in
                                       [("lr","Logistic Reg","#94a3b8"),
                                        ("rf","Random Forest","#f59e0b"),
                                        ("xgboost","XGBoost","#6366f1")]]:
                f1_vals = [cv[k][mt]["overall"]["f1"] if k in cv and mt in cv[k] else np.nan
                           for k in keys]
                mac_vals = [cv[k][mt]["overall"]["f1_macro"] if k in cv and mt in cv[k] else np.nan
                            for k in keys]
                axes[0].plot(labels_x, f1_vals, "o-", label=name, color=color, lw=2, ms=7)
                axes[1].plot(labels_x, mac_vals, "o-", label=name, color=color, lw=2, ms=7)

            axes[0].set_title("F1 (binary)"); axes[0].set_ylim(0,1)
            axes[1].set_title("F1 (macro)");  axes[1].set_ylim(0,1)
            axes[0].legend(fontsize=9); axes[1].legend(fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            st.markdown("### Per-Event F1 — XGBoost N=10")
            n_sel = st.selectbox("Window size", [5,10,20], index=1, key="pe_n")
            if n_sel in cv and "xgboost" in cv[n_sel]:
                folds = cv[n_sel]["xgboost"]["folds"]
                evts = [f["event"] for f in folds]
                f1s  = [f["f1"] for f in folds]
                macs = [f["f1_macro"] for f in folds]

                fig2, ax2 = plt.subplots(figsize=(9,4))
                fig2.patch.set_facecolor("#0f172a"); ax2.set_facecolor("#1e293b")
                x = np.arange(len(evts)); w=.35
                ax2.bar(x-w/2, f1s,  w, label="F1 binary", color="#6366f1", alpha=.85)
                ax2.bar(x+w/2, macs, w, label="F1 macro",  color="#f59e0b", alpha=.85)
                ax2.set_xticks(x); ax2.set_xticklabels(evts, rotation=18, ha="right", color="#94a3b8")
                ax2.set_ylim(0,1); ax2.legend(fontsize=9)
                ax2.tick_params(colors="#94a3b8"); ax2.title.set_color("#a5b4fc")
                ax2.set_title(f"Per-Event Performance — XGBoost N={n_sel}")
                ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
                ax2.grid(True, axis="y", alpha=0.15, color="#334155")
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            # table
            rows=[]
            for k in [5,10,20,"full"]:
                for mt in ["lr","rf","xgboost"]:
                    if k in cv and mt in cv[k]:
                        ov=cv[k][mt]["overall"]
                        rows.append({"N": str(k), "Model":mt,
                                     "Acc":round(ov["accuracy"],3),
                                     "P":round(ov["precision"],3),
                                     "R":round(ov["recall"],3),
                                     "F1":round(ov["f1"],3),
                                     "F1-mac":round(ov["f1_macro"],3)})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with tab3:
            st.markdown("### Model Ablation — N=10")
            if 10 in cv:
                abl_rows = []
                for mt,name in [("lr","Logistic Regression"),("rf","Random Forest"),("xgboost","XGBoost")]:
                    if mt in cv[10]:
                        ov=cv[10][mt]["overall"]
                        abl_rows.append({"Model":name,
                                         "Accuracy":f"{ov['accuracy']:.3f}",
                                         "Precision":f"{ov['precision']:.3f}",
                                         "Recall":f"{ov['recall']:.3f}",
                                         "F1":f"{ov['f1']:.3f}",
                                         "F1-macro":f"{ov['f1_macro']:.3f}"})
                st.dataframe(pd.DataFrame(abl_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — LIVE TEST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧪 Live Test":
    st.markdown("# 🧪 Live Tweet Test")
    st.markdown("Extract a tweet from a URL or upload a JSON file to predict rumour risk.")

    input_method = st.radio("Input Method", ["🌐 Extract from URL", "📂 Upload JSON File"], horizontal=True)
    
    thread_to_predict = None

    if input_method == "🌐 Extract from URL":
        st.info("⚠️ Live extraction uses limited public APIs. Replies are simulated.")

        url_input = st.text_input("Twitter/X Thread URL", placeholder="https://twitter.com/user/status/123456789")

        if st.button("Extract & Predict", type="primary"):

            if not url_input.strip():
                st.error("Please enter a valid URL.")

            else:
                with st.spinner("Fetching tweet data..."):

                    import re, requests
                    from datetime import datetime, timezone

                    # Extract tweet ID
                    match = re.search(r"status/(\d+)", url_input)
                    if not match:
                        st.error("Invalid Twitter URL format.")
                        st.stop()

                    tweet_id = match.group(1)

                    # Fetch tweet
                    try:
                        api_url = f"https://react-tweet.vercel.app/api/tweet/{tweet_id}"
                        r = requests.get(api_url, timeout=5)

                        if r.status_code != 200:
                            st.error("Failed to fetch tweet.")
                            st.stop()

                        tweet_data = r.json().get("data", {})

                    except Exception as e:
                        st.error(f"Error fetching tweet: {e}")
                        st.stop()

                    # Extract fields
                    text = tweet_data.get("text", "")
                    author = tweet_data.get("user", {})

                    now = datetime.now(timezone.utc)

                    # Build thread
                    thread_to_predict = {
                        "thread_id": "live_" + tweet_id,
                        "event": "live_test",
                        "rumour": True,
                        "label_str": "unverified",
                        "label_binary": 1,
                        "label_3class": 2,
                        "source": {
                            "id": "src_1",
                            "text": text,
                            "created_at": tweet_data.get("created_at", now),
                            "user_followers": author.get("followers_count", 0),
                            "user_verified": 1 if author.get("verified", False) else 0,
                            "user_created_at": "",
                            "retweet_count": tweet_data.get("retweet_count", 0),
                        },
                        "reactions": []
                    }

                    # Simulate simple early replies
                    thread_to_predict["reactions"] = [
                        {"id": "r1", "text": "Is this true?", "created_at": now, "delay_seconds": 60, "in_reply_to": "src_1", "user_followers": 10, "user_verified": 0},
                        {"id": "r2", "text": "This looks fake", "created_at": now, "delay_seconds": 120, "in_reply_to": "src_1", "user_followers": 5, "user_verified": 0},
                        {"id": "r3", "text": "Any source?", "created_at": now, "delay_seconds": 180, "in_reply_to": "src_1", "user_followers": 50, "user_verified": 0},
                    ]

                    st.success("Tweet extracted successfully!")
                    # ── MODEL + SHAP PREDICTION ─────────────────────────────

                    from src.feature_extractor import extract_features
                    from src.model import predict_thread
                    from src.explainer import MisinfoExplainer

                    import pandas as pd

                    # Step 1: Extract features
                    features = extract_features(thread_to_predict, thread_to_predict["reactions"])

                    # Step 2: Load model (N=10)
                    model, feat_cols = load_model(10)

                    if model is None:
                        st.error("Model not found. Run train.py first.")
                        st.stop()

                    # Step 3: Align features
                    features_df = pd.DataFrame([features])[feat_cols]

                    # Step 4: Predict
                    label, risk = predict_thread(model, feat_cols, features_df.iloc[0].to_dict())

                    # ── SHOW RESULT ─────────────────────────────────────────

                    st.subheader("🔍 Prediction Result")

                    risk_pct = int(risk * 100)

                    if risk < 0.4:
                        st.success(f"✅ Low Risk ({risk_pct}%)")
                    elif risk < 0.7:
                        st.warning(f"⚠️ Medium Risk ({risk_pct}%)")
                    else:
                        st.error(f"🚨 High Risk ({risk_pct}%)")

                    # ── SHAP EXPLANATION ───────────────────────────────────

                    st.subheader("🧠 Why this prediction? (SHAP)")

                    try:
                        explainer = MisinfoExplainer(model, feat_cols)
                        fig = explainer.plot_local_waterfall(features, top_n=10)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"SHAP failed: {e}")
                

    else:
        st.markdown("Upload a standard PHEME-formatted JSON file containing `source` and `reactions` arrays.")
        uploaded_file = st.file_uploader("Upload Tweet Thread JSON", type=["json"])
        if uploaded_file is not None:
            import json
            try:
                data = json.load(uploaded_file)
                # Auto-format basic list to our thread structure if needed
                if isinstance(data, list) and len(data) > 0:
                    thread_to_predict = {
                        "thread_id": "upload", "event": "upload", "rumour": True,
                        "label_str": "unverified", "label_binary": 1, "label_3class": 2,
                        "source": data[0],
                        "reactions": data[1:]
                    }
                elif isinstance(data, dict) and "source" in data and "reactions" in data:
                    thread_to_predict = data
                    # ensure labels exist
                    thread_to_predict.setdefault("label_binary", 1)
                    thread_to_predict.setdefault("label_3class", 2)
                    thread_to_predict.setdefault("label_str", "unverified")
                else:
                    st.error("Invalid JSON format. Must contain 'source' and 'reactions'.")
            except Exception as e:
                st.error(f"Error parsing JSON: {e}")
                
        if thread_to_predict and st.button("🔮 Predict Risk", type="primary"):
            pass # Continues below

    if thread_to_predict is not None and (input_method == "🌐 Extract from URL" or (input_method == "📂 Upload JSON File" and uploaded_file is not None)):
        from src.feature_extractor import extract_features
        from src.explainer import MisinfoExplainer
        from src.model import predict_thread

        # Fix missing 'delay_seconds' in uploaded reactions by defaulting to indices if missing
        for i, r in enumerate(thread_to_predict.get("reactions", [])):
            if "delay_seconds" not in r or r["delay_seconds"] is None:
                r["delay_seconds"] = (i + 1) * 60

        model, feat_cols = load_model(10)
        if model is None:
            st.warning("Model N=10 not found. Please train models first.")
        else:
            feats = extract_features(thread_to_predict, thread_to_predict.get("reactions", [])[:10])
            label, risk = predict_thread(model, feat_cols, feats)

            st.divider()
            st.markdown("### 📊 Prediction Results")
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                risk_pct = int(risk * 100)
                risk_label = "✅ Likely True" if label == 0 else "🚨 Likely False/Unverified"
                badge_cls = "badge-green" if label == 0 else "badge-red"
                fill_color = "#22c55e" if risk<.4 else ("#f59e0b" if risk<.7 else "#ef4444")
                
                render_gauge_chart(risk)
                
                st.markdown(f'<div style="text-align:center"><span class="badge {badge_cls}">{risk_label}</span></div>', unsafe_allow_html=True)

            with r_col2:
                st.markdown("#### 🔍 SHAP Explanation (Why?)")
                try:
                    explainer = MisinfoExplainer(model, feat_cols)
                    render_shap_waterfall(explainer, feats, top_n=8)
                except Exception as e:
                    st.error(f"Could not generate SHAP plot: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About MisinfoRadar")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### 🎯 What this system does
**MisinfoRadar** predicts whether a Twitter rumour thread will be confirmed
**false or unverified** — using only the *first few replies*, before misinformation
has a chance to spread widely.

It operationalises the idea that early-conversation signals (how fast people reply,
what language they use, the shape of the reply tree) are predictive of eventual
veracity — even with just 5–20 replies.

### 📦 Dataset — PHEME
The [PHEME dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
contains annotated Twitter rumour threads from 5 real-world events:
- Charlie Hebdo shooting
- Ferguson unrest
- Germanwings plane crash
- Ottawa shooting
- Sydney siege

Each thread has a source tweet, a reply tree, and a veracity label
(`true`, `false`, `unverified`).
""")

    with col2:
        st.markdown("""
### 🔬 Features Used
| Group | Features |
|---|---|
| **Text** | VADER sentiment, uncertainty words, negation, questions |
| **Temporal** | Time-to-first-reply, burst score, reply velocity |
| **Structural** | Tree depth, branching factor, structural virality |
| **User** | Followers, verified ratio, account age |

### 🧠 Models
- **Logistic Regression** — interpretable baseline
- **Random Forest** — handles non-linear interactions
- **XGBoost** — primary model; best performance

### 📐 Evaluation
Leave-one-event-out cross-validation: train on 4 events, test on 1.
This avoids topic contamination and tests real generalisation.

### 🔍 Explainability
SHAP (SHapley Additive exPlanations) from `shap.TreeExplainer`
provides both global feature importance and local per-thread explanations.
""")

    st.divider()
    st.markdown("""
### ⚠️ Limitations
- PHEME includes reply trees only — not full retweet cascades
- Rumour threads are scarce for some events (e.g. Germanwings)
- Early window with N=5 may have too little signal for difficult cases
- Model may not generalise to non-English rumour events

### 📚 Citation
> Zubiaga et al. (2016). *Analysing How People Orient to and Spread Rumours in Social Media
> by Looking at Conversational Threads*. PLOS ONE.
""")


