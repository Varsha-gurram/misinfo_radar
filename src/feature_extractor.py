"""
feature_extractor.py
====================
Transforms a thread dict + early-window reactions into a flat numeric feature
vector ready for scikit-learn / XGBoost.

Feature groups
──────────────
1. Text        – VADER sentiment, uncertainty/negation counts, TF-IDF
2. Temporal    – reply velocity, inter-reply gaps, burst detection
3. Structural  – tree depth, branching, structural virality (NetworkX)
4. User        – follower counts, verified ratio, account age
"""

import re
import math
import logging
from typing import Optional

import numpy as np
import networkx as nx
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

from src.data_parser import build_tree, apply_early_window_n, apply_early_window_t

logger = logging.getLogger(__name__)

# ─────────────────────────── word lists ────────────────────────────────────

UNCERTAINTY_WORDS = {
    "allegedly", "reportedly", "unconfirmed", "rumour", "rumor",
    "claim", "claimed", "claims", "suggests", "suggestion",
    "apparently", "supposedly", "possible", "possibly", "maybe",
    "perhaps", "uncertain", "unclear", "speculated", "speculation",
    "said to", "believed", "believe", "they say", "heard",
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nothing", "nowhere",
    "nobody", "none", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't",
    "false", "fake", "hoax", "wrong", "incorrect", "deny", "denied",
    "debunked", "misleading",
}

QUESTION_PATTERN = re.compile(r"\?")
EXCLAIM_PATTERN = re.compile(r"!")
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")


# ─────────────────────────── singleton ─────────────────────────────────────

_vader: Optional[SentimentIntensityAnalyzer] = None


def _get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


# ─────────────────────────── text features ─────────────────────────────────

def extract_text_features(source_text: str, reaction_texts: list[str]) -> dict:
    """
    Returns a dict of scalar text features.
    TF-IDF vectorisation is done externally (needs a fitted vectoriser).
    """
    all_texts = [source_text] + reaction_texts
    combined = " ".join(all_texts)
    tokens = combined.lower().split()

    vader = _get_vader()

    src_scores = vader.polarity_scores(source_text)
    reply_scores = [vader.polarity_scores(t) for t in reaction_texts] or [{}]

    mean_compound = np.mean([s.get("compound", 0) for s in reply_scores])
    src_compound = src_scores.get("compound", 0)

    # lexical counts
    uncertainty_count = sum(
        1 for t in tokens if any(u in t for u in UNCERTAINTY_WORDS)
    )
    negation_count = sum(1 for t in tokens if t in NEGATION_WORDS)
    question_count = len(QUESTION_PATTERN.findall(combined))
    exclaim_count = len(EXCLAIM_PATTERN.findall(combined))
    url_count = len(URL_PATTERN.findall(combined))
    mention_count = len(MENTION_PATTERN.findall(combined))
    hashtag_count = len(HASHTAG_PATTERN.findall(combined))

    avg_reply_len = (
        np.mean([len(t.split()) for t in reaction_texts]) if reaction_texts else 0
    )
    src_len = len(source_text.split())

    # sentiment polarity spread across replies
    compounds = [s.get("compound", 0) for s in reply_scores]
    sentiment_std = float(np.std(compounds)) if len(compounds) > 1 else 0.0
    # proportion of negative replies
    neg_ratio = sum(1 for c in compounds if c < -0.05) / max(1, len(compounds))
    # proportion of positive replies
    pos_ratio = sum(1 for c in compounds if c > 0.05) / max(1, len(compounds))

    return {
        # source sentiment
        "src_vader_compound": src_compound,
        "src_vader_pos": src_scores.get("pos", 0),
        "src_vader_neg": src_scores.get("neg", 0),
        "src_vader_neu": src_scores.get("neu", 0),
        # reply sentiment
        "mean_reply_compound": mean_compound,
        "sentiment_std": sentiment_std,
        "neg_reply_ratio": neg_ratio,
        "pos_reply_ratio": pos_ratio,
        # lexical
        "uncertainty_count": uncertainty_count,
        "negation_count": negation_count,
        "question_count": question_count,
        "exclaim_count": exclaim_count,
        "url_count": url_count,
        "mention_count": mention_count,
        "hashtag_count": hashtag_count,
        # length
        "src_word_count": src_len,
        "avg_reply_word_count": avg_reply_len,
        "total_word_count": len(tokens),
    }


# ─────────────────────────── temporal features ─────────────────────────────

def extract_temporal_features(early_reactions: list[dict]) -> dict:
    """Inter-reply gap statistics and reply velocity."""
    delays = [
        r["delay_seconds"]
        for r in early_reactions
        if r["delay_seconds"] is not None
    ]

    if not delays:
        return {
            "n_replies": 0,
            "time_to_first_reply": 0,
            "mean_gap": 0,
            "std_gap": 0,
            "max_gap": 0,
            "reply_velocity": 0,
            "burst_score": 0,
            "span_seconds": 0,
        }

    delays.sort()
    gaps = [delays[i] - delays[i - 1] for i in range(1, len(delays))]

    span = delays[-1] if delays else 0
    velocity = len(delays) / max(1, span / 60)  # replies per minute
    std_gap = float(np.std(gaps)) if gaps else 0.0
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    # burst score: low std relative to mean → rapid burst
    burst_score = 1.0 / (1.0 + std_gap / max(1, mean_gap))

    return {
        "n_replies": len(delays),
        "time_to_first_reply": delays[0],
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "max_gap": float(max(gaps)) if gaps else 0.0,
        "reply_velocity": velocity,
        "burst_score": burst_score,
        "span_seconds": span,
    }


# ─────────────────────────── structural features ───────────────────────────

def _structural_virality(G: nx.DiGraph) -> float:
    """
    Average shortest-path length between all node pairs in the undirected tree.
    Wiener index / n*(n-1) — a proxy for how tree-like vs star-like the thread is.
    """
    if len(G) <= 1:
        return 0.0
    UG = G.to_undirected()
    try:
        # work only on the largest connected component
        largest_cc = max(nx.connected_components(UG), key=len)
        sub = UG.subgraph(largest_cc)
        if len(sub) <= 1:
            return 0.0
        total = sum(
            d for _, path_len in nx.all_pairs_shortest_path_length(sub)
            for _, d in path_len.items()
        )
        n = len(sub)
        return total / (n * (n - 1)) if n > 1 else 0.0
    except Exception:
        return 0.0


def extract_structural_features(G: nx.DiGraph, source_id: str) -> dict:
    """Tree topology features."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes <= 1:
        return {
            "tree_depth": 0,
            "avg_branching": 0,
            "n_leaf_nodes": 0,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "structural_virality": 0,
            "max_outdegree": 0,
            "avg_outdegree": 0,
        }

    # depth via BFS from source
    try:
        lengths = nx.single_source_shortest_path_length(G, source_id)
        max_depth = max(lengths.values()) if lengths else 0
    except Exception:
        max_depth = 0

    # leaf nodes = nodes with out-degree 0 (no children)
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    n_leaves = len(leaves)

    # branching factor
    out_degrees = [G.out_degree(n) for n in G.nodes if G.out_degree(n) > 0]
    avg_branching = float(np.mean(out_degrees)) if out_degrees else 0.0
    max_outdegree = int(max(out_degrees)) if out_degrees else 0
    avg_outdegree = float(np.mean([G.out_degree(n) for n in G.nodes]))

    sv = _structural_virality(G)

    return {
        "tree_depth": max_depth,
        "avg_branching": avg_branching,
        "n_leaf_nodes": n_leaves,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "structural_virality": sv,
        "max_outdegree": max_outdegree,
        "avg_outdegree": avg_outdegree,
    }


# ─────────────────────────── user features ─────────────────────────────────

def extract_user_features(source: dict, early_reactions: list[dict]) -> dict:
    """Aggregate user credibility signals."""
    src_time = source.get("created_at")

    all_users = [source] + early_reactions
    followers = [u.get("user_followers", 0) or 0 for u in all_users]
    verified = [u.get("user_verified", 0) or 0 for u in all_users]

    # account age for each user
    ages = []
    for u in all_users:
        if src_time and u.get("user_created_at"):
            from utils.helpers import account_age_days
            age = account_age_days(u["user_created_at"], src_time)
            ages.append(age)

    return {
        "src_followers": source.get("user_followers", 0) or 0,
        "src_verified": source.get("user_verified", 0) or 0,
        "max_followers": float(max(followers)) if followers else 0,
        "mean_followers": float(np.mean(followers)) if followers else 0,
        "log_max_followers": math.log1p(max(followers)) if followers else 0,
        "verified_ratio": float(np.mean(verified)) if verified else 0,
        "n_verified": int(sum(verified)),
        "mean_account_age_days": float(np.mean(ages)) if ages else 0,
        "min_account_age_days": float(min(ages)) if ages else 0,
    }


# ─────────────────────────── combined extractor ────────────────────────────

def extract_features(thread: dict, early_reactions: list[dict]) -> dict:
    """
    Full feature extraction pipeline for one thread.
    Returns a flat dict of feature_name → float.
    """
    src = thread["source"]
    src_id = src["id"]

    reaction_texts = [r["text"] for r in early_reactions]

    # build tree
    G = build_tree(thread, early_reactions)

    text_feats = extract_text_features(src["text"], reaction_texts)
    temp_feats = extract_temporal_features(early_reactions)
    struct_feats = extract_structural_features(G, src_id)
    user_feats = extract_user_features(src, early_reactions)

    combined = {}
    combined.update({f"txt_{k}": v for k, v in text_feats.items()})
    combined.update({f"tmp_{k}": v for k, v in temp_feats.items()})
    combined.update({f"str_{k}": v for k, v in struct_feats.items()})
    combined.update({f"usr_{k}": v for k, v in user_feats.items()})

    # meta
    combined["event"] = thread["event"]
    combined["thread_id"] = thread["thread_id"]
    combined["label_binary"] = thread["label_binary"]
    combined["label_3class"] = thread["label_3class"]
    combined["label_str"] = thread["label_str"]

    return combined


def build_feature_matrix(
    threads,
    n: int = 10,
    strategy: str = "n",
    t_minutes: float = 60.0,
) -> "pd.DataFrame":
    """
    Build a DataFrame of features for all threads.

    Parameters
    ----------
    threads   : list of parsed thread dicts
    n         : number of early replies (strategy='n')
    strategy  : 'n' or 't'
    t_minutes : minutes cutoff (strategy='t')
    """
    import pandas as pd
    from tqdm import tqdm

    rows = []
    for thread in tqdm(threads, desc=f"Extracting features (N={n})"):
        if strategy == "n":
            early = apply_early_window_n(thread, n)
        else:
            early = apply_early_window_t(thread, t_minutes)
        feats = extract_features(thread, early)
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────── feature names ─────────────────────────────────

FEATURE_COLS = None  # populated at runtime after first build


def get_feature_cols(df: "pd.DataFrame") -> list[str]:
    """Return numeric feature column names (excluding meta columns)."""
    meta = {"event", "thread_id", "label_binary", "label_3class", "label_str"}
    return [c for c in df.columns if c not in meta]
