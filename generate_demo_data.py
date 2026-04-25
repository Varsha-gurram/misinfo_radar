"""
generate_demo_data.py
=====================
Generates synthetic PHEME-like threads and trains demo models so the
Streamlit app is fully functional without the real PHEME download.

Run once:
    python generate_demo_data.py
"""

import os, sys, pickle, random, json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

sys.path.insert(0, ".")
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

random.seed(42)
np.random.seed(42)

EVENTS = ["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege"]
LABELS = ["true","false","unverified"]

RUMOUR_TEXTS = [
    "Breaking: Reports of shots fired near parliament building #breaking",
    "UNCONFIRMED: Gunman reportedly still at large, police say area not safe",
    "Sources claim attack was coordinated — details still unclear #alert",
    "Allegedly 3 people shot at shopping centre, ambulances on scene",
    "RUMOUR: Government building evacuated amid bomb threat, unconfirmed",
    "Reportedly multiple explosions heard downtown, cause unknown #news",
    "Claims circulating that suspect had links to extremist groups — unverified",
    "Breaking reports: hospital on lockdown after shooting incident nearby",
    "UNVERIFIED: Police scanner mentions gunshots fired near school",
    "Alleged eyewitness says saw suspect flee in blue vehicle",
]

SKEPTICAL_REPLIES = [
    "Can anyone confirm this? Waiting for official sources",
    "This is unverified. Don't share until confirmed",
    "No official statement yet. Be careful spreading this",
    "I'm near the area, nothing seems unusual",
    "Local news says nothing happened. Likely false alarm",
    "Police scanner not showing this. Skeptical",
    "Allegedly? That word is doing a lot of work here",
    "Multiple sources needed before I believe this",
    "Sounds like misinformation, wait for official word",
    "The account that posted this was created yesterday",
]

SUPPORT_REPLIES = [
    "I saw this too! Scary stuff",
    "My friend near there confirmed it",
    "Sharing this, people need to know!",
    "OMG this is serious, stay safe everyone",
    "Can confirm, my coworker just texted me",
    "Breaking news alert from local station confirms this",
    "Multiple eyewitnesses reporting the same thing",
    "Police on scene according to scanner feed",
    "Hospital staff confirming influx of patients",
    "Live updates: situation developing rapidly",
]

NEUTRAL_REPLIES = [
    "What's happening exactly? More details?",
    "Anyone have more information on this?",
    "Following this story closely",
    "Waiting for official statement",
    "Thoughts with those affected",
    "Checking other sources now",
    "Has anyone heard from people on the ground?",
    "News channels aren't covering this yet",
    "This just appeared in my feed",
    "Stay safe everyone in the area",
]

def random_twitter_date(base: datetime, max_delay_minutes: int = 120) -> datetime:
    delay = random.expovariate(1/10) * 60  # seconds, exponential gaps
    delay = min(delay, max_delay_minutes * 60)
    return base + timedelta(seconds=delay)

def make_thread(thread_id: str, event: str, label: str, n_reactions: int = None) -> dict:
    from src.data_parser import LABEL_MAP, LABEL_MAP_3
    n_reactions = n_reactions or random.randint(3, 35)
    src_time = datetime(2015, 1, random.randint(1,28), random.randint(8,22),
                        random.randint(0,59), tzinfo=timezone.utc)

    # Weighted reply mix based on label
    if label == "false":
        weights = [0.5, 0.1, 0.4]   # more skeptical
    elif label == "true":
        weights = [0.15, 0.6, 0.25]  # more supportive
    else:
        weights = [0.35, 0.3, 0.35]  # mixed

    pools = [SKEPTICAL_REPLIES, SUPPORT_REPLIES, NEUTRAL_REPLIES]

    reactions = []
    current_time = src_time
    for i in range(n_reactions):
        pool = random.choices(pools, weights=weights)[0]
        text = random.choice(pool)
        delay = random.expovariate(1/300) + 30  # at least 30s
        current_time = current_time + timedelta(seconds=delay)

        followers = int(np.random.lognormal(4, 2))
        account_age = random.randint(30, 3000)
        acct_created = current_time - timedelta(days=account_age)

        reactions.append({
            "id":            f"rxn_{thread_id}_{i}",
            "text":          text,
            "created_at":    current_time,
            "delay_seconds": (current_time - src_time).total_seconds(),
            "in_reply_to":   thread_id if random.random() < 0.7 else f"rxn_{thread_id}_{max(0,i-1)}",
            "user_followers": followers,
            "user_verified":  int(random.random() < 0.05),
            "user_created_at": acct_created.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "retweet_count":  random.randint(0, 50),
        })

    src_followers = int(np.random.lognormal(5, 2))
    src_acct_age  = random.randint(100, 5000)
    src_acct_time = src_time - timedelta(days=src_acct_age)

    return {
        "thread_id":    thread_id,
        "event":        event,
        "rumour":       True,
        "label_str":    label,
        "label_binary": LABEL_MAP.get(label, 1),
        "label_3class": LABEL_MAP_3.get(label, 2),
        "source": {
            "id":            thread_id,
            "text":          random.choice(RUMOUR_TEXTS),
            "created_at":    src_time,
            "user_followers": src_followers,
            "user_verified":  int(random.random() < 0.08),
            "user_created_at": src_acct_time.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "retweet_count":  random.randint(0, 200),
        },
        "reactions": reactions,
    }


def generate_threads(n_per_event: int = 40) -> list:
    threads = []
    label_dist = {"true": 0.35, "false": 0.40, "unverified": 0.25}
    for event in EVENTS:
        for i in range(n_per_event):
            label = random.choices(
                list(label_dist.keys()),
                weights=list(label_dist.values())
            )[0]
            tid = f"{event}_{i:04d}"
            threads.append(make_thread(tid, event, label))
    print(f"Generated {len(threads)} synthetic threads")
    return threads


def main():
    print("=== Generating demo data ===")
    threads = generate_threads(n_per_event=50)

    # save processed
    with open("data/pheme_processed.pkl","wb") as f:
        pickle.dump(threads, f)
    print("Saved → data/pheme_processed.pkl")

    # build feature matrices
    print("\n=== Building feature matrices ===")
    from src.feature_extractor import build_feature_matrix, get_feature_cols
    dfs = {}
    for n in [5, 10, 20]:
        dfs[n] = build_feature_matrix(threads, n=n, strategy="n")
        print(f"  N={n}: {dfs[n].shape}")

    max_n = max(len(t["reactions"]) for t in threads)
    dfs["full"] = build_feature_matrix(threads, n=max_n, strategy="n")

    # LOEO CV
    print("\n=== Running leave-one-event-out CV ===")
    from src.model import leave_one_event_out_cv, train_final_model
    all_results = {}
    for n in [5, 10, 20, "full"]:
        all_results[n] = {}
        df = dfs[n]
        for mt in ["lr","rf","xgboost"]:
            print(f"  N={n} | {mt} ...", end=" ", flush=True)
            res = leave_one_event_out_cv(df, model_type=mt)
            all_results[n][mt] = res
            print(f"F1={res['overall']['f1']:.3f}  F1-mac={res['overall']['f1_macro']:.3f}")

    with open("results/cv_results.pkl","wb") as f:
        pickle.dump(all_results, f)
    print("Saved → results/cv_results.pkl")

    # Train final models
    print("\n=== Training final XGBoost models ===")
    for n in [5, 10, 20]:
        train_final_model(dfs[n], model_type="xgboost",
                          save_path=f"models/xgb_model_N{n}.joblib")
        print(f"  Saved → models/xgb_model_N{n}.joblib")

    # Save feature cols
    feat_cols = get_feature_cols(dfs[10])
    with open("models/feature_cols.pkl","wb") as f:
        pickle.dump(feat_cols, f)

    print("\n✅ Demo data ready. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
