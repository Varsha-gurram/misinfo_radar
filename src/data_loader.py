"""
data_loader.py
==============
Loads the PHEME dataset into thread structures.
"""

import os
import glob
import pickle
import logging
import concurrent.futures
from typing import Optional

from utils.helpers import parse_twitter_date, load_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EVENTS = [
    "charliehebdo",
    "ferguson",
    "germanwings-crash",
    "ottawashooting",
    "sydneysiege",
    "ebola-essien",
    "gurlitt",
    "prince-toronto",
    "putinmissing",
]

LABEL_MAP = {
    "true": 0,
    "false": 1,
    "unverified": 1,   # binary: 0=benign, 1=risky
}

LABEL_MAP_3 = {
    "true": 0,
    "false": 1,
    "unverified": 2,
}

def load_thread(thread_dir: str, event: str, rumour: bool) -> Optional[dict]:
    """
    Load a single thread directory into a structured dict.
    Returns None if the thread is malformed.
    """
    thread_id = os.path.basename(thread_dir)

    # ── source tweet ──────────────────────────────────────────────────────
    src_files = glob.glob(os.path.join(thread_dir, "source-tweets", "*.json"))
    if not src_files:
        logger.debug("No source tweet in %s", thread_dir)
        return None
    src = load_json(src_files[0])
    src_time = parse_twitter_date(src.get("created_at", ""))

    # ── annotation ────────────────────────────────────────────────────────
    ann_path = os.path.join(thread_dir, "annotation.json")
    label_str = "unverified"
    if os.path.exists(ann_path):
        ann = load_json(ann_path)
        if "true" in ann and "misinformation" in ann:
            t_val = str(ann.get("true", "0")).strip()
            m_val = str(ann.get("misinformation", "0")).strip()
            if t_val == "1":
                label_str = "true"
            elif m_val == "1":
                label_str = "false"
            else:
                label_str = "unverified"
        else:
            raw = (
                ann.get("veracity")
                or ann.get("label")
                or ann.get("true")
                or "unverified"
            )
            if isinstance(raw, bool):
                label_str = "true" if raw else "false"
            else:
                label_str = str(raw).lower().strip()

    label_binary = LABEL_MAP.get(label_str, 1)
    label_3class = LABEL_MAP_3.get(label_str, 2)

    # ── reactions ─────────────────────────────────────────────────────────
    reactions = []
    rxn_dir = os.path.join(thread_dir, "reactions")
    if os.path.isdir(rxn_dir):
        for rxn_file in glob.glob(os.path.join(rxn_dir, "*.json")):
            try:
                rxn = load_json(rxn_file)
                rxn_time = parse_twitter_date(rxn.get("created_at", ""))
                delay = (
                    (rxn_time - src_time).total_seconds()
                    if rxn_time and src_time
                    else None
                )
                reactions.append(
                    {
                        "id": str(rxn.get("id", rxn.get("id_str", ""))),
                        "text": rxn.get("text", ""),
                        "created_at": rxn_time,
                        "delay_seconds": delay,
                        "in_reply_to": str(
                            rxn.get("in_reply_to_status_id", "")
                            or rxn.get("in_reply_to_status_id_str", "")
                        ),
                        "user_followers": rxn.get("user", {}).get(
                            "followers_count", 0
                        )
                        or 0,
                        "user_verified": int(
                            rxn.get("user", {}).get("verified", False) or False
                        ),
                        "user_created_at": rxn.get("user", {}).get(
                            "created_at", ""
                        ),
                        "retweet_count": rxn.get("retweet_count", 0) or 0,
                    }
                )
            except Exception as e:
                logger.debug("Error loading reaction %s: %s", rxn_file, e)

    # sort reactions chronologically
    reactions.sort(key=lambda r: r["delay_seconds"] if r["delay_seconds"] is not None else float("inf"))

    return {
        "thread_id": thread_id,
        "event": event,
        "rumour": rumour,
        "label_str": label_str,
        "label_binary": label_binary,
        "label_3class": label_3class,
        "source": {
            "id": str(src.get("id", src.get("id_str", thread_id))),
            "text": src.get("text", ""),
            "created_at": src_time,
            "user_followers": src.get("user", {}).get("followers_count", 0) or 0,
            "user_verified": int(src.get("user", {}).get("verified", False) or False),
            "user_created_at": src.get("user", {}).get("created_at", ""),
            "retweet_count": src.get("retweet_count", 0) or 0,
        },
        "reactions": reactions,
    }


def load_pheme_dataset(data_root: str) -> list:
    """
    Walk the PHEME directory tree and return a list of thread dicts.
    """
    all_subdirs = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')
    ]

    event_dirs = {}
    for d in all_subdirs:
        clean = d.replace('-all-rnr-threads', '')
        event_dirs[clean] = os.path.join(data_root, d)

    for event in EVENTS:
        direct = os.path.join(data_root, event)
        if os.path.isdir(direct) and event not in event_dirs:
            event_dirs[event] = direct

    if not event_dirs:
        logger.error("No event folders found in %s", data_root)
        return []

    logger.info("Found events: %s", list(event_dirs.keys()))

    tasks = []
    for event, event_dir in event_dirs.items():
        for rumour_flag, sub in [(True, "rumours"), (False, "non-rumours")]:
            sub_dir = os.path.join(event_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            for thread_dir in os.listdir(sub_dir):
                if thread_dir.startswith('.'):
                    continue
                full_path = os.path.join(sub_dir, thread_dir)
                if not os.path.isdir(full_path):
                    continue
                tasks.append((full_path, event, rumour_flag))

    threads = []
    logger.info("Found %d threads to parse. Starting thread pool...", len(tasks))

    def _load_wrapper(args):
        try:
            return load_thread(*args)
        except Exception as e:
            logger.error("Failed loading %s: %s", args[0], e)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for thread in executor.map(_load_wrapper, tasks):
            if thread is not None:
                threads.append(thread)
                if len(threads) % 500 == 0:
                    logger.info("Parsed %d/%d threads...", len(threads), len(tasks))

    logger.info("Loaded %d threads total", len(threads))
    return threads

def save_processed(threads: list, path: str = "data/pheme_processed.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(threads, f)
    logger.info("Saved %d threads → %s", len(threads), path)


def load_processed(path: str = "data/pheme_processed.pkl") -> list:
    with open(path, "rb") as f:
        threads = pickle.load(f)
    logger.info("Loaded %d threads ← %s", len(threads), path)
    return threads
