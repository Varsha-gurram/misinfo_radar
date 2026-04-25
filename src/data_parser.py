"""
data_parser.py
==============
Reconstructs conversation trees using NetworkX and applies early-window slicing.
"""

import networkx as nx

def build_tree(thread: dict, early_reactions: list) -> nx.DiGraph:
    """
    Build a directed conversation tree from source + early_reactions.
    Nodes carry tweet text; edges point from parent → child.
    """
    G = nx.DiGraph()
    src_id = thread["source"]["id"]
    G.add_node(src_id, text=thread["source"]["text"], is_source=True)

    id_set = {src_id}
    for rxn in early_reactions:
        id_set.add(rxn["id"])

    for rxn in early_reactions:
        nid = rxn["id"]
        G.add_node(nid, text=rxn["text"], is_source=False)
        parent = rxn["in_reply_to"]
        if parent in id_set:
            G.add_edge(parent, nid)
        else:
            # attach orphans to source
            G.add_edge(src_id, nid)

    return G

def apply_early_window_n(thread: dict, n: int) -> list:
    """Return first N reactions (sorted chronologically)."""
    return thread["reactions"][:n]

def apply_early_window_t(thread: dict, t_minutes: float) -> list:
    """Return reactions within the first T minutes of the source tweet."""
    t_seconds = t_minutes * 60
    return [
        r for r in thread["reactions"]
        if r["delay_seconds"] is not None and r["delay_seconds"] <= t_seconds
    ]
