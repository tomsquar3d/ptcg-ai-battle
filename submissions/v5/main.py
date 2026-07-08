"""v5 submission entry point.

ONNX net-guided MCTS-H agent (see agent.py) wrapped in a watchdog: any exception or malformed input
falls back to a legal move so the agent can never crash or return an illegal action (crash == timeout
== instant loss). The network (value.onnx/policy.onnx, run on CPU via onnxruntime) guides the search;
if it or onnxruntime is unavailable at runtime the agent degrades to plain heuristic MCTS-H.
"""
import os
import random
import sys

# The Kaggle env exec's this file without setting __file__; the agent dir is already on
# sys.path there. When imported normally, add our own dir so `import agent` / `cg` resolve.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import agent as _brain  # noqa: E402


def read_deck_csv():
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"
    with open(file_path, "r") as f:
        csv = f.read().split("\n")
    return [int(csv[i]) for i in range(60)]


def _fallback(obs_dict):
    """Guaranteed-legal move with no dependencies on the brain."""
    try:
        sel = obs_dict.get("select")
        if sel is None:
            return read_deck_csv()
        n = len(sel["option"])
        if n == 0:
            return []
        return random.sample(range(n), min(sel["maxCount"], n))
    except Exception:
        return [0]


def agent(obs_dict):
    try:
        if obs_dict.get("select") is None:
            deck = read_deck_csv()
            _brain.set_deck(deck)
            return deck
        return _brain.choose(obs_dict, time_left=obs_dict.get("remainingOverageTime"))
    except Exception:
        return _fallback(obs_dict)
