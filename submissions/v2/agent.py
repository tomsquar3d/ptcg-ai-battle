"""v2 agent brain: evaluator-guided 2-ply search (our turn + opponent reply) via the native
Search API.

Same skeleton as v1 (determinize once with search_begin, branch one search_step per legal
root option, greedily roll our turn to its end, score with the heuristic evaluator, argmax),
with one addition: after our turn ends we roll the **opponent's best reply** to its end and
score that post-reply state. v1 was turn-aware but reply-blind — it could pick a turn that
scores well and then walk into a KO it never searched. Rolling the opponent's turn makes this
a true 2-ply (our turn -> their turn -> eval) that sees the reply. See
docs/ptcg_agent_strategy.md §3 (layer 5 lookahead) and experiments/log.md.

`_greedy_rollout` is perspective-parameterized, so the opponent's turn is just the same rollout
called with the opponent's index; `_score_state` already scores from our view and negates when
the reached state is the opponent's to move.

Determinization note (unchanged from v1): the opponent's hidden zones are filled with our own
decklist as valid filler — no real archetype template yet (that comes from replay mining, §5).
So the simulated reply is played from a wrong hand distribution: it catches gross blunders
(obvious KOs, easy gusts) but its full value is gated on a real opponent model later. Toggle
`opponent_reply=False` (or OPPONENT_REPLY) to recover exact v1 behaviour for ablation.

Everything is wrapped so it degrades to a legal move on any error or clock pressure — main.py
adds the outer watchdog.
"""
import os
import random
import sys
import time

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from cg.api import (  # noqa: E402
    to_observation_class, all_card_data, all_attack,
    search_begin, search_step, search_end,
)
from ptcg.perception import build_game_state  # noqa: E402
from ptcg.evaluator import evaluate, WIN_SCORE  # noqa: E402

_CARD_INDEX = None
_ATTACK_INDEX = None
_DECK = None

MOVE_BUDGET_S = 2.0      # soft wall-clock cap per decision
LOW_CLOCK_S = 15.0       # below this remaining bank, skip search and move instantly
ROOT_BEAM = 6            # how many root options to fully roll out to end-of-turn
DEPTH_CAP = 24           # max our-turn actions explored in a rollout
OPP_DEPTH_CAP = 24       # max opponent-turn actions explored in the reply rollout
NODE_BUDGET = 7000       # max search_step calls per decision (anytime cap; ~<1s)
OPPONENT_REPLY = True    # default: simulate the opponent's reply (2-ply). False == v1 behaviour.

# Lightweight instrumentation (read by scripts/v2_eval.py). opp_rollouts must be > 0 when
# reply is on, else the opponent-turn rollout is inert (e.g. yourIndex not flipping mid-search).
STATS = {"our_rollouts": 0, "opp_rollouts": 0}


def _lazy_init():
    global _CARD_INDEX, _ATTACK_INDEX
    if _CARD_INDEX is None:
        _CARD_INDEX = {c.cardId: c for c in all_card_data()}
        _ATTACK_INDEX = {a.attackId: a for a in all_attack()}


def set_deck(deck):
    global _DECK
    _DECK = list(deck)


def _first_basic(deck):
    for cid in deck:
        cd = _CARD_INDEX.get(cid)
        if cd and getattr(cd, "basic", False) and getattr(cd, "cardType", None) == 0:
            return cid
    return deck[0]


def _determinize(obs, deck):
    """Args for search_begin: our decklist fills every hidden zone (valid ids, has Basics)."""
    st = obs.current
    opp = 1 - st.yourIndex
    opp_active = []
    ops = st.players[opp]
    if ops.active and ops.active[0] is None:  # opponent's active is face-down → must predict it
        opp_active = [_first_basic(deck)]
    d = list(deck)
    return (d, d, d, d, d, opp_active, False)


def _legal_random(sel):
    n = len(sel.option)
    if n == 0:
        return []
    return random.sample(range(n), min(sel.maxCount, n))


def _score_state(search_state, my_index):
    obs = search_state.observation
    cur = getattr(obs, "current", None)
    if cur is None:
        return 0.0
    if getattr(cur, "result", -1) != -1:  # terminal
        if cur.result == my_index:
            return WIN_SCORE
        return -WIN_SCORE if cur.result in (0, 1) else 0.0
    gs = build_game_state(obs, _CARD_INDEX)
    s = evaluate(gs, card_index=_CARD_INDEX, attack_index=_ATTACK_INDEX).score
    # resulting state may be the opponent's turn → score is from their view; flip to ours.
    return s if getattr(cur, "yourIndex", my_index) == my_index else -s


def _is_our_turn(state, my_index):
    obs = getattr(state, "observation", None)
    cur = getattr(obs, "current", None)
    return (cur is not None and getattr(cur, "result", -1) == -1
            and getattr(obs, "select", None) is not None
            and getattr(cur, "yourIndex", None) == my_index)


def _greedy_rollout(state, my_index, ctx, depth_cap=DEPTH_CAP):
    """From a `my_index`-to-move state, greedily play the rest of that player's turn (each step =
    the option that maximizes the resulting state's score from `my_index`'s view), until the
    turn ends / game ends / budget hit. Returns the reached end-of-turn (or cut) state; the
    caller scores it as the line's value. Perspective-parameterized: pass the opponent's index
    to roll the opponent's turn."""
    depth = 0
    while _is_our_turn(state, my_index) and depth < depth_cap:
        if ctx["n"] >= NODE_BUDGET or time.monotonic() > ctx["deadline"]:
            break
        sel = state.observation.select
        nopt = len(sel.option)
        if nopt == 0:
            break
        if sel.maxCount == 1 and sel.minCount <= 1:
            best = None
            for j in range(nopt):
                if ctx["n"] >= NODE_BUDGET or time.monotonic() > ctx["deadline"]:
                    break
                try:
                    cj = search_step(state.searchId, [j])
                    ctx["n"] += 1
                except Exception:
                    continue
                s = _score_state(cj, my_index)
                if best is None or s > best[0]:
                    best = (s, cj)
            if sel.minCount == 0:  # option of taking no more actions
                try:
                    c0 = search_step(state.searchId, [])
                    ctx["n"] += 1
                    if best is None or _score_state(c0, my_index) >= best[0]:
                        best = (0, c0)
                except Exception:
                    pass
            if best is None:
                break
            state = best[1]
        else:  # multi-select mid-turn: take a valid default set and continue
            k = max(sel.minCount, min(sel.maxCount, nopt))
            try:
                state = search_step(state.searchId, list(range(k)))
                ctx["n"] += 1
            except Exception:
                break
        depth += 1
    return state


def _rollout_line(child_state, my_index, ctx, opponent_reply):
    """Value a root child: greedily roll our turn to its end, then (if opponent_reply) roll the
    opponent's best reply to its end. Returns the reached state to be scored from our view.

    2-ply = our turn -> their turn -> leaf eval. If our turn already ended the game, or it is
    somehow not our turn to move, we skip straight to scoring. The opponent rollout reuses the
    same greedy rollout with the opponent's index (they play to maximize their own position)."""
    end = child_state
    if _is_our_turn(child_state, my_index):
        STATS["our_rollouts"] += 1
        end = _greedy_rollout(child_state, my_index, ctx, DEPTH_CAP)
    if opponent_reply:
        opp = 1 - my_index
        if _is_our_turn(end, opp):  # their turn to act, non-terminal → simulate the reply
            STATS["opp_rollouts"] += 1
            end = _greedy_rollout(end, opp, ctx, OPP_DEPTH_CAP)
    return end


def choose(obs_dict, time_left=None, opponent_reply=None):
    """2-ply search: for the top root options, roll our turn to its end, then the opponent's
    reply, and pick the option whose resulting state scores highest for us. Falls back to a
    legal move on any trouble. `opponent_reply` overrides the OPPONENT_REPLY default (used by
    the ablation harness to compare reply-on vs reply-off in one process)."""
    if opponent_reply is None:
        opponent_reply = OPPONENT_REPLY
    obs = to_observation_class(obs_dict)
    sel = obs.select
    if sel is None:
        return list(_DECK) if _DECK else []
    n = len(sel.option)
    if n == 0:
        return []
    my_index = obs.current.yourIndex if obs.current else 0

    if time_left is not None and time_left < LOW_CLOCK_S:
        return _legal_random(sel)

    try:
        _lazy_init()
        root = search_begin(obs, *_determinize(obs, _DECK))
        ctx = {"deadline": time.monotonic() + MOVE_BUDGET_S, "n": 0}
        try:
            # Rank root options by immediate score, keep their child states.
            imm = []
            for i in range(n):
                if ctx["n"] >= NODE_BUDGET or time.monotonic() > ctx["deadline"]:
                    break
                try:
                    ci = search_step(root.searchId, [i])
                    ctx["n"] += 1
                    imm.append((_score_state(ci, my_index), i, ci))
                except Exception:
                    continue
            if not imm:
                return _legal_random(sel)

            # Multi-select root: greedy top-maxCount by immediate score (no turn rollout).
            if sel.maxCount > 1:
                if sel.minCount <= 1:
                    imm.sort(key=lambda t: t[0], reverse=True)
                    pick = sorted({i for _, i, _ in imm[:sel.maxCount]})
                    if len(pick) >= sel.minCount:
                        return pick[:sel.maxCount]
                return _legal_random(sel)

            # Single-select: roll out the top-BEAM options through our turn + opponent reply.
            imm.sort(key=lambda t: t[0], reverse=True)
            best_i, best_v = None, float("-inf")
            for _s, i, ci in imm[:ROOT_BEAM]:
                if best_i is not None and time.monotonic() > ctx["deadline"]:
                    break
                end = _rollout_line(ci, my_index, ctx, opponent_reply)
                v = _score_state(end, my_index)
                if v > best_v:
                    best_v, best_i = v, i

            # Selecting nothing (minCount 0): compare its turn value too.
            if sel.minCount == 0:
                try:
                    c0 = search_step(root.searchId, [])
                    ctx["n"] += 1
                    end0 = _rollout_line(c0, my_index, ctx, opponent_reply)
                    if _score_state(end0, my_index) > best_v:
                        return []
                except Exception:
                    pass

            return [best_i] if best_i is not None else _legal_random(sel)
        finally:
            search_end()
    except Exception:
        return _legal_random(sel)
