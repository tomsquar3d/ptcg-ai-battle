"""Shared agent code for the Pokémon TCG AI Battle entry.

Layers (see docs/ptcg_agent_strategy.md §3):
  1. Perception  -> ptcg.perception  (raw Observation -> GameState)   [this milestone]
  2. Belief      -> ptcg.belief       (hidden-zone / threat inference)  [next]
  ...
"""
