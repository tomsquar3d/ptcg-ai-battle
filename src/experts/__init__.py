"""Scripted line-experts (docs §3d step 2). `get_expert(name)` → a fresh pilot instance with the
`set_deck`/`choose` agent contract. Register new archetypes here."""
from __future__ import annotations

from .base import ScriptedExpert
from .dragapult_dusknoir import DragapultDusknoirExpert

_REGISTRY = {
    DragapultDusknoirExpert.name: DragapultDusknoirExpert,
}


def get_expert(name: str) -> ScriptedExpert:
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"no scripted expert for {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def available() -> list[str]:
    return sorted(_REGISTRY)
