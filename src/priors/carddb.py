"""Card database + TrainerHill→competition-pool mapping (offline meta-priors pipeline).

Joins the human-meta skeleton CSVs (TrainerHill) onto the competition card pool. Two card DBs:
  * `data/cards/EN_Card_Data.csv` — the English card DB. "Card ID" == the engine's `cardId`
    (verified: EN spans exactly the 1267-card engine pool). One row per attack/ability, so we
    dedupe to one record per Card ID and collect the effect text.
  * `cg.api.all_card_data()` — the engine DB (the *pool* itself), source of truth for
    `cardType` and abilities (`skills`). A card is "in pool" iff its id is in this set.

Mapping a skeleton row `(card_name, set_code)` to a pool card id, in priority order:
  1. **set_code exact join.** "TWM-130" → (Expansion="TWM", Collection No.=130) → Card ID.
     TrainerHill and the EN DB mostly agree on set codes, but not always (different set
     abbreviations for some sets), so this is only ~69% of rows.
  2. **basic-energy alias.** "Psychic Energy" → EN "Basic {P} Energy" (SVE). The EN DB names
     basic energy by symbol, not colour, so these never match by set code or raw name.
  3. **normalized-name fallback.** Lowercase, drop apostrophes (EN uses a curly ’ that ascii-
     folds away while TrainerHill uses a straight '; deleting both unifies "Boss's Orders"),
     collapse non-alphanumerics to spaces. Only accepted if the matched id is in the pool.

Residual unmapped cards are genuinely out-of-pool (e.g. the "CRI" set, incl. Special Red Card
and Mega Greninja ex) — a real constructibility gap, not a mapping bug.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EN_CSV = os.path.join(REPO, "data", "cards", "EN_Card_Data.csv")

# The engine SDK (`cg`) ships inside each submission; vendor it onto the path for offline use.
sys.path.insert(0, os.path.join(REPO, "submissions", "v1"))
from cg.api import all_card_data, CardType  # noqa: E402

# Colour name (TrainerHill) → energy symbol (EN "Basic {SYM} Energy").
_ENERGY_SYMBOL = {
    "grass": "{G}", "fire": "{R}", "water": "{W}", "lightning": "{L}",
    "psychic": "{P}", "fighting": "{F}", "darkness": "{D}", "metal": "{M}",
}


def norm(s: str) -> str:
    """Normalize a card name for fuzzy matching: fold accents, delete apostrophes (so a curly
    ’ and a straight ' both vanish rather than one becoming a space), lowercase, and collapse
    every remaining non-alphanumeric run to a single space."""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("’", "").replace("'", "").replace("`", "")
    s = s.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@dataclass
class PoolCard:
    """One competition-pool card, merged from the engine DB (authoritative types/abilities)
    and the EN CSV (name, set, trainer effect text)."""
    card_id: int
    name: str
    card_type: str            # CardType enum name: POKEMON/ITEM/TOOL/SUPPORTER/STADIUM/*ENERGY
    expansion: str = ""
    collection_no: str = ""
    effect_text: str = ""     # concatenated EN effect explanations (for supporter classing)
    has_ability: bool = False  # POKEMON with a non-empty engine `skills` list

    @property
    def is_basic_energy(self) -> bool:
        return self.card_type == "BASIC_ENERGY"

    @property
    def is_special_energy(self) -> bool:
        return self.card_type == "SPECIAL_ENERGY"

    @property
    def is_tool(self) -> bool:
        return self.card_type == "TOOL"

    @property
    def is_pokemon(self) -> bool:
        return self.card_type == "POKEMON"

    @property
    def is_draw_supporter(self) -> bool:
        """A Supporter whose text draws cards — the class Judge / Iono / Red Card punish."""
        return self.card_type == "SUPPORTER" and bool(re.search(r"\bdraw", self.effect_text.lower()))


@dataclass
class MapResult:
    card_id: Optional[int]
    method: str               # "set_code" | "energy_alias" | "name" | "unmapped"

    @property
    def mapped(self) -> bool:
        return self.card_id is not None


@dataclass
class CardDB:
    pool: dict[int, PoolCard]
    _by_setcode: dict[tuple[str, str], int] = field(default_factory=dict)
    _by_name: dict[str, int] = field(default_factory=dict)

    def resolve(self, card_name: str, set_code: str) -> MapResult:
        # 1) set_code exact join
        exp, _, num = set_code.partition("-")
        num = str(int(num)) if num.strip().isdigit() else num.strip()
        cid = self._by_setcode.get((exp.strip(), num))
        if cid is not None and cid in self.pool:
            return MapResult(cid, "set_code")
        # 2) basic-energy alias
        low = card_name.strip().lower()
        for colour, sym in _ENERGY_SYMBOL.items():
            if low == f"{colour} energy":
                cid = self._by_name.get(norm(f"Basic {sym} Energy"))
                if cid is not None and cid in self.pool:
                    return MapResult(cid, "energy_alias")
        # 3) normalized-name fallback (must land in pool)
        cid = self._by_name.get(norm(card_name))
        if cid is not None and cid in self.pool:
            return MapResult(cid, "name")
        return MapResult(None, "unmapped")


def load_card_db() -> CardDB:
    """Load the engine pool + EN CSV and build the join indexes."""
    engine = {c.cardId: c for c in all_card_data()}

    # EN rows: dedupe to one record per Card ID; concatenate effect text across its rows.
    en_name: dict[int, str] = {}
    en_exp: dict[int, str] = {}
    en_num: dict[int, str] = {}
    en_effect: dict[int, list[str]] = {}
    by_setcode: dict[tuple[str, str], int] = {}
    by_name: dict[str, int] = {}
    with open(EN_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = int(r["Card ID"])
            cn = r["Collection No."].strip()
            num = str(int(cn)) if cn.isdigit() else cn
            en_name.setdefault(cid, r["Card Name"])
            en_exp.setdefault(cid, r["Expansion"].strip())
            en_num.setdefault(cid, num)
            by_setcode.setdefault((r["Expansion"].strip(), num), cid)
            by_name.setdefault(norm(r["Card Name"]), cid)
            eff = (r.get("Effect Explanation") or "").strip()
            if eff and eff.lower() != "n/a":
                en_effect.setdefault(cid, []).append(eff)

    pool: dict[int, PoolCard] = {}
    for cid, ec in engine.items():
        pool[cid] = PoolCard(
            card_id=cid,
            name=ec.name,
            card_type=CardType(ec.cardType).name,
            expansion=en_exp.get(cid, ""),
            collection_no=en_num.get(cid, ""),
            effect_text=" ".join(en_effect.get(cid, [])),
            has_ability=(CardType(ec.cardType) == CardType.POKEMON and bool(ec.skills)),
        )
    return CardDB(pool=pool, _by_setcode=by_setcode, _by_name=by_name)
