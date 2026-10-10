"""Entity master data, alias linker and the corporate exposure graph."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from . import config as C

# bare tickers that collide with ordinary words/indices and are therefore only matched as $cashtags
_AMBIGUOUS_BARE = {"CAR", "KEY", "DOW", "ALL", "NOW", "ONE", "GE"}


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    kind: str                 # listed | legal_entity | group
    ticker: str | None
    parent_id: str | None
    group_id: str | None
    sector: str
    country: str
    rating: str | None
    aliases: tuple[str, ...]
    industry: str = ""
    watch_only: bool = False

    @property
    def region(self) -> str:
        return "IN" if self.country == "IN" else "WEST"


class EntityTable:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.by_id: dict[str, Entity] = {}
        for r in df.itertuples():
            ali = tuple(a for a in str(r.aliases).split("|") if a and a != "nan")
            self.by_id[r.id] = Entity(
                r.id, r.name, r.kind,
                r.ticker if isinstance(r.ticker, str) else None,
                r.parent_id if isinstance(r.parent_id, str) else None,
                r.group_id if isinstance(r.group_id, str) else None,
                r.sector, r.country,
                r.rating if isinstance(r.rating, str) else None,
                ali,
                r.industry if isinstance(r.industry, str) else "",
                bool(r.watch_only),
            )

    def ultimate_parent(self, eid: str) -> str:
        seen = set()
        while self.by_id[eid].parent_id and eid not in seen:
            seen.add(eid)
            eid = self.by_id[eid].parent_id  # type: ignore[assignment]
        return eid

    def family(self, eid: str) -> set[str]:
        """Entities consolidated with `eid`: its ultimate parent and every subsidiary beneath it."""
        root = self.ultimate_parent(eid)
        return {e for e in self.by_id if self.ultimate_parent(e) == root}

    def promoter_group(self, eid: str) -> set[str]:
        """Entities that share a promoter/group node with `eid` (no control link required)."""
        gid = self.by_id[eid].group_id
        if not gid:
            return set()
        return {e for e, v in self.by_id.items() if v.group_id == gid and e != gid}


@lru_cache(maxsize=1)
def load_entities() -> EntityTable:
    return EntityTable(pd.read_csv(C.DATA / "entities.csv"))


# ---------------------------------------------------------------------------------------------------
@dataclass
class Mention:
    entity_id: str
    alias: str
    start: int
    cashtag: bool = False


@dataclass
class Linker:
    """Dictionary linker: word-boundary alias match, cashtags, case-sensitive for short/upper aliases."""

    table: EntityTable
    _patterns: list[tuple[re.Pattern[str], str, str, bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        raw: list[tuple[str, str, bool]] = []
        for e in self.table.by_id.values():
            if e.kind == "group":
                for a in e.aliases:
                    raw.append((a, e.id, False))
                continue
            for a in e.aliases:
                raw.append((a, e.id, a.startswith("$")))
            raw.append((e.name, e.id, False))
            if e.ticker:
                base = e.ticker.split(".")[0]
                raw.append((f"${base}", e.id, True))
                if len(base) >= 3 and base not in _AMBIGUOUS_BARE and e.country != "IN":
                    raw.append((base, e.id, False))
        raw.sort(key=lambda t: -len(t[0]))  # longest alias first
        seen: set[tuple[str, str]] = set()
        for alias, eid, cash in raw:
            if (alias, eid) in seen:
                continue
            seen.add((alias, eid))
            esc = re.escape(alias)
            if cash:
                pat = re.compile(rf"(?<![\w$]){esc}(?![\w])", re.IGNORECASE)
            else:
                flags = 0 if (len(alias) <= 5 or alias.isupper()) else re.IGNORECASE
                pat = re.compile(rf"(?<![\w$]){esc}(?![\w])", flags)
            self._patterns.append((pat, eid, alias, cash))

    def link(self, text: str) -> list[Mention]:
        out: list[Mention] = []
        taken: list[tuple[int, int]] = []
        for pat, eid, alias, cash in self._patterns:
            for m in pat.finditer(text):
                s, e = m.span()
                if any(s < te and e > ts for ts, te in taken):
                    continue
                taken.append((s, e))
                out.append(Mention(eid, alias, s, cash))
        out.sort(key=lambda m: m.start)
        return out


@lru_cache(maxsize=1)
def get_linker() -> Linker:
    return Linker(load_entities())


# ---------------------------------------------------------------------------------------------------
@dataclass
class Exposure:
    entity_id: str
    role: str                 # SELF | GROUP | PEER
    path: list[str]           # human-readable propagation path
    edge: str                 # CONTROL | PROMOTER_GROUP | PEER


class ExposureGraph:
    """Propagates a hit on one or more named entities to every entity in the book.

    Edge types (strongest first):
      SELF            the named entity itself
      CONTROL         parent / subsidiary of a named entity (same consolidated family)
      PROMOTER_GROUP  sibling listed company under the same promoter group, no control link
      PEER            same sector and region, no ownership link
    CONTROL is treated as SELF for shock purposes (a subsidiary shares its parent's fate).
    """

    def __init__(self, table: EntityTable | None = None):
        self.t = table or load_entities()

    def propagate(self, named: list[str]) -> dict[str, Exposure]:
        t = self.t
        out: dict[str, Exposure] = {}
        for n in named:
            if n not in t.by_id:
                continue
            ne = t.by_id[n]
            out.setdefault(n, Exposure(n, "SELF", [f"{ne.name} (named in text)"], "SELF"))
            for f in t.family(n):
                if f == n or t.by_id[f].kind == "group":
                    continue
                fe = t.by_id[f]
                if t.ultimate_parent(n) == n:
                    rel = "subsidiary" if fe.parent_id else "parent"
                elif f == t.ultimate_parent(n):
                    rel = "parent"
                else:
                    rel = "sibling subsidiary"
                path = [f"{ne.name} (named)", f"consolidated {rel}", fe.name]
                out.setdefault(f, Exposure(f, "SELF", path, "CONTROL"))
            for g in t.promoter_group(n):
                if g in out or t.by_id[g].kind == "group":
                    continue
                ge = t.by_id[g]
                gname = t.by_id[ne.group_id].name if ne.group_id else "group"
                out.setdefault(g, Exposure(g, "GROUP", [f"{ne.name} (named)", f"promoter group: {gname}", ge.name], "PROMOTER_GROUP"))
        # peers last: lowest priority
        for n in named:
            if n not in t.by_id:
                continue
            ne = t.by_id[n]
            for eid, e in t.by_id.items():
                if eid in out or e.kind == "group" or e.kind == "legal_entity":
                    continue
                if e.watch_only or not ne.industry:
                    continue
                if e.industry == ne.industry and e.region == ne.region:
                    out[eid] = Exposure(eid, "PEER", [f"{ne.name} (named)", f"same industry ({ne.industry}, {ne.region})", e.name], "PEER")
        # legal entities inherit the role of their ultimate parent when not already linked
        for eid, e in t.by_id.items():
            if eid in out or e.kind != "legal_entity":
                continue
            root = t.ultimate_parent(eid)
            if root in out:
                src = out[root]
                out[eid] = Exposure(eid, src.role, src.path[:-1] + [f"{t.by_id[root].name}", f"subsidiary → {e.name}"], src.edge)
        return out
