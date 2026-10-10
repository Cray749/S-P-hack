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


def _trie_regex(words: list[str]) -> str:
    """Compile literals into a prefix-sharing regex (greedy, so the longest alias wins)."""
    trie: dict = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[""] = True

    def build(node: dict) -> str:
        terminal = "" in node
        kids = sorted((k, v) for k, v in node.items() if k != "")
        if not kids:
            return ""
        alts = [re.escape(k) + build(v) for k, v in kids]
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return body if not terminal else f"(?:{body})?"

    return build(trie)


@dataclass
class Linker:
    """Dictionary linker: boundary-aware alias match, cashtags, case-sensitive for short/upper aliases.

    One trie-compiled regex per case mode keeps linking O(len(text)) regardless of alias count.
    """

    table: EntityTable
    _ci: dict[str, str] = field(default_factory=dict)
    _cs: dict[str, str] = field(default_factory=dict)
    _ci_re: re.Pattern[str] | None = None
    _cs_re: re.Pattern[str] | None = None

    def __post_init__(self) -> None:
        def add(alias: str, eid: str) -> None:
            cash = alias.startswith("$")
            ci = cash or not (len(alias) <= 5 or alias.isupper())
            target = self._ci if ci else self._cs
            key = alias.lower() if ci else alias
            target.setdefault(key, eid)

        # listed parents first so a shared alias resolves to the parent, not a subsidiary
        order = sorted(self.table.by_id.values(), key=lambda e: (e.kind != "listed", e.kind != "group"))
        for e in order:
            for a in e.aliases:
                add(a, e.id)
            if e.kind != "group":
                add(e.name, e.id)
            if e.ticker:
                base = e.ticker.split(".")[0]
                add(f"${base}", e.id)
                if len(base) >= 3 and base not in _AMBIGUOUS_BARE and e.country != "IN":
                    add(base, e.id)
        wrap = lambda keys: re.compile(r"(?<![\w$])" + _trie_regex(sorted(keys)) + r"(?![\w])", 0)  # noqa: E731
        self._cs_re = wrap(list(self._cs))
        self._ci_re = re.compile(r"(?<![\w$])" + _trie_regex(sorted(self._ci)) + r"(?![\w])", re.IGNORECASE)

    def link(self, text: str) -> list[Mention]:
        found: list[Mention] = []
        for rx, table, ci in ((self._cs_re, self._cs, False), (self._ci_re, self._ci, True)):
            assert rx is not None
            for m in rx.finditer(text):
                if m.end() == m.start():
                    continue
                surface = m.group(0)
                eid = table.get(surface.lower() if ci else surface)
                if eid:
                    found.append(Mention(eid, surface, m.start(), surface.startswith("$")))
        found.sort(key=lambda m: (m.start, -len(m.alias)))
        out: list[Mention] = []
        end = -1
        for m in found:  # drop overlaps, keep the earliest/longest
            if m.start >= end:
                out.append(m)
                end = m.start + len(m.alias)
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
        for n in named:                       # a promoter group named as a whole lights up every member
            if n in t.by_id and t.by_id[n].kind == "group":
                gn = t.by_id[n]
                for m, me in t.by_id.items():
                    if me.group_id == n and me.kind != "group":
                        out.setdefault(m, Exposure(m, "GROUP", [f"{gn.name} (named)", "group member", me.name], "PROMOTER_GROUP"))
        named = [n for n in named if n in t.by_id and t.by_id[n].kind != "group"]
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
