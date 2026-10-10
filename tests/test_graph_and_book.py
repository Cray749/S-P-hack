import numpy as np

from tremor.book import build_book, exposure_by_obligor
from tremor.entities import ExposureGraph, get_linker, load_entities


# ---- entity linker -----------------------------------------------------------------------------------
def ids(text: str) -> list[str]:
    return [m.entity_id for m in get_linker().link(text)]


def test_linker_resolves_names_and_cashtags():
    assert ids("Hertz Seeks Coronavirus Bailout") == ["htz"]
    assert ids("$HTZ going to zero, $CAR next") == ["htz", "car"]
    assert ids("Exxon and Chevron cut spending") == ["xom", "cvx"]


def test_linker_avoids_common_false_positives():
    assert ids("Good morning GM! have a nice day") == []           # 'GM' only matches as $GM
    assert ids("Reliance on suppliers is a risk") == []             # bare 'Reliance' is not Reliance Industries
    assert ids("Silicon Valley Bank collapse hits First Republic") == ["svb", "firstrepublic"]


def test_longest_alias_wins():
    assert ids("Avis Budget Group shares fall") == ["car"]


# ---- exposure graph ----------------------------------------------------------------------------------
def test_subsidiaries_are_reached_through_the_parent():
    x = ExposureGraph().propagate(["htz"])
    assert {"hertzcorp", "dollarthrifty", "donlen"} <= set(x)
    assert all(x[e].role == "SELF" and x[e].edge == "CONTROL" for e in ("hertzcorp", "donlen"))
    assert "Hertz Global Holdings" in x["hertzcorp"].path[0]


def test_promoter_group_siblings_get_group_role_without_control_link():
    x = ExposureGraph().propagate(["adanient"])
    assert x["adaniports"].role == "GROUP" and x["adaniports"].edge == "PROMOTER_GROUP"
    assert "tatamotors" not in x or x["tatamotors"].role != "GROUP"


def test_naming_the_group_lights_up_every_member():
    x = ExposureGraph().propagate(["adani_group"])
    assert {"adanient", "adaniports", "adanipower", "adanigreen"} <= {e for e, v in x.items() if v.role == "GROUP"}


def test_peers_are_same_industry_and_region_only():
    x = ExposureGraph().propagate(["htz"])
    peers = {e for e, v in x.items() if v.role == "PEER"}
    assert peers and peers <= {"car", "avisrac"}


def test_watch_only_entity_propagates_to_peers_but_is_not_in_book():
    x = ExposureGraph().propagate(["svb"])
    peers = {e for e, v in x.items() if v.role == "PEER"}
    assert {"wal", "zion", "key"} <= peers
    assert "svb" not in set(build_book().obligors["entity_id"])


# ---- book --------------------------------------------------------------------------------------------
def test_book_is_deterministic():
    a, b = build_book(), build_book()
    assert a.hash == b.hash
    assert a.positions.equals(b.positions)


def test_book_shape_and_instrument_mix():
    b = build_book()
    assert set(b.positions["type"]) == {"LOAN", "BOND", "SWAP"}
    assert 6_000 < b.value < 40_000                       # USD m: a plausible mid-sized wholesale book
    assert b.positions["entity_id"].isin(set(load_entities().by_id)).all()
    assert (b.positions.loc[b.positions.type == "LOAN", "value_usd_m"] > 0).all()


def test_exposure_is_positive_for_every_obligor():
    e = exposure_by_obligor(build_book())
    assert (e > 0).all() and np.isfinite(e).all()
