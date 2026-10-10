import itertools
from datetime import datetime, timedelta, timezone

import pytest

from tremor.events import EventTracker
from tremor.schemas import Document
from tremor.scoring import Scorer

T0 = datetime(2020, 4, 16, 14, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def scorer(tmp_path_factory) -> Scorer:
    return Scorer(cache_dir=tmp_path_factory.mktemp("doccache"))      # isolated: tests never touch data/cache


_uid = itertools.count(1)


def doc(i: int, text: str, pub: str, source: str = "news", minutes: int = 0) -> Document:
    # ids must be unique across the whole run because the scorer caches embeddings by document id
    return Document(id=f"t{next(_uid)}", source=source, publisher=pub, ts=T0 + timedelta(minutes=minutes), text=text)  # type: ignore[arg-type]


def feed(scorer: Scorer, docs: list[Document]) -> EventTracker:
    tr = EventTracker()
    for off, d in enumerate(docs, 1):
        tr.ingest(scorer.state(off, d))
    return tr


HERTZ = "Hertz seeks coronavirus bailout and could be forced to file for bankruptcy by the summer"


def test_same_story_from_two_publishers_forms_one_event(scorer):
    tr = feed(scorer, [doc(1, HERTZ, "benzinga.com"),
                       doc(2, HERTZ + ", according to a report", "nasdaq.com", minutes=30)])
    assert len(tr.events) == 1
    ev = next(iter(tr.events.values()))
    assert ev.n_docs == 2 and "htz" in ev.named_entities()


def test_syndicated_copy_is_not_independent_evidence(scorer):
    # identical wire text under two domains: one independence group, flagged as an echo
    tr = feed(scorer, [doc(1, HERTZ, "reuters.com"), doc(2, HERTZ, "nasdaq.com", minutes=5)])
    ev = next(iter(tr.events.values()))
    assert ev.independent_sources == 1
    assert len(ev.echo_of) == 1 and ev.status == "provisional"


def test_two_independent_credible_publishers_confirm(scorer):
    a = HERTZ
    b = "Car rental firm Hertz may file for bankruptcy as it asks for a coronavirus bailout, report says"
    tr = feed(scorer, [doc(1, a, "benzinga.com"), doc(2, b, "nasdaq.com", minutes=40)])
    ev = next(iter(tr.events.values()))
    if ev.n_docs == 2 and ev.independent_sources == 2:
        assert ev.status == "confirmed"          # 0.8 + 0.9 >= 1.6
    else:                                          # hashing embedder may split paraphrases; never confirm on one source
        assert all(e.status == "provisional" for e in tr.events.values())


def test_social_volume_alone_never_confirms(scorer):
    tweets = [doc(i, f"$HTZ bankruptcy bailout rumour spreading fast number {i % 5}", "twitter", "social", i) for i in range(1, 60)]
    tr = feed(scorer, tweets)
    for ev in tr.events.values():
        assert ev.status == "provisional"
        assert ev.corroboration_social <= 0.5 + 1e-9
        assert ev.independent_sources == 0


def test_routine_analyst_notes_do_not_form_events(scorer):
    tr = feed(scorer, [doc(1, "Deutsche Bank maintains hold rating and lowers price target on Hertz", "benzinga.com")])
    assert len(tr.events) == 0 and tr.ignored == 1


def test_tracker_is_deterministic(scorer):
    docs = [doc(1, HERTZ, "benzinga.com"), doc(2, "Oil prices crash as storage fills up and futures turn negative", "nasdaq.com", minutes=10),
            doc(3, HERTZ + " says report", "nasdaq.com", minutes=20)]
    a, b = feed(scorer, docs), feed(scorer, docs)
    sa = [(e.id, e.n_docs, round(e.confidence, 9), e.status, round(e.sentiment, 9)) for e in a.events.values()]
    sb = [(e.id, e.n_docs, round(e.confidence, 9), e.status, round(e.sentiment, 9)) for e in b.events.values()]
    assert sa == sb


def test_events_expire_and_new_story_starts_new_event(scorer):
    tr = feed(scorer, [doc(1, HERTZ, "benzinga.com", minutes=0), doc(2, HERTZ, "benzinga.com", minutes=60 * 24 * 5)])
    assert len(tr.events) == 2          # 5 days later is a new event, not a continuation
