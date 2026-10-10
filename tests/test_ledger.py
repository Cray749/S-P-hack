from datetime import datetime, timezone

import pytest

from tremor.ledger import Ledger
from tremor.schemas import Document


def _docs(n: int) -> list[Document]:
    return [
        Document(id=f"d{i}", source="news", publisher="example.com",
                 ts=datetime(2020, 4, 16, 15, 0, tzinfo=timezone.utc), text=f"headline number {i}")
        for i in range(n)
    ]


def test_append_is_idempotent_and_ordered():
    led = Ledger()
    docs = _docs(5)
    assert led.append_many(docs) == 5
    assert led.append_many(docs) == 0          # replaying the same feed adds nothing
    assert len(led) == 5
    assert [e.offset for e in led.read()] == [1, 2, 3, 4, 5]


def test_hash_chain_verifies():
    led = Ledger()
    led.append_many(_docs(20))
    assert led.verify()
    assert led.verify(upto=7)


def test_ledger_rejects_update_and_delete():
    led = Ledger()
    led.append_many(_docs(3))
    with pytest.raises(Exception, match="append-only"):
        led.db.execute("UPDATE entries SET payload='tampered' WHERE offset=1")
    with pytest.raises(Exception, match="append-only"):
        led.db.execute("DELETE FROM entries WHERE offset=1")
    assert led.verify()


def test_tampering_is_detected_even_if_triggers_are_bypassed():
    led = Ledger()
    led.append_many(_docs(4))
    led.db.execute("DROP TRIGGER entries_no_update")
    led.db.execute("UPDATE entries SET payload=replace(payload,'number 2','number 99') WHERE offset=3")
    assert not led.verify()
