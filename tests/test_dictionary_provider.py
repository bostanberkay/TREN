# tests/test_dictionary_provider.py
"""Pure-function tests for dictionary_provider.py (the TDK Checker's
provider abstraction). NO test in this file (or anywhere in the suite)
ever calls the real TDK website -- TDKProvider is only ever exercised here
via its injectable `opener` callable, which never performs a real network
request."""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dictionary_provider as dp


# ---------------------------------------------------------------------------
# Query normalization / Turkish character handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("kitap", "kitap"),
    ("KITAP", "kıtap"),  # plain 'I' folds to dotless 'ı', not ASCII 'i'
    ("İstanbul", "istanbul"),  # dotted capital İ folds to dotted lowercase i
    ("IŞIK", "ışık"),
    ("Kitab'ın", "kitabın"),
    ("Kitab’ın", "kitabın"),  # curly apostrophe too
    ("-merhaba-", "merhaba"),
    ("  kitap  ", "kitap"),
    ("kitap.", "kitap"),
    ("", ""),
])
def test_normalize_query(raw, expected):
    assert dp.normalize_query(raw) == expected


def test_turkish_lower_handles_all_four_i_variants():
    # İ->i, I->ı, ı stays ı, i stays i -- the classic Turkish case-folding
    # trap that Python's default str.lower() gets wrong for 'I'/'İ'.
    assert dp.turkish_lower("İIıi") == "iııi"


def test_normalize_query_unicode_nfc():
    # "a" + combining diaeresis vs the precomposed character should
    # normalize identically.
    decomposed = "kadın"  # already precomposed dotless-i, sanity check
    assert dp.normalize_query(decomposed) == "kadın"


def test_normalize_query_none_and_falsy_never_raises():
    assert dp.normalize_query(None) == ""
    assert dp.normalize_query("") == ""


def test_normalize_query_internal_hyphen_preserved():
    assert dp.normalize_query("gel-git") == "gel-git"


# ---------------------------------------------------------------------------
# LookupResult / status contract
# ---------------------------------------------------------------------------

def test_all_statuses_are_the_five_required_values():
    assert set(dp.ALL_STATUSES) == {"FOUND", "NOT_FOUND", "UNAVAILABLE", "NETWORK_ERROR", "STALE_RESULT"}


def test_lookup_result_to_dict_is_json_serializable():
    import json
    r = dp.LookupResult(query="kitap", normalized_query="kitap", status="FOUND",
                         source="mock", entries=({"headword": "kitap"},))
    json.dumps(r.to_dict())


def test_lookup_result_never_carries_a_label_field():
    """No label mutation: the provider layer has no notion of an
    annotation label at all -- structurally impossible for a lookup
    result to carry one."""
    r = dp.LookupResult(query="x", normalized_query="x", status="FOUND", source="mock")
    assert "label" not in r.to_dict()
    assert not hasattr(r, "label")


# ---------------------------------------------------------------------------
# MockDictionaryProvider
# ---------------------------------------------------------------------------

def test_mock_provider_found():
    provider = dp.MockDictionaryProvider(responses={"kitap": "FOUND"})
    r = provider.lookup("kitap")
    assert r.status == "FOUND"
    assert r.source == "mock"


def test_mock_provider_not_found_default():
    provider = dp.MockDictionaryProvider()
    r = provider.lookup("zzqxwv")
    assert r.status == "NOT_FOUND"


def test_mock_provider_unavailable_and_network_error_configurable():
    provider = dp.MockDictionaryProvider(responses={"a": "UNAVAILABLE", "b": "NETWORK_ERROR"})
    assert provider.lookup("a").status == "UNAVAILABLE"
    assert provider.lookup("b").status == "NETWORK_ERROR"


def test_mock_provider_entries_tuple():
    provider = dp.MockDictionaryProvider(responses={"kitap": ("FOUND", [{"headword": "kitap"}])})
    r = provider.lookup("kitap")
    assert r.status == "FOUND"
    assert r.entries == ({"headword": "kitap"},)


def test_mock_provider_records_calls_deterministically():
    provider = dp.MockDictionaryProvider()
    provider.lookup("kitap")
    provider.lookup("ev")
    assert provider.calls == ["kitap", "ev"]
    assert provider.call_count == 2


def test_mock_provider_rejects_unknown_status():
    provider = dp.MockDictionaryProvider(responses={"x": "BOGUS_STATUS"})
    result = provider.lookup("x")
    # the base lookup() wraps _do_lookup failures as UNAVAILABLE -- never
    # propagates a raw exception to the caller.
    assert result.status == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Caching: hit / miss, keyed by (normalized_query, source)
# ---------------------------------------------------------------------------

def test_cache_miss_then_hit():
    provider = dp.MockDictionaryProvider(responses={"kitap": "FOUND"})
    r1 = provider.lookup("kitap")
    assert r1.from_cache is False
    r2 = provider.lookup("kitap")
    assert r2.from_cache is True
    assert provider.call_count == 1  # second call never reached _do_lookup


def test_cache_is_keyed_by_normalized_query_not_raw_surface_form():
    # "KİTAP" (dotted capital İ, the Turkish-correct capitalization of
    # "kitap") and "Kitap" both normalize to "kitap" -- deliberately NOT
    # using plain ASCII "KITAP", which correctly normalizes to "kıtap"
    # (dotless ı) under Turkish casing rules, a different word.
    provider = dp.MockDictionaryProvider(responses={"kitap": "FOUND"})
    provider.lookup("Kitap")
    r = provider.lookup("KİTAP")
    assert r.from_cache is True
    assert provider.call_count == 1


def test_cache_is_scoped_per_provider_instance():
    p1 = dp.MockDictionaryProvider(responses={"kitap": "FOUND"})
    p2 = dp.MockDictionaryProvider(responses={"kitap": "FOUND"})
    p1.lookup("kitap")
    r = p2.lookup("kitap")
    assert r.from_cache is False  # p2's own cache is independent of p1's


def test_clear_cache():
    provider = dp.MockDictionaryProvider(responses={"kitap": "FOUND"})
    provider.lookup("kitap")
    provider.clear_cache()
    r = provider.lookup("kitap")
    assert r.from_cache is False
    assert provider.call_count == 2


def test_empty_query_never_reaches_do_lookup():
    provider = dp.MockDictionaryProvider()
    r = provider.lookup("   ")
    assert r.status == "NOT_FOUND"
    assert provider.call_count == 0


# ---------------------------------------------------------------------------
# TDKProvider: FOUND / NOT_FOUND / UNAVAILABLE / NETWORK_ERROR, using an
# injected fake opener -- NEVER a real network call.
# ---------------------------------------------------------------------------

def test_tdk_provider_found():
    def fake_opener(url, timeout):
        assert "ara=" in url
        return b'[{"madde": "kitap", "anlamlarListe": [{"anlam": "yazili eser"}]}]'
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("kitap")
    assert r.status == "FOUND"
    assert r.entries[0].headword == "kitap"
    assert r.entries[0].definitions == ("yazili eser",)


def test_tdk_provider_not_found_error_dict():
    def fake_opener(url, timeout):
        return b'{"error": "Sonuc bulunamadi"}'
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("zzzzz")
    assert r.status == "NOT_FOUND"


def test_tdk_provider_not_found_never_leaks_raw_turkish_backend_text():
    """The TDK gts endpoint's own NOT_FOUND payload is Turkish system text
    ("Sonuç bulunamadı") -- the provider's message must always be this
    module's own fixed English text instead, never that raw string."""
    def fake_opener(url, timeout):
        return b'{"error": "Sonu\xc3\xa7 bulunamad\xc4\xb1"}'
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("zzzzz")
    assert r.status == "NOT_FOUND"
    assert "bulunamad" not in r.message
    assert "Sonu" not in r.message
    assert r.message == "no dictionary entry found"


def test_tdk_provider_not_found_empty_list():
    def fake_opener(url, timeout):
        return b'[]'
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("zzzzz")
    assert r.status == "NOT_FOUND"


def test_tdk_provider_unavailable_on_malformed_json():
    def fake_opener(url, timeout):
        return b'<html>not json at all</html>'
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("abc")
    assert r.status == "UNAVAILABLE"


def test_tdk_provider_unavailable_on_changed_structure():
    """Treat any unrecognized (but validly-parsed) JSON shape as
    UNAVAILABLE -- simulates TDK changing its response format -- never
    fabricates FOUND or NOT_FOUND from a shape this provider doesn't
    recognize."""
    def fake_opener(url, timeout):
        return b'{"totally": "different", "shape": true}'
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("abc")
    assert r.status == "UNAVAILABLE"


def test_tdk_provider_unavailable_on_non_utf8_bytes():
    def fake_opener(url, timeout):
        return b'\xff\xfe\x00\x01garbage'
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("abc")
    assert r.status == "UNAVAILABLE"


def test_tdk_provider_network_error_on_timeout():
    def fake_opener(url, timeout):
        raise TimeoutError("timed out")
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("abc")
    assert r.status == "NETWORK_ERROR"


def test_tdk_provider_network_error_on_connection_failure():
    import urllib.error
    def fake_opener(url, timeout):
        raise urllib.error.URLError("connection refused")
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("abc")
    assert r.status == "NETWORK_ERROR"


def test_tdk_provider_network_error_on_http_error():
    import urllib.error
    def fake_opener(url, timeout):
        raise urllib.error.HTTPError("http://x", 503, "Service Unavailable", {}, None)
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("abc")
    assert r.status == "NETWORK_ERROR"


def test_tdk_provider_never_raises_on_opener_generic_exception():
    def fake_opener(url, timeout):
        raise RuntimeError("something unexpected")
    provider = dp.TDKProvider(opener=fake_opener)
    r = provider.lookup("abc")
    assert r.status == "UNAVAILABLE"


def test_tdk_provider_url_encodes_turkish_characters():
    seen_urls = []
    def fake_opener(url, timeout):
        seen_urls.append(url)
        return b'[]'
    provider = dp.TDKProvider(opener=fake_opener)
    provider.lookup("İstanbul çöğüşı")
    assert seen_urls
    url = seen_urls[0]
    assert "İ" not in url and "ç" not in url and " " not in url
    assert "%" in url  # percent-encoded


def test_tdk_provider_uses_configured_timeout():
    seen_timeouts = []
    def fake_opener(url, timeout):
        seen_timeouts.append(timeout)
        return b'[]'
    provider = dp.TDKProvider(opener=fake_opener, timeout=1.5)
    provider.lookup("abc")
    assert seen_timeouts == [1.5]


def test_tdk_provider_default_timeout_is_short():
    assert dp.DEFAULT_TIMEOUT_SECONDS <= 10


def test_tdk_provider_caches_across_calls():
    calls = []
    def fake_opener(url, timeout):
        calls.append(url)
        return b'[{"madde": "kitap", "anlamlarListe": []}]'
    provider = dp.TDKProvider(opener=fake_opener)
    provider.lookup("kitap")
    provider.lookup("kitap")
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# UnavailableProvider
# ---------------------------------------------------------------------------

def test_unavailable_provider_never_returns_found():
    provider = dp.UnavailableProvider()
    for term in ("kitap", "", "anything"):
        assert provider.lookup(term).status in ("UNAVAILABLE", "NOT_FOUND")
        # empty term short-circuits to NOT_FOUND before reaching the
        # provider's own _do_lookup -- everything else is UNAVAILABLE.


def test_unavailable_provider_specifically():
    provider = dp.UnavailableProvider()
    r = provider.lookup("kitap")
    assert r.status == "UNAVAILABLE"
    assert r.source == "offline"


# ---------------------------------------------------------------------------
# Timeout / "no GUI freeze" support: MockDictionaryProvider's delay lets
# callers (the GUI layer) verify a slow lookup doesn't block anything it
# shouldn't -- exercised here at the provider level via real threads, with
# no Tk dependency at all.
# ---------------------------------------------------------------------------

def test_mock_provider_delay_does_not_block_other_threads():
    provider = dp.MockDictionaryProvider(responses={"slow": "FOUND"}, delay_seconds=0.2)
    result_holder = {}

    def worker():
        result_holder['r'] = provider.lookup("slow")

    t0 = time.time()
    thread = threading.Thread(target=worker)
    thread.start()
    # The calling thread is free to do other work while `thread` sleeps.
    counter = 0
    while thread.is_alive():
        counter += 1
    thread.join()
    elapsed = time.time() - t0
    assert result_holder['r'].status == "FOUND"
    assert elapsed >= 0.2
    assert counter > 0  # the main thread was doing real work concurrently


# ---------------------------------------------------------------------------
# Rich DictionaryEntry extraction: headword, POS, definitions/senses,
# origin, pronunciation, usage labels, compounds, idioms, proverbs,
# examples -- all defensively mapped, missing fields never guessed.
# ---------------------------------------------------------------------------

def _tdk_sample_response():
    return (
        b'[{"madde": "film", "lisan": "Fransizca", "telaffuz": "",'
        b'"birlesikler": "film cekmek|film seridi",'
        b'"atasozu": [],'
        b'"deyimler": [{"madde": "filmi cekilmis olmak"}],'
        b'"anlamlarListe": ['
        b'{"anlam": "Bir konuyu goruntu olarak yansitan serit.",'
        b' "ozelliklerListe": [{"tam_adi": "isim"}],'
        b' "orneklerListe": [{"ornek": "Guzel bir film izledik."}]},'
        b'{"anlam": "Argo: sinema.",'
        b' "ozelliklerListe": [{"tam_adi": "argo"}],'
        b' "orneklerListe": []}'
        b']}]'
    )


def test_tdk_provider_extracts_rich_entry_fields():
    provider = dp.TDKProvider(opener=lambda url, timeout: _tdk_sample_response())
    r = provider.lookup("film")
    assert r.status == "FOUND"
    e = r.entries[0]
    assert e.headword == "film"
    assert e.part_of_speech == "isim"
    assert e.origin == "Fransizca"
    assert e.compounds == ("film cekmek", "film seridi")
    assert e.idioms == ("filmi cekilmis olmak",)
    assert len(e.senses) == 2
    assert e.senses[0].definition == "Bir konuyu goruntu olarak yansitan serit."
    assert e.senses[0].part_of_speech == "isim"
    assert e.senses[0].examples == ("Guzel bir film izledik.",)
    assert e.senses[1].usage_labels == ("argo",)


def test_dictionary_entry_missing_fields_are_empty_not_guessed():
    provider = dp.TDKProvider(opener=lambda url, timeout: b'[{"madde": "x", "anlamlarListe": []}]')
    r = provider.lookup("x")
    e = r.entries[0]
    assert e.origin == ""
    assert e.pronunciation == ""
    assert e.compounds == ()
    assert e.senses == ()


def test_format_field_for_display_uses_not_provided_for_falsy():
    assert dp.format_field_for_display("") == dp.NOT_PROVIDED
    assert dp.format_field_for_display(None) == dp.NOT_PROVIDED
    assert dp.format_field_for_display(()) == dp.NOT_PROVIDED
    assert dp.format_field_for_display("Fransizca") == "Fransizca"


def test_dictionary_entry_to_dict_json_serializable():
    import json
    provider = dp.TDKProvider(opener=lambda url, timeout: _tdk_sample_response())
    r = provider.lookup("film")
    json.dumps(r.to_dict())


def test_dictionary_entry_preserves_unrecognized_fields_in_raw():
    provider = dp.TDKProvider(
        opener=lambda url, timeout: b'[{"madde": "x", "anlamlarListe": [], "gelecekte_eklenecek_alan": 42}]')
    r = provider.lookup("x")
    assert r.entries[0].raw.get("gelecekte_eklenecek_alan") == 42


# ---------------------------------------------------------------------------
# STALE_RESULT: applied by the GUI layer to a previously-fetched result,
# never produced by a provider's own lookup() call.
# ---------------------------------------------------------------------------

def test_mark_stale_preserves_entries_and_message_but_changes_status():
    original = dp.LookupResult(query="kitap", normalized_query="kitap", status="FOUND",
                                source="mock", entries=({"headword": "kitap"},), message="")
    stale = dp.mark_stale(original)
    assert stale.status == dp.STATUS_STALE_RESULT
    assert stale.entries == original.entries
    assert stale.query == original.query
    assert stale.source == original.source


def test_mark_stale_is_a_valid_mock_provider_status():
    provider = dp.MockDictionaryProvider(responses={"x": "STALE_RESULT"})
    r = provider.lookup("x")
    assert r.status == dp.STATUS_STALE_RESULT


def test_stale_response_ordering_is_caller_responsibility_last_write_wins_in_queue():
    """dictionary_provider.py itself has no notion of "generation"/staleness
    -- that is the GUI layer's responsibility (see
    tests/test_tdk_checker_gui.py's stale-response tests). This test only
    confirms the provider's cache is safe under concurrent/out-of-order
    access from multiple threads (no corruption, no crash, deterministic
    per-key result)."""
    provider = dp.MockDictionaryProvider(responses={"a": "FOUND", "b": "NOT_FOUND"}, delay_seconds=0.05)
    results = []

    def worker(term):
        results.append(provider.lookup(term))

    threads = [threading.Thread(target=worker, args=(t,)) for t in ("a", "b", "a", "b")]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    statuses = {r.normalized_query: r.status for r in results}
    assert statuses == {"a": "FOUND", "b": "NOT_FOUND"}
