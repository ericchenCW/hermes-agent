"""Cover for the outbound fingerprint gate (``agent/leak_fingerprints.py``).

The 2026-09-09 impact scan turned up two sessions where a user talked the agent
into ``read_file`` on its own ``config.yaml`` and it quoted the file back
verbatim — a well-formed reply, so no reasoning-shape rule could see it.  The
protected text used below is the style block the 2026-09-10 leak actually spilled
("Be direct… No filler…"), which is the same class of content Haro fingerprints
in production.

The digest format is a cross-language contract with the Haro (Go) generator, so
two kinds of test live here:

* ``tests/agent/fixtures/guard_vectors.json`` pins the expected hashes for a
  handful of awkward lines (Chinese, mixed script, full-width/NFKC, CRLF,
  under-length, exactly one gram).  The Go side must reproduce it byte for byte.
* :func:`_ref_line_hash` / :func:`_ref_ngrams` re-derive those same values from
  ``hashlib`` and a hand-written FNV-1a, so the fixture is not merely the
  implementation agreeing with itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from types import SimpleNamespace

import pytest

from agent import leak_fingerprints as fp
from agent import leak_guard as lg


PROTECTED_PROMPT = (
    "Be direct. No filler, no restating the question back at the user.\n"
    "Plain claims only; agree because it is right, not because it is asked.\n"
    "Match the length of the reply to the weight of the ask.\n"
    "wecom_secret: s3cr3t-do-not-echo-this-token-anywhere\n"
)

INNOCENT_REPLY = (
    "在主机上确认 bkmonitor 是否正常，可以先看进程再看日志：\n"
    "1. `ps -ef | grep bkmonitorbeat`\n"
    "2. `tail -n 100 /var/log/gse/bkmonitorbeat.log`\n"
    "没有 ERROR 就说明采集正常。\n"
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "guard_vectors.json")


# ── independent reference implementation (cross-check, no module reuse) ─
def _ref_normalize(text: str) -> list[str]:
    out = []
    for raw in unicodedata.normalize("NFKC", text.replace("\r\n", "\n")).split("\n"):
        line = " ".join(raw.split())  # trim + collapse whitespace runs
        out.append("".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in line))
    return out


def _ref_line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()[:32]


def _ref_fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) % (1 << 64)
    return h


def _ref_ngrams(line: str) -> list[str]:
    return [
        "%016x" % _ref_fnv1a64(line[i : i + 8].encode("utf-8"))
        for i in range(len(line) - 7)
    ]


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_THINK_BUDGET_HINT", raising=False)
    monkeypatch.delenv("HARO_API_URL", raising=False)
    monkeypatch.delenv("HARO_RUNTIME_TOKEN", raising=False)
    # The store is process-wide; drop any state a sibling test left behind.
    fp._STORE.__init__()
    return tmp_path


def _install(home_dir, text=PROTECTED_PROMPT, bot_id="sre-bot"):
    path = home_dir / "guard" / "prompt-fingerprints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = fp.build_fingerprints([("system_prompt", text)], bot_id=bot_id)
    path.write_text(json.dumps(digest, ensure_ascii=False), encoding="utf-8")
    return path


# ── the cross-language vectors ─────────────────────────────────────────


@pytest.fixture(scope="module")
def vectors():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_fixture_declares_the_contract_parameters(vectors):
    assert vectors["normalize"] == fp.NORMALIZE_ID == "nfkc+trim+collapse-ws+ascii-lower"
    assert vectors["line"] == {"hash": "sha256-16", "minRunes": 12}
    assert vectors["ngram"] == {"n": 8, "unit": "rune", "hash": "fnv1a-64"}
    assert len(vectors["cases"]) >= 5


def test_implementation_reproduces_the_vectors(vectors):
    """What Hermes computes must equal the pinned fixture, case by case."""
    for case in vectors["cases"]:
        lines = [line for line in fp.normalize_lines(case["raw"]) if line]
        assert lines == [case["normalized"]], case["name"]
        line = lines[0]
        assert len(line) == case["runes"], case["name"]
        registered = fp.line_fingerprints(case["raw"])
        assert registered == ([case["lineHash"]] if case["lineHash"] else []), case["name"]
        assert [fp.ngram_hash(w) for w in fp.line_windows(line)] == case["ngrams"], case["name"]


def test_vectors_match_an_independent_implementation(vectors):
    """…and so must a hand-written sha256 + FNV-1a that shares no code with it."""
    for case in vectors["cases"]:
        lines = [line for line in _ref_normalize(case["raw"]) if line]
        assert lines == [case["normalized"]], case["name"]
        line = lines[0]
        expected = _ref_line_hash(line) if len(line) >= 12 else None
        assert expected == case["lineHash"], case["name"]
        assert _ref_ngrams(line) == case["ngrams"], case["name"]


def test_ascii_lower_leaves_other_scripts_alone():
    """``str.lower()`` would fold Cyrillic and diverge from the Go side."""
    assert fp.ascii_lower("ПРИВЕТ Haro") == "ПРИВЕТ haro"
    assert "ПРИВЕТ Haro".lower() != fp.ascii_lower("ПРИВЕТ Haro")


def test_short_lines_and_short_grams_are_not_registered():
    assert fp.line_fingerprints("步骤如下：\nok fine.\n") == []
    assert fp.ngram_fingerprints("步骤如下：\n") == []
    assert len(fp.ngram_fingerprints("ok fine.\n")) == 1


def test_windows_never_straddle_a_line_break():
    """Two 4-rune lines must not produce a gram spanning both."""
    assert fp.ngram_fingerprints("abcd\nefgh\n") == []


# ── file format ────────────────────────────────────────────────────────


def test_digest_format_carries_hashes_only():
    digest = fp.build_fingerprints(
        [("answer_rules", PROTECTED_PROMPT)], bot_id="sre-bot",
        generated_at="2026-09-10T00:00:00Z",
    )
    assert digest["version"] == 1
    assert digest["botId"] == "sre-bot"
    assert digest["generatedAt"] == "2026-09-10T00:00:00Z"
    assert digest["normalize"] == "nfkc+trim+collapse-ws+ascii-lower"
    assert digest["line"] == {"hash": "sha256-16", "minRunes": 12}
    assert digest["ngram"] == {"n": 8, "unit": "rune", "hash": "fnv1a-64"}
    assert digest["sources"] == [
        {"id": "answer_rules", "lines": len(digest["lines"]),
         "ngrams": len(digest["ngrams"])}
    ]
    assert all(len(h) == 32 and set(h) <= set("0123456789abcdef") for h in digest["lines"])
    assert all(len(h) == 16 and set(h) <= set("0123456789abcdef") for h in digest["ngrams"])
    assert digest["lines"] == sorted(set(digest["lines"]))
    assert digest["ngrams"] == sorted(set(digest["ngrams"]))
    blob = json.dumps(digest, ensure_ascii=False)
    assert "Be direct" not in blob
    assert "s3cr3t" not in blob


def test_generated_at_defaults_to_rfc3339_utc():
    generated = fp.build_fingerprints([("s", "x")], bot_id="b")["generatedAt"]
    assert generated.endswith("Z") and len(generated) == 20


def test_multiple_sources_are_counted_separately():
    digest = fp.build_fingerprints(
        [("answer_rules", PROTECTED_PROMPT), ("style", "Match the length of the reply\n")],
        bot_id="sre-bot",
    )
    assert [s["id"] for s in digest["sources"]] == ["answer_rules", "style"]
    assert all(s["lines"] >= 1 and s["ngrams"] >= 1 for s in digest["sources"])


def test_generator_script_emits_the_same_digest(tmp_path):
    from scripts.replyguard_fingerprints import main

    source = tmp_path / "prompt.md"
    source.write_text(PROTECTED_PROMPT, encoding="utf-8")
    out = tmp_path / "prompt-fingerprints.json"
    assert main(["--bot-id", "sre-bot", "--source", f"answer_rules={source}",
                 "--generated-at", "2026-09-10T00:00:00Z", "-o", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == fp.build_fingerprints(
        [("answer_rules", PROTECTED_PROMPT)], bot_id="sre-bot",
        generated_at="2026-09-10T00:00:00Z",
    )


def test_generator_rejects_a_malformed_source(tmp_path):
    from scripts.replyguard_fingerprints import main

    with pytest.raises(SystemExit):
        main(["--bot-id", "b", "--source", "no-equals-sign"])


def test_store_reads_the_contract_path(home):
    _install(home)
    assert fp.fingerprints_path() == str(home / "guard" / "prompt-fingerprints.json")
    assert fp.store().empty is False
    assert fp.store().bot_id == "sre-bot"


# ── matching ───────────────────────────────────────────────────────────


def test_protected_text_is_caught(home):
    _install(home)
    tripped, hits = fp.scan_text(
        "这是配置里的内容：\nBe direct. No filler, no restating the question back at the user.\n",
        fp.store(),
    )
    assert tripped is True
    assert hits >= 1


def test_a_secret_line_alone_trips_the_gate(home):
    _install(home)
    tripped, _ = fp.scan_text("wecom_secret: s3cr3t-do-not-echo-this-token-anywhere\n", fp.store())
    assert tripped is True


def test_a_requoted_line_still_trips_via_adjacent_ngrams(home):
    """Re-wrapped text loses the line hash but keeps a long run of grams."""
    _install(home)
    tripped, hits = fp.scan_text(
        "参考：Plain claims only; agree because it is right, not because it is "
        "asked. 就这样。", fp.store(),
    )
    assert tripped is True
    assert hits >= fp.NGRAM_RUN_THRESHOLD


def test_a_lone_ngram_hit_does_not_trip(home):
    """One 8-gram of a common phrase is a collision, not a leak."""
    _install(home, "the length of the reply must never be echoed anywhere at all\n")
    scanner = fp.FingerprintScanner(fp.store())
    # A single window of the protected text, embedded in unrelated prose.
    scanner.feed("我们讨论一下 the leng 之外的问题吧,和上面无关。")
    scanner.flush()
    assert scanner.tripped is False


def test_three_adjacent_ngrams_are_needed(home):
    _install(home, "abcdefghijklmnop\n")
    protected = fp.build_fingerprints([("s", "abcdefghijklmnop\n")], bot_id="b")["ngrams"]
    assert len(protected) >= fp.NGRAM_RUN_THRESHOLD
    # Two adjacent windows only ("abcdefghi" gives windows 0 and 1) → acquitted.
    assert fp.scan_text("xx abcdefghi yy", fp.store())[0] is False
    # Three adjacent windows ("abcdefghij") → convicted.
    assert fp.scan_text("xx abcdefghij yy", fp.store())[0] is True


def test_innocent_reply_is_not_caught(home):
    _install(home)
    tripped, hits = fp.scan_text(INNOCENT_REPLY, fp.store())
    assert (tripped, hits) == (False, 0)


def test_missing_digest_leaves_the_hook_inert(home):
    assert fp.store().empty is True
    assert fp.scan_text(PROTECTED_PROMPT, fp.store()) == (False, 0)


def test_unreadable_digest_fails_open(home):
    path = home / "guard" / "prompt-fingerprints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert fp.store().empty is True
    assert fp.scan_text(PROTECTED_PROMPT, fp.store()) == (False, 0)


def test_legacy_digest_keys_are_ignored(home):
    """A pre-contract file (``hashes``/``ngram``) must read as empty, not crash."""
    path = home / "guard" / "prompt-fingerprints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "ngram": 8, "hashes": ["deadbeef"]}),
                    encoding="utf-8")
    assert fp.store().empty is True


def test_a_line_split_across_deltas_still_matches(home):
    _install(home)
    scanner = fp.FingerprintScanner(fp.store())
    text = "wecom_secret: s3cr3t-do-not-echo-this-token-anywhere\n"
    for i in range(0, len(text), 3):
        scanner.feed(text[i : i + 3])
    scanner.flush()
    assert scanner.tripped is True


def test_an_unterminated_final_line_is_judged_on_flush(home):
    _install(home)
    scanner = fp.FingerprintScanner(fp.store())
    scanner.feed("Match the length of the reply to the weight of the ask.")
    assert scanner.flush() is True


def test_a_crlf_source_matches_an_lf_reply(home):
    _install(home, "wecom_secret: s3cr3t-do-not-echo-this-token\r\n")
    assert fp.scan_text("wecom_secret: s3cr3t-do-not-echo-this-token\n", fp.store())[0] is True


def test_nfkc_and_whitespace_variants_still_match(home):
    _install(home, "Haro 运维助手 接入企业微信\n")
    assert fp.scan_text("　Ｈａｒｏ　运维助手　　接入企业微信　\n", fp.store())[0] is True


# ── hot reload ─────────────────────────────────────────────────────────


def test_digest_is_reloaded_when_the_file_changes(home):
    _install(home, "alpha bravo charlie delta echo foxtrot golf hotel\n")
    assert fp.scan_text("alpha bravo charlie delta echo foxtrot golf hotel", fp.store())[0] is True
    assert fp.scan_text("november oscar papa quebec romeo sierra tango", fp.store())[0] is False

    path = _install(home, "november oscar papa quebec romeo sierra tango\n")
    os.utime(path, (1e9, 1e9))  # force a distinct mtime
    assert fp.scan_text("november oscar papa quebec romeo sierra tango", fp.store())[0] is True


def test_removing_the_file_disarms_the_hook(home):
    path = _install(home)
    assert fp.store().empty is False
    path.unlink()
    assert fp.store().empty is True


# ── shared replacement / audit path with the reasoning-leak guard ──────


def test_guard_replaces_a_fingerprinted_reply_and_audits_it(home):
    _install(home)
    guard = lg.StreamLeakGuard(budget=2500, platform="wecom", session="s", subject="u")
    outcome = guard.on_content_delta(
        "配置文件内容如下：\nwecom_secret: s3cr3t-do-not-echo-this-token-anywhere\n"
    )

    assert outcome.convicted is True
    assert outcome.emit == ""
    assert guard.rules == ["fingerprint"]
    assert guard.final_text("whatever") == lg.FINGERPRINT_REDACTION_TEXT

    events = [json.loads(line) for line in
              (home / "logs" / "replyguard.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["rule"] == "fingerprint"
    assert events[0]["hit_count"] >= 1
    assert "s3cr3t" not in json.dumps(events[0], ensure_ascii=False)


def test_guard_leaves_an_innocent_reply_alone_with_fingerprints_loaded(home):
    _install(home)
    guard = lg.StreamLeakGuard(budget=2500, platform="wecom", session="s", subject="u")
    emitted = "".join(guard.on_content_delta(chunk).emit
                      for chunk in [INNOCENT_REPLY[:40], INNOCENT_REPLY[40:]])
    guard.on_usage(SimpleNamespace(
        completion_tokens_details=SimpleNamespace(reasoning_tokens=600)))
    guard.finish()

    assert guard.convicted is False
    assert emitted == INNOCENT_REPLY
    assert not (home / "logs" / "replyguard.jsonl").exists()
