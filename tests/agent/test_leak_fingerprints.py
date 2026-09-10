"""Cover for the outbound fingerprint gate (``agent/leak_fingerprints.py``).

The 2026-09-09 impact scan turned up two sessions where a user talked the agent
into ``read_file`` on its own ``config.yaml`` and it quoted the file back
verbatim — a well-formed reply, so no reasoning-shape rule could see it.  The
protected text used below is the style block the 2026-09-10 leak actually spilled
("Be direct… No filler…"), which is the same class of content Haro fingerprints
in production.
"""
from __future__ import annotations

import json
import os
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


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_THINK_BUDGET_HINT", raising=False)
    # The store is process-wide; drop any state a sibling test left behind.
    fp._STORE.__init__()
    return tmp_path


def _install(home_dir, text=PROTECTED_PROMPT, ngram=fp.DEFAULT_NGRAM):
    path = home_dir / "replyguard" / "fingerprints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp.build_fingerprints(text, ngram), ensure_ascii=False),
                    encoding="utf-8")
    return path


# ── format ─────────────────────────────────────────────────────────────


def test_digest_format_carries_hashes_only():
    digest = fp.build_fingerprints(PROTECTED_PROMPT)
    assert digest["version"] == 1
    assert digest["ngram"] == 8
    assert digest["hashes"] and digest["lines"]
    assert all(len(h) == 40 and set(h) <= set("0123456789abcdef") for h in digest["hashes"])
    blob = json.dumps(digest, ensure_ascii=False)
    assert "Be direct" not in blob
    assert "s3cr3t" not in blob


def test_generator_script_emits_the_same_digest(tmp_path):
    from scripts.replyguard_fingerprints import main

    source = tmp_path / "prompt.md"
    source.write_text(PROTECTED_PROMPT, encoding="utf-8")
    out = tmp_path / "fingerprints.json"
    assert main([str(source), "-o", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == fp.build_fingerprints(PROTECTED_PROMPT)


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


def test_innocent_reply_is_not_caught(home):
    _install(home)
    tripped, hits = fp.scan_text(INNOCENT_REPLY, fp.store())
    assert (tripped, hits) == (False, 0)


def test_missing_digest_leaves_the_hook_inert(home):
    assert fp.store().empty is True
    assert fp.scan_text(PROTECTED_PROMPT, fp.store()) == (False, 0)


def test_unreadable_digest_fails_open(home):
    path = home / "replyguard" / "fingerprints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert fp.store().empty is True
    assert fp.scan_text(PROTECTED_PROMPT, fp.store()) == (False, 0)


def test_ngrams_straddling_delta_boundaries_still_match(home):
    _install(home)
    scanner = fp.FingerprintScanner(fp.store())
    text = "Match the length of the reply to the weight of the ask."
    for i in range(0, len(text), 3):
        scanner.feed(text[i : i + 3])
    scanner.flush()
    assert scanner.tripped is True


def test_short_lines_are_not_fingerprinted(home):
    """A one-word line would otherwise fire the line rule on every reply."""
    _install(home, "是\nok\n步骤如下\n")
    assert fp.build_fingerprints("是\nok\n步骤如下\n")["lines"] == []
    assert fp.scan_text("是\nok\n", fp.store()) == (False, 0)


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
