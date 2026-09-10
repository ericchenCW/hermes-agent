#!/usr/bin/env python3
"""Build the reply-guard fingerprint digest from the text that must not be echoed.

    python scripts/replyguard_fingerprints.py --bot-id sre-bot \
        --source answer_rules=config/answer_rules.md \
        --source system_prompt=config/system_prompt.md \
        -o ~/.hermes/guard/prompt-fingerprints.json

Each ``--source id=path`` is fingerprinted separately (its per-source counts go
into the ``sources`` manifest) and the hash sets are then merged.  The output
holds hashes ONLY — never the protected text — so it is safe to ship to every
gateway host.  Install it at ``$HERMES_HOME/guard/prompt-fingerprints.json``
(Haro does this over WriteHome, tmp+rename 0644); the gateway polls the file's
mtime and picks up a new digest without a restart.  Absent file ⇒ the hook is
inert.

This is the reference writer for the format described in
``agent/leak_fingerprints.py``, which also holds the matching side and the
thresholds.  The Haro Go generator must agree with it byte for byte;
``tests/agent/fixtures/guard_vectors.json`` is the cross-check corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.leak_fingerprints import build_fingerprints, read_sources  # noqa: E402


def parse_source(raw: str) -> tuple[str, str]:
    """``id=path`` → ``(id, path)``."""
    source_id, sep, path = str(raw).partition("=")
    if not sep or not source_id.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            f"--source expects id=path, got {raw!r}"
        )
    return source_id.strip(), path.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bot-id", required=True,
                        help="runtime bot id this digest belongs to")
    parser.add_argument("--source", action="append", type=parse_source, default=[],
                        metavar="ID=PATH", dest="sources",
                        help="a file whose content must never be echoed (repeatable)")
    parser.add_argument("--generated-at",
                        help="RFC3339 timestamp (default: now, UTC) — pin it for "
                             "reproducible output")
    parser.add_argument("-o", "--output", help="write here instead of stdout")
    args = parser.parse_args(argv)

    if not args.sources:
        parser.error("at least one --source id=path is required")

    digest = build_fingerprints(
        read_sources(args.sources),
        bot_id=args.bot_id,
        generated_at=args.generated_at,
    )
    payload = json.dumps(digest, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"{len(digest['lines'])} line + {len(digest['ngrams'])} 8-gram "
              f"fingerprints → {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
