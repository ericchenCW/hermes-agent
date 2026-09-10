#!/usr/bin/env python3
"""Build the reply-guard fingerprint digest from the text that must not be echoed.

    python scripts/replyguard_fingerprints.py config/system_prompt.md > fingerprints.json
    python scripts/replyguard_fingerprints.py prompt.md config.yaml \
        -o ~/.hermes/replyguard/fingerprints.json

Several source files may be given; the digest covers all of them.  The output
holds sha1 hashes ONLY — never the protected text — so it is safe to ship to
every gateway host.  Install it at ``$HERMES_HOME/replyguard/fingerprints.json``
(Haro does this over WriteHome); the gateway polls the file's mtime and picks up
a new digest without a restart.  Absent file ⇒ the hook is inert.

See ``agent/leak_fingerprints.py`` for the matching side and the thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.leak_fingerprints import (  # noqa: E402
    DEFAULT_NGRAM, build_fingerprints, iter_protected_sources,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+", help="file(s) whose content must never be echoed")
    parser.add_argument("-n", "--ngram", type=int, default=DEFAULT_NGRAM,
                        help=f"character n-gram size (default {DEFAULT_NGRAM})")
    parser.add_argument("-o", "--output", help="write here instead of stdout")
    args = parser.parse_args(argv)

    if args.ngram < 2:
        parser.error("--ngram must be at least 2")

    digest = build_fingerprints(iter_protected_sources(args.sources), args.ngram)
    payload = json.dumps(digest, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"{len(digest['hashes'])} n-gram + {len(digest['lines'])} line fingerprints "
              f"→ {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
