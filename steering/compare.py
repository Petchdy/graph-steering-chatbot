"""Side-by-side A/B: unsteered ('none') vs each steered strategy for ONE message.

Same model + same neutral prompt for all — the only difference is the steering vector — so this is
the honest "what does steering add" comparison. Needs the steering service running on STEER_URL.

Run (steering venv/env; service on :8100):
  python steering/compare.py                       # default example message
  python steering/compare.py "your message here"
  python steering/compare.py -n 3 "message"        # 3 samples per strategy (sampling varies)
"""

from __future__ import annotations

import argparse
import os

import requests

URL = os.environ.get("STEER_URL", "http://localhost:8100").rstrip("/")


def gen(strategy: str, message: str) -> str:
    r = requests.post(f"{URL}/generate",
                      json={"messages": [{"role": "user", "content": message}], "strategy": strategy},
                      timeout=200)
    r.raise_for_status()
    return r.json()["response"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?",
                    default="I keep feeling like everyone at work is judging me and I might get fired.")
    ap.add_argument("-n", type=int, default=1, help="samples per strategy")
    args = ap.parse_args()

    offered = requests.get(f"{URL}/strategies", timeout=60).json()["strategies"]
    order = ["none"] + [s for s in offered if s != "none"]

    print(f"MESSAGE: {args.message}\n" + "=" * 78)
    for s in order:
        print(f"\n### {s}")
        for i in range(args.n):
            tag = f" (sample {i+1})" if args.n > 1 else ""
            try:
                print(f"{tag} {gen(s, args.message)}".strip())
            except Exception as exc:  # noqa: BLE001
                print(f"{tag} [error: {exc}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
