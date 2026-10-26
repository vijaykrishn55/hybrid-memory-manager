from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_memory import HybridMemoryManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an assembled memory context payload.")
    parser.add_argument("query", help="Current user query")
    parser.add_argument("--state", type=Path, default=Path("reports/memory_state.json"))
    args = parser.parse_args()

    if not args.state.exists():
        raise SystemExit(f"State file not found: {args.state}. Run scripts/verify_memory.py first.")
    manager = HybridMemoryManager.load(args.state)
    print(json.dumps(manager.build_context_payload(args.query), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
