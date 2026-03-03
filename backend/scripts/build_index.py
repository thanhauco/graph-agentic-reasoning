"""CLI: build the GraphRAG index.

Usage:
    python -m scripts.build_index
"""

from __future__ import annotations

import logging

from app.graphrag.index import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    result = run()
    print("Index build complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
