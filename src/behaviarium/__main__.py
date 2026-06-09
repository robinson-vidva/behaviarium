"""Enables ``python -m behaviarium ...`` (how the Streamlit shell spawns background jobs)."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
