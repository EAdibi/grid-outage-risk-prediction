"""Dashboard entrypoint.

Auto-discovers any `sections/*.py` that exports `show()` and builds the
sidebar from the modules' TITLE / ICON / ORDER metadata. Add a page by
dropping a new file in `sections/` — no edits here are needed.
"""
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import streamlit as st

st.set_page_config(
    page_title="U.S. Power Grid Outage Risk",
    page_icon="⚡",
    layout="wide",
)


@dataclass
class Page:
    title: str
    icon: str
    order: int
    show: Callable[[], None]


def _discover_pages() -> list[Page]:
    sections_dir = Path(__file__).parent / "sections"
    pages: list[Page] = []
    for path in sorted(sections_dir.glob("*.py")):
        if path.stem.startswith("_") or path.stem == "__init__":
            continue
        module = importlib.import_module(f"sections.{path.stem}")
        show = getattr(module, "show", None)
        if not callable(show):
            continue
        pages.append(Page(
            title=getattr(module, "TITLE", path.stem),
            icon=getattr(module, "ICON", "•"),
            order=getattr(module, "ORDER", 999),
            show=show,
        ))
    pages.sort(key=lambda p: (p.order, p.title))
    return pages


pages = _discover_pages()

st.title("⚡ U.S. Power Grid Outage Risk Intelligence Platform")
st.markdown("Predicting county-level outage risk across all 50 states.")

st.sidebar.title("Navigation")
if not pages:
    st.sidebar.info("No sections found in `app/sections/`.")
    st.stop()

labels = [f"{p.icon}  {p.title}" for p in pages]
choice = st.sidebar.radio("Go to", labels, label_visibility="collapsed")
pages[labels.index(choice)].show()
