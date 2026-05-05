import importlib
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="U.S. Power Grid Outage Risk",
    page_icon="⚡",
    layout="wide",
)

SECTION_ORDER = ["Live Analysis", "AI Analysis"]


def discover_pages():
    here = Path(__file__).parent / "sections"
    pages = []
    for path in sorted(here.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        mod = importlib.import_module(f"sections.{path.stem}")
        if not hasattr(mod, "show"):
            continue
        pages.append({
            "title": getattr(mod, "TITLE", path.stem),
            "icon": getattr(mod, "ICON", "•"),
            "order": getattr(mod, "ORDER", 999),
            "section": getattr(mod, "SECTION", "Other"),
            "show": mod.show,
        })
    pages.sort(key=lambda p: (p["order"], p["title"]))
    return pages


pages = discover_pages()
if not pages:
    st.error("No sections found in app/sections/.")
    st.stop()

groups = {s: [] for s in SECTION_ORDER}
for p in pages:
    groups.setdefault(p["section"], []).append(p)

if "active" not in st.session_state:
    st.session_state.active = pages[0]["title"]

st.sidebar.title("⚡ Outage Risk")
for section, items in groups.items():
    if not items:
        continue
    st.sidebar.markdown(f"#### {section}")
    for p in items:
        active = p["title"] == st.session_state.active
        if st.sidebar.button(
            f"{p['icon']}  {p['title']}",
            key=f"nav_{p['title']}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active = p["title"]
            st.rerun()

current = next(p for p in pages if p["title"] == st.session_state.active)
current["show"]()
