"""HTML outputs for EVOC quadrant tables."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def evoc_quadrants_to_html(terms: pd.DataFrame, path: str | Path | None = None, title: str = "EVOC Quadrants") -> str:
    html = [f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>", f"<h1>{title}</h1>"]
    for q, grp in terms.groupby("quadrant", sort=False):
        html.append(f"<h2>{q}</h2>")
        html.append(grp.to_html(index=False, escape=False))
    html.append("</body></html>")
    result = "\n".join(html)
    if path:
        Path(path).write_text(result, encoding="utf-8")
    return result
