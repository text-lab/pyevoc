"""General HTML report helpers."""
from __future__ import annotations
from pathlib import Path


def write_html_report(sections: dict[str, str], path: str | Path, title: str = "PyEvoc Report") -> Path:
    html = [f"<html><head><meta charset='utf-8'><title>{title}</title></head><body><h1>{title}</h1>"]
    for name, content in sections.items():
        html.append(f"<h2>{name}</h2>\n{content}")
    html.append("</body></html>")
    p = Path(path)
    p.write_text("\n".join(html), encoding="utf-8")
    return p
