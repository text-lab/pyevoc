"""EVOC thresholding, quadrant assignment, and compact HTML reporting.

This module assigns terms to the four classical EVOC/HEM quadrants after
term-level statistics, concreteness labelling, and emoji description labelling
have been computed.

The module implements the EVOC-ready logic used in PyEvoc:

- optional minimal-frequency filtering;
- POS-specific AFE and AOE thresholds;
- user-level or document-level relative diffusion;
- rounded comparison values for stable table construction;
- diffusion/salience classes;
- ordered quadrant labels;
- publication-ready EVOC table;
- threshold and quadrant diagnostics;
- optional compact HTML reports by UPOS category.

The expected input is a term-level dataframe containing at least:

- term
- upos
- term_type
- n_docs or n_posts
- n_users
- R_pos
- R_struct
- S
- Rank
- freq_for_quadrant
- user_penetration
- posts_per_user

Optional enrichment columns such as ``concreteness_label`` and
``emoji_description`` are preserved when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import html
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FOCAL_UPOS = {"NOUN", "ADJ", "EMOJI"}

QUADRANT_ORDER = [
    "Central nucleus",
    "First periphery",
    "Contrast zone",
    "Peripheral system",
]

UPOS_LABELS = {
    "NOUN": "Nouns",
    "ADJ": "Adjectives",
    "EMOJI": "Emojis",
}

MACHINE_TO_HUMAN_QUADRANT = {
    "central_nucleus": "Central nucleus",
    "first_periphery": "First periphery",
    "contrast_zone": "Contrast zone",
    "peripheral_system": "Peripheral system",
}

HUMAN_TO_MACHINE_QUADRANT = {
    value: key
    for key, value in MACHINE_TO_HUMAN_QUADRANT.items()
}


@dataclass(frozen=True)
class QuadrantConfig:
    """Configuration for EVOC quadrant assignment and optional HTML reporting."""

    minimal_freq: int = 2
    round_digits: int = 2

    focal_upos: set[str] = field(default_factory=lambda: set(DEFAULT_FOCAL_UPOS))
    quadrant_order: list[str] = field(default_factory=lambda: list(QUADRANT_ORDER))

    diffusion_basis: str = "user_penetration"

    term_col: str = "term"
    upos_col: str = "upos"
    term_type_col: str = "term_type"

    n_docs_col: str = "n_docs"
    n_posts_col: str = "n_posts"
    n_users_col: str = "n_users"

    r_pos_col: str = "R_pos"
    r_struct_col: str = "R_struct"
    salience_col: str = "S"
    rank_col: str = "Rank"

    freq_col: str = "freq_for_quadrant"

    user_penetration_col: str = "user_penetration"
    document_penetration_col: str = "document_penetration"
    post_penetration_col: str = "post_penetration"
    posts_per_user_col: str = "posts_per_user"

    concreteness_score_col: str = "concreteness_score"
    concreteness_label_col: str = "concreteness_label"
    concreteness_in_lexicon_col: str = "concreteness_in_lexicon"

    emoji_description_col: str = "emoji_description"

    human_readable: bool = True

    generate_html: bool = False
    html_output_dir: str | Path = "evoc_outputs"
    html_top_n: int = 20
    html_max_width_px: int = 950

    verbose: bool = True


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    """Validate required columns."""

    missing = columns.difference(df.columns)

    if missing:
        raise ValueError(f"term_stats_df is missing required columns: {sorted(missing)}")


def _normalise_input_quadrant(value: object) -> object:
    """Normalise existing quadrant labels to human-readable labels."""

    if value is None or pd.isna(value):
        return pd.NA

    value = str(value)

    if value in MACHINE_TO_HUMAN_QUADRANT:
        return MACHINE_TO_HUMAN_QUADRANT[value]

    return value


def _ensure_n_posts_alias(df: pd.DataFrame, cfg: QuadrantConfig) -> pd.DataFrame:
    """Ensure that the dataframe contains an n_posts-compatible column."""

    out = df.copy()

    if cfg.n_posts_col not in out.columns and cfg.n_docs_col in out.columns:
        out[cfg.n_posts_col] = out[cfg.n_docs_col]

    if cfg.n_docs_col not in out.columns and cfg.n_posts_col in out.columns:
        out[cfg.n_docs_col] = out[cfg.n_posts_col]

    return out


def _select_diffusion_basis(
    df: pd.DataFrame,
    cfg: QuadrantConfig,
) -> pd.Series:
    """Return the relative diffusion series used for quadrant assignment."""

    valid = {
        "user_penetration",
        "post_penetration",
        "document_penetration",
        "diffusion_penetration",
    }

    if cfg.diffusion_basis not in valid:
        raise ValueError(
            "diffusion_basis must be one of "
            f"{sorted(valid)}."
        )

    if cfg.diffusion_basis == "user_penetration":
        col = cfg.user_penetration_col
    elif cfg.diffusion_basis == "post_penetration":
        col = cfg.post_penetration_col
    elif cfg.diffusion_basis == "document_penetration":
        col = cfg.document_penetration_col
    else:
        col = "diffusion_penetration"

    if col not in df.columns:
        raise ValueError(
            f"diffusion_basis='{cfg.diffusion_basis}' requires column '{col}'."
        )

    return pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------------------------
# Compact EVOC HTML report helpers
# ---------------------------------------------------------------------------

def format_value(value: object, digits: int = 2) -> str:
    """Format a numeric value for EVOC compact HTML tables."""

    if value is None or pd.isna(value):
        return "-"

    return f"{float(value):.{digits}f}"


def format_int(value: object) -> str:
    """Format an integer-like value for EVOC compact HTML tables."""

    if value is None or pd.isna(value):
        return "-"

    return f"{int(round(float(value))):,}"


def concreteness_class(label: object) -> str:
    """Return a CSS class for a concreteness label."""

    value = str(label).strip().lower()

    if value == "concrete":
        return "conc-concrete"

    if value == "abstract":
        return "conc-abstract"

    if value == "mixed":
        return "conc-mixed"

    return "conc-missing"


def get_threshold_values(
    pos_thresholds_round_df: pd.DataFrame,
    upos: str,
) -> tuple[str, str]:
    """Return formatted AFE/AOE threshold values for one UPOS category."""

    required = {"upos", "AFE_thr_round", "AOE_thr_round"}
    _require_columns(pos_thresholds_round_df, required)

    sub = pos_thresholds_round_df.loc[
        pos_thresholds_round_df["upos"].astype(str).eq(upos)
    ]

    if sub.empty:
        return "-", "-"

    afe = format_value(sub["AFE_thr_round"].iloc[0], digits=2)
    aoe = format_value(sub["AOE_thr_round"].iloc[0], digits=2)

    return afe, aoe


def quadrant_rule_label(quadrant: str, afe: str, aoe: str) -> str:
    """Return the rule label printed under each quadrant title."""

    if quadrant == "Central nucleus":
        return f"Rel. diff. ≥ {afe}; AOE ≤ {aoe}"

    if quadrant == "First periphery":
        return f"Rel. diff. ≥ {afe}; AOE > {aoe}"

    if quadrant == "Contrast zone":
        return f"Rel. diff. < {afe}; AOE ≤ {aoe}"

    if quadrant == "Peripheral system":
        return f"Rel. diff. < {afe}; AOE > {aoe}"

    return ""


def select_top_terms(
    df: pd.DataFrame,
    quadrant: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Select the top terms for one quadrant using the legacy EVOC ordering."""

    required_cols = {"relative_diffusion", "frequency", "Rank"}
    missing = required_cols.difference(df.columns)

    if missing:
        raise ValueError(
            f"Input dataframe is missing required ordering columns: {missing}"
        )

    if quadrant in {"Central nucleus", "First periphery"}:
        sort_cols = ["relative_diffusion", "Rank", "frequency"]
        ascending = [False, True, False]

    elif quadrant == "Contrast zone":
        sort_cols = ["Rank", "relative_diffusion", "frequency"]
        ascending = [False, False, False]

    elif quadrant == "Peripheral system":
        sort_cols = ["relative_diffusion", "Rank", "frequency"]
        ascending = [False, False, False]

    else:
        raise ValueError(f"Unknown quadrant: {quadrant}")

    return df.sort_values(sort_cols, ascending=ascending).head(top_n)


def build_quadrant_block(
    df: pd.DataFrame,
    quadrant: str,
    upos: str,
    afe: str,
    aoe: str,
    top_n: int = 20,
) -> str:
    """Build one HTML card for a given quadrant and UPOS category."""

    sub = df[
        (df["quadrant"].astype(str) == quadrant)
        & (df["upos"].astype(str) == upos)
    ].copy()

    if not sub.empty:
        sub = select_top_terms(sub, quadrant=quadrant, top_n=top_n)

    if upos == "EMOJI":
        term_header = "Emoji"
        extra_header = "Description"
    else:
        term_header = "Term"
        extra_header = "Conc."

    if sub.empty:
        rows = """
        <tr>
            <td colspan="5" class="empty">No retained terms</td>
        </tr>
        """
    else:
        row_list = []

        for _, row in sub.iterrows():
            term = html.escape(str(row["term"]))

            relative_diffusion = format_value(
                row.get("relative_diffusion", pd.NA),
                digits=2,
            )

            absolute_frequency = format_int(row.get("frequency", pd.NA))

            rank = format_value(row.get("Rank", pd.NA), digits=2)

            if upos == "EMOJI":
                extra = html.escape(str(row.get("emoji_description", "-")))
                extra_cell = f'<td class="desc">{extra}</td>'
            else:
                label = str(row.get("concreteness_label", "-"))
                label_html = html.escape(label)
                label_class = concreteness_class(label)

                extra_cell = (
                    f'<td class="conc {label_class}">'
                    f'<strong>{label_html}</strong>'
                    f'</td>'
                )

            row_list.append(
                f"""
                <tr>
                    <td class="term">{term}</td>
                    <td class="num">{relative_diffusion}</td>
                    <td class="num">{absolute_frequency}</td>
                    <td class="num">{rank}</td>
                    {extra_cell}
                </tr>
                """
            )

        rows = "\n".join(row_list)

    rule = html.escape(quadrant_rule_label(quadrant, afe, aoe))
    css_class = quadrant.lower().replace(" ", "-")

    return f"""
    <div class="quad-card {css_class}">
        <div class="quad-title">
            <div>{html.escape(quadrant)}</div>
            <div class="quad-rule">{rule}</div>
        </div>

        <table class="evoc-table">
            <thead>
                <tr>
                    <th>{term_header}</th>
                    <th class="num">Rel. diff.</th>
                    <th class="num">Abs.</th>
                    <th class="num">Rank</th>
                    <th>{extra_header}</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """


def build_evoc_html_for_upos(
    evoc_quadrants_df: pd.DataFrame,
    upos: str,
    pos_thresholds_round_df: pd.DataFrame,
    *,
    output_dir: str | Path = "evoc_outputs",
    top_n: int = 20,
    max_width_px: int = 950,
) -> str:
    """Write the compact EVOC HTML report for one UPOS category."""

    required_cols = {
        "term",
        "upos",
        "quadrant",
        "relative_diffusion",
        "frequency",
        "Rank",
    }

    missing = required_cols.difference(evoc_quadrants_df.columns)

    if missing:
        raise ValueError(
            f"evoc_quadrants_df is missing required columns: {missing}. "
            "Re-run the updated EVOC threshold module first."
        )

    df = evoc_quadrants_df.copy()

    if "concreteness_label" not in df.columns:
        df["concreteness_label"] = "-"

    if "emoji_description" not in df.columns:
        df["emoji_description"] = "-"

    # Empty reports are allowed. If a UPOS category contains
    # no retained terms, an HTML report is still generated with
    # four empty quadrants displaying "No retained terms".

    upos_label = UPOS_LABELS.get(upos, upos)
    afe, aoe = get_threshold_values(pos_thresholds_round_df, upos)

    blocks = "\n".join(
        build_quadrant_block(
            df=df,
            quadrant=quadrant,
            upos=upos,
            afe=afe,
            aoe=aoe,
            top_n=top_n,
        )
        for quadrant in QUADRANT_ORDER
    )

    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EVOC Quadrants - {html.escape(upos_label)}</title>

<style>

    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 12px auto;
        padding: 0 6px;
        background: #ffffff;
        color: #222222;
        max-width: {max_width_px}px;
    }}

    h1 {{
        font-size: 18px;
        margin: 0 0 4px 0;
        font-weight: 700;
    }}

    .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }}

    .quad-card {{
        border: 1px solid #d8d8d8;
        background: #ffffff;
        break-inside: avoid;
    }}

    .quad-title {{
        font-size: 15px;
        font-weight: 700;
        padding: 5px 7px;
        background: #ebebeb;
        border-bottom: 1px solid #d8d8d8;
    }}

    .quad-rule {{
        font-size: 10px;
        font-weight: 500;
        color: #555555;
        margin-top: 2px;
        line-height: 1.15;
    }}

    .central-nucleus .quad-title {{ color: #4f7fa6; }}
    .first-periphery .quad-title {{ color: #4f9a55; }}
    .contrast-zone .quad-title {{ color: #b84848; }}
    .peripheral-system .quad-title {{ color: #8250a0; }}

    .evoc-table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }}

    .evoc-table th {{
        font-size: 10.5px;
        padding: 4px 4px;
        background: #f4f4f4;
        border-bottom: 1px solid #dddddd;
        text-align: left;
        font-weight: 700;
    }}

    .evoc-table td {{
        font-size: 11.5px;
        padding: 3px 4px;
        border-bottom: 1px solid #eeeeee;
        vertical-align: top;
        line-height: 1.18;
    }}

    .evoc-table th:nth-child(1),
    .evoc-table td:nth-child(1) {{
        width: 30%;
        max-width: 30%;
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        overflow-wrap: anywhere;
        word-break: normal;
    }}

    .evoc-table th:nth-child(2),
    .evoc-table td:nth-child(2) {{
        width: 15%;
    }}

    .evoc-table th:nth-child(3),
    .evoc-table td:nth-child(3) {{
        width: 11%;
    }}

    .evoc-table th:nth-child(4),
    .evoc-table td:nth-child(4) {{
        width: 12%;
    }}

    .evoc-table th:nth-child(5),
    .evoc-table td:nth-child(5) {{
        width: 32%;
    }}

    .evoc-table th.num,
    .evoc-table td.num {{
        text-align: center;
        white-space: nowrap;
    }}

    td.term {{
        font-weight: 500;
    }}

    td.desc {{
        font-size: 10px;
        color: #555555;
        word-break: normal;
        overflow-wrap: anywhere;
    }}

    .conc-concrete {{ color: #008578; }}
    .conc-abstract {{ color: #0067c7; }}
    .conc-mixed {{ color: #7a2ca0; }}
    .conc-missing {{ color: #777777; }}

    .empty {{
        text-align: center;
        color: #888888;
        font-style: italic;
        padding: 8px;
    }}

</style>
</head>

<body>

    <h1>EVOC Quadrants: {html.escape(upos_label)}</h1>

    <div class="grid">
        {blocks}
    </div>

</body>
</html>
"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"evoc_quadrants_{upos.lower()}_compact.html"
    output_file.write_text(html_doc, encoding="utf-8")

    print(f"Saved: {output_file}")

    return str(output_file)


def generate_evoc_html_reports(
    evoc_quadrants_df: pd.DataFrame,
    pos_thresholds_round_df: pd.DataFrame,
    *,
    output_dir: str | Path = "evoc_outputs",
    top_n: int = 20,
    max_width_px: int = 950,
    upos_values: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Generate compact EVOC HTML reports for the requested UPOS categories."""

    upos_values = list(upos_values or ["NOUN", "ADJ", "EMOJI"])

    outputs: dict[str, str] = {}

    for upos in upos_values:

        outputs[upos] = build_evoc_html_for_upos(
            evoc_quadrants_df=evoc_quadrants_df,
            upos=upos,
            pos_thresholds_round_df=pos_thresholds_round_df,
            output_dir=output_dir,
            top_n=top_n,
            max_width_px=max_width_px,
        )

    return outputs


def assign_evoc_quadrants(
    term_stats_df: pd.DataFrame,
    *,
    minimal_freq: int = 2,
    focal_upos: set[str] | None = None,
    quadrant_order: list[str] | None = None,
    round_digits: int = 2,
    diffusion_basis: str = "user_penetration",
    generate_html: bool = False,
    html_output_dir: str | Path = "evoc_outputs",
    html_top_n: int = 20,
    html_max_width_px: int = 950,
    config: QuadrantConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign EVOC quadrants using relative AFE and mean AOE thresholds.

    Parameters
    ----------
    term_stats_df:
        Term-level dataframe.
    minimal_freq:
        Minimal absolute frequency used to retain terms for quadrant
        assignment.
    focal_upos:
        POS categories retained for EVOC assignment.
    quadrant_order:
        Ordered quadrant labels.
    round_digits:
        Number of digits used for rounded threshold comparisons.
    diffusion_basis:
        Relative diffusion variable used to compute AFE thresholds. Supported
        values are ``user_penetration``, ``document_penetration``,
        ``post_penetration`` and ``diffusion_penetration``.
    generate_html:
        If True, write compact EVOC HTML reports by UPOS category.
    html_output_dir:
        Directory where HTML files are written.
    html_top_n:
        Number of terms displayed in each quadrant card.
    html_max_width_px:
        Maximum width of the generated HTML page.
    config:
        Optional complete configuration. If supplied, explicit keyword
        arguments above are ignored unless they are already encoded in config.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        ``evoc_quadrants_df``, ``quadrant_counts_df`` and
        ``pos_thresholds_round_df``.

    Notes
    -----
    For backward compatibility, the function always returns three objects.
    When HTML reports are generated, their paths are stored in
    ``evoc_quadrants_df.attrs["html_outputs"]``.
    """

    cfg = config or QuadrantConfig(
        minimal_freq=minimal_freq,
        focal_upos=set(focal_upos or DEFAULT_FOCAL_UPOS),
        quadrant_order=list(quadrant_order or QUADRANT_ORDER),
        round_digits=round_digits,
        diffusion_basis=diffusion_basis,
        generate_html=generate_html,
        html_output_dir=html_output_dir,
        html_top_n=html_top_n,
        html_max_width_px=html_max_width_px,
    )

    if cfg.verbose:
        print(
            "Assigning EVOC quadrants using relative AFE and mean AOE thresholds..."
        )

    df = _ensure_n_posts_alias(term_stats_df, cfg)

    required_cols = {
        cfg.term_col,
        cfg.upos_col,
        cfg.term_type_col,
        cfg.n_posts_col,
        cfg.n_users_col,
        cfg.r_pos_col,
        cfg.r_struct_col,
        cfg.salience_col,
        cfg.rank_col,
        cfg.freq_col,
        cfg.user_penetration_col,
        cfg.posts_per_user_col,
    }

    _require_columns(df, required_cols)

    df = df[df[cfg.upos_col].isin(cfg.focal_upos)].copy()

    if df.empty:
        raise ValueError("After POS filtering, no terms remain.")

    df[cfg.freq_col] = pd.to_numeric(df[cfg.freq_col], errors="coerce")
    df[cfg.rank_col] = pd.to_numeric(df[cfg.rank_col], errors="coerce")
    df[cfg.user_penetration_col] = pd.to_numeric(
        df[cfg.user_penetration_col],
        errors="coerce",
    )

    df["relative_diffusion"] = _select_diffusion_basis(df, cfg)

    df = df.loc[
        df[cfg.freq_col].notna()
        & df["relative_diffusion"].notna()
        & df[cfg.rank_col].notna()
        & (df[cfg.freq_col] >= cfg.minimal_freq)
    ].copy()

    if df.empty:
        raise ValueError(
            "No terms survive the frequency filter; check minimal_freq."
        )

    pos_thresholds_df = (
        df
        .groupby(cfg.upos_col, observed=True, sort=False)
        .agg(
            n_terms_pos=(cfg.term_col, "size"),
            AFE_thr=("relative_diffusion", "mean"),
            AOE_thr=(cfg.rank_col, "mean"),
            mean_absolute_frequency=(cfg.freq_col, "mean"),
        )
        .reset_index()
        .rename(columns={cfg.upos_col: "upos"})
    )

    df = df.merge(
        pos_thresholds_df[
            [
                "upos",
                "AFE_thr",
                "AOE_thr",
                "mean_absolute_frequency",
            ]
        ],
        left_on=cfg.upos_col,
        right_on="upos",
        how="left",
        suffixes=("", "_thr"),
    )

    if cfg.upos_col != "upos":
        df = df.rename(columns={cfg.upos_col: "upos"})
    else:
        if "upos_thr" in df.columns:
            df = df.drop(columns=["upos_thr"])

    if df["AFE_thr"].isna().any() or df["AOE_thr"].isna().any():
        raise ValueError("Some POS-specific AFE/AOE thresholds are missing.")

    df["frequency"] = df[cfg.freq_col]

    df["relative_diffusion_r"] = df["relative_diffusion"].round(cfg.round_digits)
    df["Rank_r"] = df[cfg.rank_col].round(cfg.round_digits)
    df["AFE_thr_r"] = df["AFE_thr"].round(cfg.round_digits)
    df["AOE_thr_r"] = df["AOE_thr"].round(cfg.round_digits)

    df["diffusion_class"] = np.where(
        df["relative_diffusion_r"] >= df["AFE_thr_r"],
        "high_diffusion",
        "low_diffusion",
    )

    df["salience_class"] = np.where(
        df["Rank_r"] <= df["AOE_thr_r"],
        "high_salience",
        "low_salience",
    )

    quadrant_values = np.select(
        [
            (
                df["diffusion_class"].eq("high_diffusion")
                & df["salience_class"].eq("high_salience")
            ),
            (
                df["diffusion_class"].eq("high_diffusion")
                & df["salience_class"].eq("low_salience")
            ),
            (
                df["diffusion_class"].eq("low_diffusion")
                & df["salience_class"].eq("high_salience")
            ),
            (
                df["diffusion_class"].eq("low_diffusion")
                & df["salience_class"].eq("low_salience")
            ),
        ],
        [
            "Central nucleus",
            "First periphery",
            "Contrast zone",
            "Peripheral system",
        ],
        default=pd.NA,
    )

    df["quadrant"] = quadrant_values

    if not cfg.human_readable:
        df["quadrant"] = df["quadrant"].map(HUMAN_TO_MACHINE_QUADRANT)

    if cfg.human_readable:
        df["quadrant"] = pd.Categorical(
            df["quadrant"],
            categories=cfg.quadrant_order,
            ordered=True,
        )

    if cfg.concreteness_score_col not in df.columns:
        df[cfg.concreteness_score_col] = np.nan

    if cfg.concreteness_label_col not in df.columns:
        df[cfg.concreteness_label_col] = "-"

    if cfg.concreteness_in_lexicon_col not in df.columns:
        df[cfg.concreteness_in_lexicon_col] = 0

    if cfg.emoji_description_col not in df.columns:
        df[cfg.emoji_description_col] = "-"

    df[cfg.concreteness_label_col] = (
        df[cfg.concreteness_label_col]
        .fillna("-")
        .astype("string")
        .replace("", "-")
    )

    df[cfg.emoji_description_col] = (
        df[cfg.emoji_description_col]
        .fillna("-")
        .astype("string")
        .replace("", "-")
    )

    output_cols = [
        cfg.term_col,
        "upos",
        cfg.term_type_col,
        cfg.n_posts_col,
        cfg.n_users_col,
        cfg.r_pos_col,
        cfg.r_struct_col,
        cfg.salience_col,
        cfg.rank_col,
        "frequency",
        "relative_diffusion",
        "quadrant",
        "diffusion_class",
        "salience_class",
        "AFE_thr",
        "AOE_thr",
        cfg.concreteness_score_col,
        cfg.concreteness_label_col,
        cfg.concreteness_in_lexicon_col,
        cfg.emoji_description_col,
        cfg.user_penetration_col,
        cfg.posts_per_user_col,
    ]

    optional_cols = [
        cfg.n_docs_col,
        cfg.document_penetration_col,
        "diffusion_penetration",
        "first_mention",
        "last_mention",
        "emoji_in_lookup",
        "emoji_match_method",
        "emoji_match_similarity",
    ]

    output_cols.extend(
        col for col in optional_cols
        if col in df.columns and col not in output_cols
    )

    evoc_quadrants_df = df.loc[:, output_cols].copy()

    if cfg.term_col != "term":
        evoc_quadrants_df = evoc_quadrants_df.rename(columns={cfg.term_col: "term"})

    if cfg.term_type_col != "term_type":
        evoc_quadrants_df = evoc_quadrants_df.rename(
            columns={cfg.term_type_col: "term_type"}
        )

    if cfg.n_posts_col != "n_posts":
        evoc_quadrants_df = evoc_quadrants_df.rename(
            columns={cfg.n_posts_col: "n_posts"}
        )

    if cfg.n_users_col != "n_users":
        evoc_quadrants_df = evoc_quadrants_df.rename(
            columns={cfg.n_users_col: "n_users"}
        )

    if cfg.r_pos_col != "R_pos":
        evoc_quadrants_df = evoc_quadrants_df.rename(
            columns={cfg.r_pos_col: "R_pos"}
        )

    if cfg.r_struct_col != "R_struct":
        evoc_quadrants_df = evoc_quadrants_df.rename(
            columns={cfg.r_struct_col: "R_struct"}
        )

    if cfg.salience_col != "S":
        evoc_quadrants_df = evoc_quadrants_df.rename(
            columns={cfg.salience_col: "S"}
        )

    if cfg.rank_col != "Rank":
        evoc_quadrants_df = evoc_quadrants_df.rename(
            columns={cfg.rank_col: "Rank"}
        )

    quadrant_counts_df = (
        evoc_quadrants_df
        .groupby("quadrant", observed=False)
        .size()
        .reset_index(name="n_terms")
    )

    if cfg.human_readable:
        quadrant_counts_df["quadrant"] = pd.Categorical(
            quadrant_counts_df["quadrant"],
            categories=cfg.quadrant_order,
            ordered=True,
        )

    quadrant_counts_df = quadrant_counts_df.sort_values("quadrant").reset_index(
        drop=True
    )

    pos_thresholds_round_df = (
        pos_thresholds_df
        .assign(
            AFE_thr_round=pos_thresholds_df["AFE_thr"].round(cfg.round_digits),
            AOE_thr_round=pos_thresholds_df["AOE_thr"].round(cfg.round_digits),
            mean_absolute_frequency_round=pos_thresholds_df[
                "mean_absolute_frequency"
            ].round(cfg.round_digits),
        )
        [
            [
                "upos",
                "n_terms_pos",
                "AFE_thr_round",
                "AOE_thr_round",
                "mean_absolute_frequency_round",
            ]
        ]
        .reset_index(drop=True)
    )

    html_outputs: dict[str, str] = {}

    if cfg.generate_html:
        html_outputs = generate_evoc_html_reports(
            evoc_quadrants_df=evoc_quadrants_df,
            pos_thresholds_round_df=pos_thresholds_round_df,
            output_dir=cfg.html_output_dir,
            top_n=cfg.html_top_n,
            max_width_px=cfg.html_max_width_px,
            upos_values=sorted(cfg.focal_upos),
        )

    evoc_quadrants_df.attrs["html_outputs"] = html_outputs

    if cfg.verbose:
        print(f"EVOC quadrants assigned; terms: {len(evoc_quadrants_df):,}.")
        print(f"Diffusion basis for quadrant assignment: {cfg.diffusion_basis}.")

        if html_outputs:
            print(f"EVOC HTML reports generated: {len(html_outputs)} file(s).")

    gc.collect()

    return evoc_quadrants_df, quadrant_counts_df, pos_thresholds_round_df


def assign_quadrants(
    term_stats_df: pd.DataFrame,
    *,
    config: QuadrantConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Alias for ``assign_evoc_quadrants``."""

    return assign_evoc_quadrants(
        term_stats_df,
        config=config,
    )


def quadrant_summary_by_pos(
    evoc_quadrants_df: pd.DataFrame,
    *,
    upos_col: str = "upos",
    quadrant_col: str = "quadrant",
) -> pd.DataFrame:
    """Return quadrant counts by UPOS category."""

    _require_columns(evoc_quadrants_df, {upos_col, quadrant_col})

    return (
        evoc_quadrants_df
        .groupby([upos_col, quadrant_col], observed=False)
        .size()
        .reset_index(name="n_terms")
        .sort_values([upos_col, quadrant_col])
        .reset_index(drop=True)
    )
