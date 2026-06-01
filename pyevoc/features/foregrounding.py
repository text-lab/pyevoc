"""Token-level structural foregrounding indicators."""
from __future__ import annotations

from dataclasses import dataclass
import re
import pandas as pd


@dataclass(frozen=True)
class ForegroundingWeights:
    opening_sentence: float = 0.35
    emphasis: float = 0.25
    list_or_quote: float = 0.20
    intensification: float = 0.20

    def normalised(self) -> "ForegroundingWeights":
        total = self.opening_sentence + self.emphasis + self.list_or_quote + self.intensification
        if total <= 0:
            raise ValueError("Foregrounding weights must sum to a positive value.")
        return ForegroundingWeights(
            self.opening_sentence / total,
            self.emphasis / total,
            self.list_or_quote / total,
            self.intensification / total,
        )


def add_foregrounding_indicators(
    tokens: pd.DataFrame,
    *,
    sentence_col: str = "sentence_id",
    token_col: str = "text",
) -> pd.DataFrame:
    out = tokens.copy()
    text = out[token_col].fillna("").astype(str)
    out["fg_opening_sentence"] = (out[sentence_col] == 1).astype(int)
    out["fg_emphasis"] = text.str.contains(r"\*\*|__|<b>|</b>", regex=True).astype(int)
    out["fg_list_or_quote"] = text.str.match(r"^[-*>•]").astype(int)
    out["fg_intensification"] = text.apply(lambda x: int(bool(re.search(r"[A-Z]{3,}|!{2,}|\?{2,}", x))))
    return out


def compute_structural_salience(tokens: pd.DataFrame, weights: ForegroundingWeights = ForegroundingWeights()) -> pd.DataFrame:
    w = weights.normalised()
    out = add_foregrounding_indicators(tokens) if "fg_opening_sentence" not in tokens.columns else tokens.copy()
    out["r_str"] = (
        w.opening_sentence * out["fg_opening_sentence"]
        + w.emphasis * out["fg_emphasis"]
        + w.list_or_quote * out["fg_list_or_quote"]
        + w.intensification * out["fg_intensification"]
    )
    return out


def add_positional_salience(tokens: pd.DataFrame, doc_col: str = "doc_id", position_col: str = "position") -> pd.DataFrame:
    out = tokens.copy()
    if position_col not in out.columns:
        out[position_col] = out.groupby(doc_col).cumcount() + 1
    lengths = out.groupby(doc_col)[position_col].transform("max")
    out["r_pos"] = 1.0
    mask = lengths > 1
    out.loc[mask, "r_pos"] = 1 - ((out.loc[mask, position_col] - 1) / (lengths[mask] - 1))
    return out
