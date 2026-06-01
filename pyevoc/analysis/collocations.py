"""Dependency-informed collocation extraction."""
from __future__ import annotations
from collections import Counter
import math
import pandas as pd


def log_likelihood_g2(k11, k12, k21, k22) -> float:
    def xlogx(x):
        return 0 if x <= 0 else x * math.log(x)
    row1, row2 = k11 + k12, k21 + k22
    col1, col2 = k11 + k21, k12 + k22
    total = row1 + row2
    expected = [row1*col1/total, row1*col2/total, row2*col1/total, row2*col2/total]
    observed = [k11, k12, k21, k22]
    return 2 * sum(xlogx(o) - xlogx(e) for o, e in zip(observed, expected))


def extract_collocations(tokens: pd.DataFrame, term_col: str = "term", doc_col: str = "doc_id", user_col: str = "user_id", window: int = 1, min_freq: int = 5) -> pd.DataFrame:
    pairs = Counter()
    pair_docs: dict[tuple[str, str], set] = {}
    pair_users: dict[tuple[str, str], set] = {}
    term_freq = Counter(tokens[term_col].dropna().astype(str))
    rows = tokens[[doc_col, user_col, term_col]].dropna()
    for doc_id, grp in rows.groupby(doc_col):
        vals = grp[term_col].astype(str).tolist()
        user = grp[user_col].iloc[0]
        for i in range(len(vals)-window):
            a, b = vals[i], vals[i+window]
            if a == b:
                continue
            pair = tuple(sorted((a, b)))
            pairs[pair] += 1
            pair_docs.setdefault(pair, set()).add(doc_id)
            pair_users.setdefault(pair, set()).add(user)
    total = sum(term_freq.values())
    out = []
    for (a, b), freq in pairs.items():
        if freq < min_freq:
            continue
        k11 = freq
        k12 = term_freq[a] - freq
        k21 = term_freq[b] - freq
        k22 = max(total - k11 - k12 - k21, 0)
        out.append({"term": f"{a} {b}", "term_a": a, "term_b": b, "freq": freq, "posts": len(pair_docs[(a,b)]), "users": len(pair_users[(a,b)]), "G2": log_likelihood_g2(k11,k12,k21,k22)})
    return pd.DataFrame(out).sort_values("G2", ascending=False).reset_index(drop=True) if out else pd.DataFrame()
