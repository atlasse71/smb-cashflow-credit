"""Cash-flow underwriting features from classified bank transactions.

All features use only transactions dated before the application date (the
generator guarantees this; in production, enforce it to avoid leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature -> expected direction of default risk, used for LightGBM monotone constraints.
# +1: higher value => higher risk; -1: higher value => lower risk; 0: unconstrained.
FEATURES = {
    "avg_monthly_revenue_log": -1,
    "revenue_cv": +1,
    "revenue_trend": -1,
    "min_month_rev_to_avg": -1,
    "expense_to_revenue": +1,
    "avg_daily_balance_to_rev": -1,
    "days_negative": +1,
    "nsf_count": +1,
    "nsf_last_30d": +1,
    "active_funders": +1,
    "lender_pmt_to_revenue": +1,
    "new_funding_90d": +1,
    "owner_draw_to_revenue": 0,
    "deposit_days_share": -1,
    "top_channel_share": 0,
}


def build_features(txns: pd.DataFrame, category_col: str = "category") -> pd.DataFrame:
    t = txns.copy()
    t["month"] = t.groupby("business_id")["date"].transform(lambda d: ((d - d.min()).dt.days // 30).clip(upper=5))
    end = t.groupby("business_id")["date"].transform("max")
    cat = t[category_col]
    rev = t[cat == "revenue"]

    g = rev.groupby(["business_id", "month"])["amount"].sum().unstack(fill_value=0)
    avg_rev = g.mean(1)
    f = pd.DataFrame(index=t["business_id"].unique()).sort_index()
    f["avg_monthly_revenue_log"] = np.log1p(avg_rev)
    f["revenue_cv"] = g.std(1) / avg_rev.replace(0, np.nan)
    x = np.arange(g.shape[1])
    f["revenue_trend"] = g.apply(lambda r: np.polyfit(x, r / max(r.mean(), 1), 1)[0], axis=1)
    f["min_month_rev_to_avg"] = g.min(1) / avg_rev.replace(0, np.nan)

    out_ops = t[cat.isin(["payroll", "rent", "operating_expense", "tax"])].groupby("business_id")["amount"].sum().abs()
    total_rev = rev.groupby("business_id")["amount"].sum()
    f["expense_to_revenue"] = out_ops / total_rev

    # daily balances from cumulative flows (relative: opening balance is unknown)
    daily = t.groupby(["business_id", "date"])["amount"].sum().groupby(level=0).cumsum()
    lvl = daily.groupby(level=0).transform(lambda s: s - s.min())  # shift so minimum is zero
    f["avg_daily_balance_to_rev"] = lvl.groupby(level=0).mean() / avg_rev
    nsf = t[cat == "nsf_fee"]
    f["days_negative"] = nsf.groupby("business_id")["date"].nunique()
    f["nsf_count"] = nsf.groupby("business_id").size()
    f["nsf_last_30d"] = nsf[nsf["date"] > end[nsf.index] - pd.Timedelta(days=30)].groupby("business_id").size()

    lp = t[cat == "lender_payment"]
    # active funders: distinct recurring payment streams (pattern + amount) in the last 30 days
    recent = lp[lp["date"] > end[lp.index] - pd.Timedelta(days=30)]
    f["active_funders"] = recent.assign(a=recent["amount"].round(0)).groupby("business_id")["a"].nunique()
    f["lender_pmt_to_revenue"] = lp.groupby("business_id")["amount"].sum().abs() / total_rev
    proc = t[cat == "loan_proceeds"]
    f["new_funding_90d"] = proc[proc["date"] > end[proc.index] - pd.Timedelta(days=90)].groupby("business_id").size()
    f["owner_draw_to_revenue"] = t[cat == "owner_draw"].groupby("business_id")["amount"].sum().abs() / total_rev
    span = t.groupby("business_id")["date"].agg(lambda d: (d.max() - d.min()).days + 1)
    f["deposit_days_share"] = rev.groupby("business_id")["date"].nunique() / span
    # concentration: share of revenue from the single largest descriptor pattern
    key = rev["description"].str.replace(r"\d+", "", regex=True).str.strip()
    by_src = rev.assign(k=key).groupby(["business_id", "k"])["amount"].sum()
    f["top_channel_share"] = by_src.groupby(level=0).max() / total_rev

    count_cols = ["days_negative", "nsf_count", "nsf_last_30d", "active_funders", "new_funding_90d"]
    f[count_cols] = f[count_cols].fillna(0)
    f = f.fillna({"lender_pmt_to_revenue": 0, "owner_draw_to_revenue": 0})
    f.index.name = "business_id"
    return f[list(FEATURES)]
