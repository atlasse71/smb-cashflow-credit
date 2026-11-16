"""PD model: monotone-constrained LightGBM vs. a logistic scorecard benchmark.

- Out-of-time validation: train on earlier applications, test on later ones.
- Isotonic calibration on a held-out slice of the training window.
- SHAP values converted to per-applicant reason codes (adverse-action style).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .features import FEATURES

REASON_TEXT = {
    "avg_monthly_revenue_log": "Low average monthly revenue",
    "revenue_cv": "Volatile monthly revenue",
    "revenue_trend": "Declining revenue trend",
    "min_month_rev_to_avg": "Weak lowest-revenue month",
    "expense_to_revenue": "High operating expenses relative to revenue",
    "avg_daily_balance_to_rev": "Low average daily balance relative to revenue",
    "days_negative": "Days with negative balance",
    "nsf_count": "Non-sufficient-funds events",
    "nsf_last_30d": "Recent non-sufficient-funds events",
    "active_funders": "Number of active financing obligations",
    "lender_pmt_to_revenue": "Existing debt payments relative to revenue",
    "new_funding_90d": "Recent new financing",
    "owner_draw_to_revenue": "Owner withdrawals relative to revenue",
    "deposit_days_share": "Infrequent deposits",
    "top_channel_share": "Revenue concentrated in one source",
}


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p):
    return {"auc": round(roc_auc_score(y, p), 4), "gini": round(2 * roc_auc_score(y, p) - 1, 4),
            "ks": round(ks_stat(y, p), 4), "brier": round(brier_score_loss(y, p), 4),
            "mean_pd": round(float(np.mean(p)), 4), "bad_rate": round(float(np.mean(y)), 4)}


def approval_at_bad_rate(y, p, target_bad_rate):
    """Highest approval rate whose approved population stays at or under target bad rate."""
    order = np.argsort(p)
    cum_bad = np.cumsum(np.asarray(y)[order]) / np.arange(1, len(y) + 1)
    ok = np.where(cum_bad <= target_bad_rate)[0]
    return float((ok.max() + 1) / len(y)) if len(ok) else 0.0


class PDModel:
    def __init__(self, seed: int = 7):
        self.features = list(FEATURES)
        self.params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           monotone_constraints=[FEATURES[f] for f in self.features],
                           monotone_constraints_method="advanced", verbose=-1, seed=seed)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_cal: pd.DataFrame, y_cal: pd.Series):
        dtr = lgb.Dataset(X[self.features], y)
        dval = lgb.Dataset(X_cal[self.features], y_cal)
        self.booster = lgb.train(self.params, dtr, num_boost_round=2000, valid_sets=[dval],
                                 callbacks=[lgb.early_stopping(100, verbose=False)])
        raw_cal = self.booster.predict(X_cal[self.features])
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(raw_cal, y_cal)
        self.explainer = shap.TreeExplainer(self.booster)
        return self

    def predict_raw(self, X):
        return self.booster.predict(X[self.features])

    def predict(self, X):
        return self.iso.predict(self.predict_raw(X))

    def reason_codes(self, X: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        sv = self.explainer.shap_values(X[self.features])
        sv = sv[1] if isinstance(sv, list) else sv
        rows = []
        for i in range(len(X)):
            top = np.argsort(-sv[i])[:k]
            rows.append([REASON_TEXT[self.features[j]] if sv[i, j] > 0 else "" for j in top])
        return pd.DataFrame(rows, index=X.index, columns=[f"reason_{n+1}" for n in range(k)]), sv


def logistic_benchmark(X, y):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000, C=0.5)).fit(X[list(FEATURES)], y)
