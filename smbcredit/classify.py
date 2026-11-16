"""Three-tier transaction classifier.

Tier 1 - rules: high-precision regexes for unambiguous descriptors (payroll vendors,
         NSF fees, known lenders). Cheap and auditable.
Tier 2 - ML:    character n-gram TF-IDF + amount features -> logistic regression,
         trained on a human-labeled sample. Handles most of the rest.
Tier 3 - LLM:   low-confidence lines go to an LLM with amount and account context,
         returning structured JSON. Results are cached by normalized descriptor,
         so each merchant string is paid for once. If no API key is set, these
         lines are routed to a human-review queue instead.

Credit-critical categories (lender_payment, loan_proceeds, transfers, revenue) get a
stricter confidence threshold because errors there distort debt-capacity features.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .generate import CATEGORIES, LENDERS

CREDIT_CRITICAL = {"lender_payment", "loan_proceeds", "transfer_in", "transfer_out", "revenue"}


def normalize(desc: str) -> str:
    """Strip reference numbers and masks so 'SQUARE INC DEP 12345' == 'SQUARE INC DEP 99881'."""
    s = desc.upper()
    s = re.sub(r"X{2,}\d*", " ", s)
    s = re.sub(r"\b[A-Z]{0,3}-?\d{3,}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_LENDER_RE = "|".join(re.escape(l[:10]) for l in LENDERS)
RULES = [
    (r"^(NSF|OVERDRAFT ITEM|RETURNED ITEM|OD) FEE", None, "nsf_fee"),
    (r"GUSTO|ADP WAGE|PAYCHEX|INTUIT PAYROLL", -1, "payroll"),
    (r"IRS USATAXPYMT|EFTPS|COMPTROLLER TAX", -1, "tax"),
    (rf"({_LENDER_RE}).*DEBIT|ACH DEBIT ({_LENDER_RE})", -1, "lender_payment"),
    (rf"WIRE IN ({_LENDER_RE})", 1, "loan_proceeds"),
    (r"TRANSFER FROM|XFER FROM|INTERNAL TRANSFER CREDIT", 1, "transfer_in"),
    (r"TRANSFER TO|XFER TO|INTERNAL TRANSFER DEBIT", -1, "transfer_out"),
    (r"REFUND|CHARGEBACK|REVERSAL", -1, "refund"),
]


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Recurrence context within each business's statement: a generic 'ACH DEBIT ... CCD'
    that repeats with an identical amount every business day is almost certainly an MCA."""
    df = df.assign(norm=df["description"].map(normalize))
    key = df["norm"].str.replace(r"\d+", "", regex=True)
    grp = df.assign(key=key, amt=df["amount"].round(0)).groupby(["business_id", "key"])["amt"]
    return df.assign(n_seen=grp.transform("size"), same_amt=(grp.transform("nunique") == 1) & (grp.transform("size") > 3))


def apply_rules(norm: pd.Series, amount: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=norm.index, dtype="object")
    sign = np.sign(amount)
    for pattern, req_sign, cat in RULES:
        hit = norm.str.contains(pattern, regex=True) & out.isna()
        if req_sign is not None:
            hit &= sign == req_sign
        out[hit] = cat
    return out


class MLClassifier:
    def __init__(self):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=2000, C=4.0)

    @staticmethod
    def _num(df):
        a = df["amount"].to_numpy()
        return csr_matrix(np.c_[np.sign(a), np.log1p(np.abs(a)) / 10, (np.abs(a) % 1 == 0).astype(float),
                                np.log1p(df["n_seen"].to_numpy()) / 5, df["same_amt"].to_numpy(float)])

    def fit(self, df: pd.DataFrame, y: pd.Series):
        X = hstack([self.vec.fit_transform(df["norm"]), self._num(df)]).tocsr()
        self.clf.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame):
        X = hstack([self.vec.transform(df["norm"]), self._num(df)]).tocsr()
        p = self.clf.predict_proba(X)
        return self.clf.classes_[p.argmax(1)], p.max(1)


LLM_PROMPT = """You classify small-business bank transactions for credit underwriting.
Categories: {cats}.
Rules: lender_payment = recurring debits to MCA/loan funders (often daily or weekly,
identical amounts); loan_proceeds = a large one-off credit from a funder; revenue =
customer/processor deposits; transfers are between the owner's own accounts.

Transaction: "{desc}"  amount: {amount:+.2f}
Same descriptor pattern seen {n_seen} times in this statement; amounts identical: {same_amt}.

Reply with JSON only: {{"category": "<one category>", "confidence": <0-1>}}"""


class LLMClassifier:
    """Anthropic-backed tier. Active only when ANTHROPIC_API_KEY is set."""

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model
        self.cache: dict[str, tuple[str, float]] = {}
        self.calls = 0
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.enabled:
            import anthropic
            self.client = anthropic.Anthropic()

    def classify(self, desc: str, norm: str, amount: float, n_seen: int, same_amt: bool):
        key = f"{norm}|{np.sign(amount)}|{same_amt}"
        if key in self.cache:
            return self.cache[key]
        msg = self.client.messages.create(
            model=self.model, max_tokens=60,
            messages=[{"role": "user", "content": LLM_PROMPT.format(
                cats=", ".join(CATEGORIES), desc=desc, amount=amount, n_seen=n_seen, same_amt=same_amt)}],
        )
        self.calls += 1
        try:
            out = json.loads(msg.content[0].text)
            res = (out["category"] if out["category"] in CATEGORIES else "operating_expense",
                   float(out["confidence"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            res = ("operating_expense", 0.0)
        self.cache[key] = res
        return res


class HybridClassifier:
    def __init__(self, threshold: float = 0.80, critical_threshold: float = 0.90, use_llm: bool = True):
        self.ml = MLClassifier()
        self.llm = LLMClassifier() if use_llm else None
        self.threshold, self.critical_threshold = threshold, critical_threshold

    def fit(self, df: pd.DataFrame):
        df = add_context(df)
        self.ml.fit(df, df["true_category"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_context(df)
        rule = apply_rules(df["norm"], df["amount"])
        ml_cat, ml_conf = self.ml.predict_proba(df)
        out = pd.DataFrame({"category": rule, "confidence": 1.0, "tier": "rules"}, index=df.index)
        m = out["category"].isna()
        out.loc[m, "category"] = ml_cat[m.to_numpy()]
        out.loc[m, "confidence"] = ml_conf[m.to_numpy()]
        out.loc[m, "tier"] = "ml"
        thr = np.where(out["category"].isin(CREDIT_CRITICAL), self.critical_threshold, self.threshold)
        low = m & (out["confidence"] < thr)
        if low.any():
            if self.llm is not None and self.llm.enabled:
                for i in df.index[low]:
                    cat, conf = self.llm.classify(df.at[i, "description"], df.at[i, "norm"], df.at[i, "amount"],
                                                  int(df.at[i, "n_seen"]), bool(df.at[i, "same_amt"]))
                    out.loc[i, ["category", "confidence", "tier"]] = [cat, conf, "llm"]
            else:
                out.loc[low, "tier"] = "review_queue"  # ML label kept, flagged for a human
        return out
