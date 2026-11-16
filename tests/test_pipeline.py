import numpy as np

from smbcredit.classify import HybridClassifier, apply_rules, normalize
from smbcredit.features import FEATURES, build_features
from smbcredit.generate import generate


def test_normalize_strips_reference_numbers():
    assert normalize("SQUARE INC DEP 12345678") == normalize("SQUARE INC DEP 99887766") == "SQUARE INC DEP"


def test_rules_respect_sign():
    import pandas as pd
    norm = pd.Series(["GUSTO PAYROLL", "GUSTO PAYROLL"])
    out = apply_rules(norm, pd.Series([-100.0, 100.0]))
    assert out[0] == "payroll" and out.isna()[1]


def test_end_to_end_small():
    apps, txns = generate(120, seed=1)
    assert (txns.groupby("business_id").date.max() < apps.set_index("business_id").application_date).all()
    clf = HybridClassifier(use_llm=False).fit(txns[txns.business_id < 60])
    te = txns[txns.business_id >= 60].copy()
    te[["category", "confidence", "tier"]] = clf.predict(te)
    assert (te.category == te.true_category).mean() > 0.9
    X = build_features(te)
    assert list(X.columns) == list(FEATURES) and len(X) == 60
    assert np.isfinite(X.drop(columns=["revenue_cv", "min_month_rev_to_avg"]).to_numpy(float)).all()
