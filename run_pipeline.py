"""End-to-end: synthetic statements -> classification -> features -> PD model -> report.

    python run_pipeline.py                 # offline: ML tier + human-review queue
    ANTHROPIC_API_KEY=... python run_pipeline.py   # enables the LLM tier
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, roc_curve

from smbcredit.classify import CREDIT_CRITICAL, HybridClassifier
from smbcredit.features import FEATURES, build_features
from smbcredit.generate import generate
from smbcredit.model import PDModel, REASON_TEXT, approval_at_bad_rate, logistic_benchmark, metrics

OUT = Path("reports"); OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9a9890", "#2b2b29"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def psi(expected, actual, bins=10):
    q = np.quantile(expected, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -np.inf, np.inf
    e = np.histogram(expected, q)[0] / len(expected) + 1e-6
    a = np.histogram(actual, q)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def main(n_businesses: int = 5000, seed: int = 7):
    report = {}
    print(f"1/5 Generating {n_businesses} businesses ...")
    apps, txns = generate(n_businesses, seed=seed)
    report["data"] = {"businesses": len(apps), "transactions": len(txns),
                      "default_rate": round(apps.default_12m.mean(), 4)}

    # ---------------------------------------------------------------- classification
    print("2/5 Training and evaluating the transaction classifier ...")
    rng = np.random.default_rng(seed)
    labeled = rng.choice(apps.business_id, size=int(0.15 * len(apps)), replace=False)  # "human-labeled" sample
    tr = txns[txns.business_id.isin(labeled)]
    te = txns[~txns.business_id.isin(labeled)].copy()
    clf = HybridClassifier().fit(tr)
    pred = clf.predict(te)
    te[["category", "confidence", "tier"]] = pred
    correct = te.category == te.true_category
    by_tier = te.assign(correct=correct).groupby("tier").agg(share=("correct", "size"), accuracy=("correct", "mean"))
    by_tier["share"] = by_tier["share"] / len(te)
    rep = classification_report(te.true_category, te.category, output_dict=True, zero_division=0)
    report["classification"] = {
        "labeled_businesses": len(labeled), "eval_transactions": len(te),
        "overall_accuracy": round(correct.mean(), 4),
        "accuracy_excluding_review_queue": round(correct[te.tier != "review_queue"].mean(), 4),
        "by_tier": by_tier.round(4).to_dict("index"),
        "credit_critical": {c: {"precision": round(rep[c]["precision"], 4), "recall": round(rep[c]["recall"], 4)}
                            for c in sorted(CREDIT_CRITICAL)},
        "llm_enabled": bool(clf.llm and clf.llm.enabled), "llm_calls": clf.llm.calls if clf.llm else 0,
    }
    te[te.tier == "review_queue"].head(200).to_csv(OUT / "review_queue_sample.csv", index=False)

    # ---------------------------------------------------------------- features
    print("3/5 Building cash-flow features ...")
    eval_apps = apps[~apps.business_id.isin(labeled)].set_index("business_id")
    X_pred = build_features(te, "category").reindex(eval_apps.index)
    X_true = build_features(te, "true_category").reindex(eval_apps.index)
    report["feature_fidelity_corr"] = {f: round(float(X_pred[f].corr(X_true[f])), 4) for f in FEATURES}

    # ---------------------------------------------------------------- PD model (out-of-time)
    print("4/5 Training the PD model with out-of-time validation ...")
    d = eval_apps.application_date
    train, cal, test = d < "2025-01-01", (d >= "2025-01-01") & (d < "2025-07-01"), d >= "2025-07-01"
    y = eval_apps.default_12m
    m = PDModel(seed).fit(X_pred[train], y[train], X_pred[cal], y[cal])
    lr = logistic_benchmark(X_pred[train], y[train])
    p_gbm, p_lr = m.predict(X_pred[test]), lr.predict_proba(X_pred[test][list(FEATURES)])[:, 1]
    m_true = PDModel(seed).fit(X_true[train], y[train], X_true[cal], y[cal])
    # value of tier 3: if the low-confidence queue were resolved correctly (LLM or human review)
    te_fixed = te.assign(category=np.where(te.tier == "review_queue", te.true_category, te.category))
    X_fix = build_features(te_fixed, "category").reindex(eval_apps.index)
    m_fix = PDModel(seed).fit(X_fix[train], y[train], X_fix[cal], y[cal])
    target_bad = 0.10  # credit-box appetite: approved population must stay at or under a 10% bad rate
    report["pd_model"] = {
        "split": {"train": int(train.sum()), "calibration": int(cal.sum()), "oot_test": int(test.sum())},
        "lightgbm_oot": metrics(y[test], p_gbm),
        "logistic_oot": metrics(y[test], p_lr),
        "lightgbm_queue_resolved_oot": metrics(y[test], m_fix.predict(X_fix[test])),
        "lightgbm_true_labels_oot": metrics(y[test], m_true.predict(X_true[test])),
        "boosting_rounds": m.booster.best_iteration,
        "target_bad_rate": round(target_bad, 4),
        "approval_rate_at_target": {
            "lightgbm": round(approval_at_bad_rate(y[test], p_gbm, target_bad), 4),
            "logistic": round(approval_at_bad_rate(y[test], p_lr, target_bad), 4),
            "lightgbm_queue_resolved": round(approval_at_bad_rate(y[test], m_fix.predict(X_fix[test]), target_bad), 4),
            "lightgbm_true_labels": round(approval_at_bad_rate(y[test], m_true.predict(X_true[test]), target_bad), 4)},
        "score_psi_train_vs_oot": round(psi(m.predict_raw(X_pred[train]), m.predict_raw(X_pred[test])), 4),
        "calibration_gap_oot": round(float(y[test].mean() - p_gbm.mean()), 4),
    }

    reasons, sv = m.reason_codes(X_pred[test])
    sample = eval_apps[test][["industry"]].assign(pd_12m=p_gbm.round(4), default_12m=y[test]).join(reasons)
    sample.sort_values("pd_12m", ascending=False).head(25).to_csv(OUT / "reason_codes_top25.csv")
    imp = pd.Series(np.abs(sv).mean(0), index=m.features).sort_values()
    report["shap_mean_abs"] = imp.sort_values(ascending=False).round(4).to_dict()

    # ---------------------------------------------------------------- charts
    print("5/5 Writing charts and report ...")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for p, name, c in [(p_gbm, "LightGBM (monotone)", BLUE), (p_lr, "Logistic scorecard", ORANGE)]:
        fpr, tpr, _ = roc_curve(y[test], p)
        ax[0].plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC {metrics(y[test], p)['auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
    ax[0].set(title="ROC: out-of-time test (H2 2025)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend(frameon=False, loc="lower right")
    for p, name, c in [(p_gbm, "LightGBM (monotone)", BLUE), (p_lr, "Logistic scorecard", ORANGE)]:
        fr, mp = calibration_curve(y[test], p, n_bins=8, strategy="quantile")
        ax[1].plot(mp, fr, color=c, lw=2, marker="o", ms=5, label=name)
    lim = max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])
    ax[1].plot([0, lim], [0, lim], color=GRAY, lw=1, ls="--")
    ax[1].set(title="Calibration on OOT: above the diagonal = drift", xlabel="Predicted PD", ylabel="Observed default rate")
    ax[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "model_performance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([REASON_TEXT[f] for f in imp.index], imp.values, color=BLUE, height=0.6)
    ax.set(title="Feature impact on OOT applicants (mean |SHAP|)", xlabel="Mean |SHAP| (log-odds)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "shap_importance.png", dpi=150); plt.close(fig)

    ar = report["pd_model"]["approval_rate_at_target"]
    labels = ["Logistic scorecard", "LightGBM", "LightGBM +\nreview queue resolved", "LightGBM +\nperfect labels"]
    vals = [ar["logistic"], ar["lightgbm"], ar["lightgbm_queue_resolved"], ar["lightgbm_true_labels"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, [v * 100 for v in vals], color=[ORANGE, BLUE, BLUE, BLUE], width=0.55)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v * 100 + 1, f"{v:.1%}", ha="center", color=INK)
    ax.set(title=f"Approval rate on OOT test at a {target_bad:.0%} bad-rate credit box", ylabel="Approval rate (%)",
           ylim=(0, 100))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "approval_at_credit_box.png", dpi=150); plt.close(fig)

    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
