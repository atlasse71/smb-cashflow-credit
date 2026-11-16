"""Synthetic small-business bank-statement generator.

Each business has a latent "health" score that drives its revenue stability,
expense ratio, loan stacking, overdrafts and, ultimately, default. The model never
sees health directly: it has to recover it from the transactions, as a real
cash-flow underwriter would.

Descriptors are deliberately messy (reference numbers, truncation, generic ACH
strings) so that transaction classification is a real problem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "revenue", "loan_proceeds", "transfer_in", "transfer_out", "lender_payment",
    "payroll", "rent", "operating_expense", "tax", "nsf_fee", "owner_draw", "refund",
]

# industry: (median monthly revenue, seasonal amplitude, peak month, revenue channels)
INDUSTRIES = {
    "restaurant":   (55_000, 0.15, 7,  ["card", "card", "card", "cash"]),
    "retail":       (70_000, 0.30, 12, ["card", "card", "shopify", "cash"]),
    "construction": (120_000, 0.35, 6, ["check", "ach_customer", "ach_customer"]),
    "healthcare":   (90_000, 0.05, 3,  ["insurance", "insurance", "card"]),
    "trucking":     (100_000, 0.10, 10, ["factoring", "ach_customer"]),
    "prof_services": (45_000, 0.08, 4, ["ach_customer", "stripe", "check"]),
}

LENDERS = ["ONDECK", "YELLOWSTONE CAP", "LIBERTAS FUNDING", "PEARL CAPITAL",
           "KAPITUS", "FORA FINANCIAL", "CREDIBLY", "RAPID FINANCE", "BLUEVINE"]
SUPPLIERS = ["SYSCO", "US FOODS", "HOME DEPOT", "LOWES", "GRAINGER", "ULINE",
             "COSTCO WHSE", "AMAZON MKTPLACE", "FASTENAL", "MCKESSON", "PILOT FLYING J",
             "LOVES TRAVEL", "STAPLES", "AT&T", "COMCAST BUSINESS", "DUKE ENERGY"]
CUSTOMERS = ["ACME HOLDINGS", "TURNER CONSTR", "CITY OF DALLAS", "BRIGHT PATH LLC",
             "NORTHSTAR INC", "GREENFIELD CO", "SUMMIT PARTNERS"]


def _ref(rng, n=8):
    return str(rng.integers(10 ** (n - 1), 10 ** n))


def _descriptor(cat: str, ch: str | None, rng, ctx: dict) -> str:
    """Return a realistic, noisy bank descriptor for a transaction."""
    r = rng.random()
    if cat == "revenue":
        if ch == "card":
            return rng.choice([f"SQUARE INC DEP {_ref(rng)}", f"SQ *DEPOSIT {_ref(rng,6)}",
                               f"CLOVER SETTLEMENT {_ref(rng)}", f"BANKCARD DEP {_ref(rng,10)}",
                               f"TOAST INC DEPOSIT {_ref(rng)}"])
        if ch == "shopify":
            return f"SHOPIFY PAYOUT {_ref(rng)}"
        if ch == "stripe":
            return f"STRIPE TRANSFER ST-{_ref(rng,6)}"
        if ch == "cash":
            return rng.choice(["BRANCH DEPOSIT CASH", "COUNTER DEPOSIT", "ATM CASH DEPOSIT"])
        if ch == "check":
            return rng.choice([f"MOBILE DEPOSIT REF {_ref(rng)}", "REMOTE DEPOSIT CAPTURE",
                               f"DEPOSIT ITEM {_ref(rng,5)}"])
        if ch == "insurance":
            return rng.choice([f"HCCLAIMPMT {rng.choice(['BCBS TX','AETNA','UHC','CIGNA'])} {_ref(rng)}",
                               f"MEDICARE PART B {_ref(rng)}"])
        if ch == "factoring":
            return rng.choice([f"RTS FINANCIAL ADV {_ref(rng)}", f"TRIUMPH PAY {_ref(rng)}"])
        # ach from customers; long tail of generic credits
        if r < 0.15:
            return f"WIRE IN {_ref(rng,10)}"  # large customer wire: same string as unnamed loan proceeds
        if r < 0.35:
            return f"ACH CREDIT {_ref(rng,10)} PPD"
        return f"{rng.choice(CUSTOMERS)} ACH PMT {_ref(rng,6)}"
    if cat == "loan_proceeds":
        if r < 0.4:
            return f"WIRE IN {_ref(rng,10)}"  # ambiguous: looks like any wire
        return f"WIRE IN {ctx['lender']} FUNDING {_ref(rng,6)}"
    if cat == "transfer_in":
        return rng.choice([f"ONLINE TRANSFER FROM SAV XXXX{_ref(rng,4)}",
                           f"XFER FROM CHK {_ref(rng,4)}", "INTERNAL TRANSFER CREDIT"])
    if cat == "transfer_out":
        return rng.choice([f"ONLINE TRANSFER TO SAV XXXX{_ref(rng,4)}",
                           f"XFER TO CHK {_ref(rng,4)}", "INTERNAL TRANSFER DEBIT"])
    if cat == "lender_payment":
        name = ctx["lender"]
        if r < 0.3:
            return f"ACH DEBIT {_ref(rng,10)} CCD"  # generic: hard to spot
        if r < 0.5:
            return f"{name[:10]} DEBIT {_ref(rng,6)}"
        return f"ACH DEBIT {name} {_ref(rng,6)}"
    if cat == "payroll":
        return rng.choice([f"GUSTO PAYROLL {_ref(rng)}", f"ADP WAGE PAY {_ref(rng)}",
                           f"PAYCHEX TAX/PAY {_ref(rng)}", f"INTUIT PAYROLL {_ref(rng)}"])
    if cat == "rent":
        return rng.choice([f"ACH RENT {rng.integers(100, 9999)} MAIN ST LLC",
                           f"{rng.choice(['OAK','ELM','CEDAR'])} PROPERTIES LEASE PMT",
                           f"CHECK {rng.integers(1000, 9999)}"])  # rent often paid by check
    if cat == "operating_expense":
        if r < 0.12:
            return f"CHECK {rng.integers(1000, 9999)}"
        if r < 0.20:
            return f"ACH DEBIT {_ref(rng,10)} CCD"  # supplier autopay: same string as unnamed MCA debits
        return f"POS DEBIT {rng.choice(SUPPLIERS)} {_ref(rng,4)}"
    if cat == "tax":
        return rng.choice([f"IRS USATAXPYMT {_ref(rng)}", f"TX COMPTROLLER TAX {_ref(rng,6)}",
                           f"EFTPS FED TAX {_ref(rng)}"])
    if cat == "nsf_fee":
        return rng.choice(["NSF FEE", "OVERDRAFT ITEM FEE", "RETURNED ITEM FEE", "OD FEE"])
    if cat == "owner_draw":
        return rng.choice([f"ZELLE TO {rng.choice(['J SMITH','M GARCIA','A PATEL'])}",
                           f"VENMO CASHOUT {_ref(rng,6)}", "ATM WITHDRAWAL"])
    if cat == "refund":
        return rng.choice([f"SQUARE REFUND {_ref(rng)}", f"CHARGEBACK ADJ {_ref(rng,6)}",
                           f"STRIPE REVERSAL {_ref(rng,6)}"])
    raise ValueError(cat)


def generate(n_businesses: int = 2000, months: int = 6, seed: int = 7):
    """Return (applications, transactions) DataFrames.

    applications: one row per business with application_date and the 12-month
    default outcome. transactions: the bank-statement lines in the 6 months
    before the application, with the true category for evaluation.
    """
    rng = np.random.default_rng(seed)
    app_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n_businesses), "D")
    apps, txns = [], []
    ind_names = list(INDUSTRIES)

    for b in range(n_businesses):
        industry = ind_names[rng.integers(len(ind_names))]
        med_rev, amp, peak, channels = INDUSTRIES[industry]
        health = rng.normal()
        sig = 1 / (1 + np.exp(-health))  # 0..1, higher = healthier
        base_rev = med_rev * np.exp(rng.normal(0, 0.7))
        growth = 0.02 * health + rng.normal(0, 0.03)
        end = app_dates[b] - pd.Timedelta(days=1)
        days = pd.date_range(end - pd.Timedelta(days=30 * months - 1), end, freq="D")
        n_days = len(days)
        month_idx = np.arange(n_days) // 30
        seas = 1 + amp * np.cos(2 * np.pi * (days.month.values - peak) / 12)
        trend = (1 + growth) ** month_idx
        daily_rev_mean = base_rev / 30 * seas * trend
        cv = 0.35 + 0.6 * (1 - sig) + rng.uniform(0, 0.2)
        shock = np.where(rng.random(n_days) < 0.01 * (1 - sig) * 3, rng.uniform(0.0, 0.3), 1.0)

        rows = []  # (date_idx, amount, category, channel, ctx)
        # --- revenue: one batched deposit per business day per channel draw
        for d in range(n_days):
            if days[d].weekday() >= 5 and rng.random() < 0.7:
                continue
            amt = daily_rev_mean[d] * 7 / 5 * shock[d] * rng.gamma(1 / cv ** 2, cv ** 2)
            ch = channels[rng.integers(len(channels))]
            if ch in ("check", "ach_customer", "insurance", "factoring") and rng.random() < 0.6:
                continue  # lumpier channels: fewer, larger deposits
            if ch in ("check", "ach_customer", "insurance", "factoring"):
                amt *= 2.5
            rows.append((d, round(amt, 2), "revenue", ch, {}))
        # --- expenses
        exp_ratio = 0.90 + 0.10 * (1 - sig) + rng.normal(0, 0.03)
        monthly_exp = base_rev * exp_ratio
        for d in range(n_days):
            if d % 14 == 3:
                rows.append((d, -round(monthly_exp * 0.30 / 2.14 * rng.uniform(0.95, 1.05), 2), "payroll", None, {}))
            if d % 30 == 0:
                rows.append((d, -round(monthly_exp * 0.08, 2), "rent", None, {}))
            if days[d].weekday() < 5 and rng.random() < 0.6:
                rows.append((d, -round(monthly_exp * 0.45 / 13 * rng.gamma(2, 0.5), 2), "operating_expense", None, {}))
            if d % 90 == 45:
                rows.append((d, -round(base_rev * 0.06 * rng.uniform(0.5, 1.5), 2), "tax", None, {}))
            if rng.random() < 0.04:
                rows.append((d, -round(base_rev * rng.uniform(0.02, 0.08), 2), "owner_draw", None, {}))
            if rng.random() < 0.02:
                rows.append((d, -round(base_rev * rng.uniform(0.002, 0.01), 2), "refund", None, {}))
        # --- stacked advances (MCA-style daily/weekly debits)
        n_funders = rng.poisson(0.2 + 2.2 * (1 - sig) ** 2)
        burden = 0.0
        for f in range(n_funders):
            lender = LENDERS[rng.integers(len(LENDERS))]
            start = int(rng.integers(0, n_days - 20))
            weekly = rng.random() < 0.4
            share = rng.uniform(0.06, 0.15)
            burden += share
            pay = base_rev * share / (4.3 if weekly else 21.5)
            rows.append((start, round(pay * (30 if weekly else 130) * rng.uniform(0.9, 1.1), 2),
                         "loan_proceeds", None, {"lender": lender}))
            for d in range(start + 1, n_days):
                if days[d].weekday() >= 5:
                    continue
                if weekly and days[d].weekday() != 1:
                    continue
                rows.append((d, -round(pay, 2), "lender_payment", None, {"lender": lender}))
        # --- internal transfers (matched in/out noise)
        for _ in range(rng.poisson(4)):
            d = int(rng.integers(0, n_days))
            amt = round(base_rev * rng.uniform(0.02, 0.2), 2)
            cat = "transfer_in" if rng.random() < 0.5 else "transfer_out"
            rows.append((d, amt if cat == "transfer_in" else -amt, cat, None, {}))

        # --- running balance and NSF fees
        rows.sort(key=lambda r: r[0])
        bal = base_rev * rng.uniform(0.05, 0.6) * sig
        out, balances = [], np.zeros(n_days)
        i = 0
        for d in range(n_days):
            while i < len(rows) and rows[i][0] == d:
                out.append(rows[i]); bal += rows[i][1]; i += 1
            if bal < 0:
                for _ in range(rng.integers(1, 4)):
                    out.append((d, -35.0, "nsf_fee", None, {})); bal -= 35
                if rng.random() < 0.5:  # owner tops up
                    amt = round(-bal + base_rev * rng.uniform(0, 0.1), 2)
                    out.append((d, amt, "transfer_in", None, {})); bal += amt
            balances[d] = bal

        for d, amt, cat, ch, ctx in out:
            txns.append((b, days[d], amt, _descriptor(cat, ch, rng, ctx), cat))

        # --- outcome: driven by latent health + stacking + an unobserved macro drift
        cohort_drift = 0.25 if app_dates[b] >= pd.Timestamp("2025-07-01") else 0.0
        # risk jumps once debt service passes ~20% of revenue, and compounds with thin cash buffers
        over = 1.3 * (burden > 0.20) + 0.8 * (burden > 0.20) * (sig < 0.4)
        logit = -2.7 - 1.0 * health + 0.15 * n_funders + over + cohort_drift + rng.normal(0, 0.4)
        default = int(rng.random() < 1 / (1 + np.exp(-logit)))
        apps.append((b, industry, app_dates[b], default))

    applications = pd.DataFrame(apps, columns=["business_id", "industry", "application_date", "default_12m"])
    transactions = pd.DataFrame(txns, columns=["business_id", "date", "amount", "description", "true_category"])
    return applications, transactions
