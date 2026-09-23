"""
Project 2: A/B Testing & Portfolio Simulation Framework for Lending Strategies
(for Credeau Data Analyst/Data Scientist role)

Runs a randomized control/treatment experiment comparing the existing rules-only
underwriting strategy (control) against a new ML-hybrid strategy (treatment),
measures statistical significance of the uplift, breaks it down by segment,
and Monte-Carlo simulates the portfolio-level P&L impact of rolling the
treatment out at scale.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

rng = np.random.default_rng(7)
N = 40000
AVG_TICKET = 85000          # INR, average disbursed loan size
YIELD_ON_GOOD_LOAN = 0.18   # annualised net interest margin on a performing loan
LGD = 0.65                  # loss given default (severity)

# ---------------------------------------------------------------
# 1. Simulate applicant pool (same generative structure as Project 1's
#    scorecard, re-fit here so the two projects are independent artifacts)
# ---------------------------------------------------------------
bureau_score = np.clip(rng.normal(700, 90, N), 300, 900).astype(int)
income = np.round(rng.lognormal(mean=10.8, sigma=0.45, size=N), -2)
dti = np.clip(rng.normal(35, 15, N), 2, 95)
bounced_txns_6m = rng.poisson(np.clip(0.6 + (700 - bureau_score) / 300, 0, None))
vintage_months = rng.integers(0, 180, N)
avg_bank_balance = np.round(np.clip(rng.lognormal(9.2, 0.6, N), 500, None), -2)
inquiries_3m = rng.poisson(1.3, N)
existing_loans = rng.poisson(1.1, N)
employment_type = rng.choice(["Salaried", "Self-Employed", "Gig"], N, p=[0.55, 0.30, 0.15])
region = rng.choice(["Metro", "Tier-2", "Tier-3"], N, p=[0.45, 0.35, 0.20])

df = pd.DataFrame(dict(
    bureau_score=bureau_score, income=income, dti=dti, bounced_txns_6m=bounced_txns_6m,
    vintage_months=vintage_months, avg_bank_balance=avg_bank_balance,
    inquiries_3m=inquiries_3m, existing_loans=existing_loans,
    employment_type=employment_type, region=region,
))

emp_risk = df.employment_type.map({"Salaried": -0.15, "Self-Employed": 0.05, "Gig": 0.30}).values
logit = (
    -3.4
    - 0.017 * (df.bureau_score - 700)
    + 0.028 * (df.dti - 35)
    + 0.22 * df.bounced_txns_6m
    - 0.16 * df.existing_loans.clip(upper=4)
    - 0.004 * (df.vintage_months / 12)
    + 0.20 * df.inquiries_3m
    - 0.35 * np.log(df.avg_bank_balance / df.income.clip(lower=1))
    + emp_risk
    + rng.normal(0, 0.55, N)
)
p_default = 1 / (1 + np.exp(-logit))
df["default"] = rng.binomial(1, p_default)

# ---------------------------------------------------------------
# 2. Train the ML scorecard used by the "treatment" strategy
# ---------------------------------------------------------------
model_df = pd.get_dummies(df, columns=["employment_type", "region"], drop_first=True)
feature_cols = [c for c in model_df.columns if c != "default"]
X_tr, X_te, y_tr, y_te = train_test_split(
    model_df[feature_cols], model_df["default"], test_size=0.4, random_state=1, stratify=model_df["default"]
)
xgb = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=1)
xgb.fit(X_tr, y_tr)

# Score the experiment population (X_te) — this is our A/B test cohort
exp = df.loc[X_te.index].copy()
exp["ml_score"] = xgb.predict_proba(X_te)[:, 1]

def rule_decision(row):
    if row.bureau_score < 650 or row.dti > 60 or row.bounced_txns_6m >= 4:
        return "Reject"
    if row.bureau_score < 700 or row.dti > 45 or row.bounced_txns_6m >= 2:
        return "Refer"
    return "Approve"

def hybrid_decision(row):
    rd = rule_decision(row)
    if rd == "Reject":
        return "Reject"
    if row.ml_score >= 0.35:
        return "Reject"
    if row.ml_score >= 0.15:
        return "Refer"
    return "Approve"

exp["rule_decision"] = exp.apply(rule_decision, axis=1)
exp["hybrid_decision"] = exp.apply(hybrid_decision, axis=1)

# ---------------------------------------------------------------
# 3. Randomized A/B split: 50% control (rules-only), 50% treatment (hybrid)
#    Each applicant is scored under BOTH strategies; the experiment applies
#    whichever strategy the applicant was randomly assigned to.
# ---------------------------------------------------------------
exp["arm"] = rng.choice(["control", "treatment"], size=len(exp), p=[0.5, 0.5])
exp["decision"] = np.where(exp.arm == "control", exp.rule_decision, exp.hybrid_decision)
exp["approved"] = exp.decision == "Approve"

control = exp[exp.arm == "control"]
treatment = exp[exp.arm == "treatment"]

def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return p1, p2, z, p_val

# --- Approval-rate uplift ---
c_appr, t_appr = control.approved.sum(), treatment.approved.sum()
p1, p2, z, pval = two_prop_ztest(c_appr, len(control), t_appr, len(treatment))
print("=== Overall Experiment Results (n = {:,}) ===".format(len(exp)))
print(f"Control approval rate  : {p1:.2%}  (n={len(control):,})")
print(f"Treatment approval rate: {p2:.2%}  (n={len(treatment):,})")
print(f"Approval-rate uplift    : {(p2-p1)*100:+.2f} pp   z={z:.2f}, p-value={pval:.2e}")

# --- NPA-rate comparison among approved loans ---
c_npa = control.loc[control.approved, "default"]
t_npa = treatment.loc[treatment.approved, "default"]
p1n, p2n, zn, pvaln = two_prop_ztest(c_npa.sum(), len(c_npa), t_npa.sum(), len(t_npa))
print(f"\nControl NPA rate (approved)  : {p1n:.2%} (n={len(c_npa):,})")
print(f"Treatment NPA rate (approved): {p2n:.2%} (n={len(t_npa):,})")
print(f"NPA-rate delta               : {(p2n-p1n)*100:+.2f} pp   z={zn:.2f}, p-value={pvaln:.2e}")

# ---------------------------------------------------------------
# 4. Segment-level (heterogeneous treatment effect) analysis
# ---------------------------------------------------------------
print("\n=== Segment-level approval-rate uplift (treatment - control) ===")
for seg_col in ["region", "employment_type"]:
    print(f"\nBy {seg_col}:")
    for seg_val, seg_df in exp.groupby(seg_col):
        c = seg_df[seg_df.arm == "control"]
        t = seg_df[seg_df.arm == "treatment"]
        if len(c) == 0 or len(t) == 0:
            continue
        uplift = t.approved.mean() - c.approved.mean()
        print(f"  {seg_val:<14} control={c.approved.mean():.2%}  treatment={t.approved.mean():.2%}  uplift={uplift*100:+.2f} pp  (n_c={len(c)}, n_t={len(t)})")

bureau_bins = pd.cut(exp.bureau_score, bins=[300, 650, 700, 750, 800, 900],
                     labels=["<650", "650-700", "700-750", "750-800", "800+"])
exp["bureau_band"] = bureau_bins
print("\nBy bureau_band:")
for seg_val, seg_df in exp.groupby("bureau_band", observed=True):
    c = seg_df[seg_df.arm == "control"]
    t = seg_df[seg_df.arm == "treatment"]
    if len(c) == 0 or len(t) == 0:
        continue
    uplift = t.approved.mean() - c.approved.mean()
    print(f"  {seg_val:<10} control={c.approved.mean():.2%}  treatment={t.approved.mean():.2%}  uplift={uplift*100:+.2f} pp  (n_c={len(c)}, n_t={len(t)})")

# ---------------------------------------------------------------
# 5. Monte Carlo simulation: portfolio-level P&L impact of rolling out
#    the treatment strategy across a monthly applicant volume
# ---------------------------------------------------------------
MONTHLY_APPLICANTS = 50000
N_SIMS = 2000

def simulate_portfolio(decisions_pool, defaults_pool, n_apps, n_sims):
    idx_pool = np.arange(len(decisions_pool))
    net_pnls = np.empty(n_sims)
    approvals = np.empty(n_sims)
    for i in range(n_sims):
        sample_idx = rng.choice(idx_pool, size=n_apps, replace=True)
        dec = decisions_pool[sample_idx]
        dft = defaults_pool[sample_idx]
        approved_mask = dec == "Approve"
        n_appr = approved_mask.sum()
        n_default = dft[approved_mask].sum()
        n_good = n_appr - n_default
        revenue = n_good * AVG_TICKET * YIELD_ON_GOOD_LOAN
        losses = n_default * AVG_TICKET * LGD
        net_pnls[i] = revenue - losses
        approvals[i] = n_appr
    return net_pnls, approvals

control_pnl, control_appr = simulate_portfolio(
    control.rule_decision.values, control.default.values, MONTHLY_APPLICANTS, N_SIMS
)
treat_pnl, treat_appr = simulate_portfolio(
    treatment.hybrid_decision.values, treatment.default.values, MONTHLY_APPLICANTS, N_SIMS
)

print("\n=== Monte Carlo Portfolio Simulation ({} applicants/month, {} sims) ===".format(MONTHLY_APPLICANTS, N_SIMS))
print(f"Control  : mean approvals={control_appr.mean():,.0f}, mean monthly net P&L=INR {control_pnl.mean():,.0f} "
      f"(90% CI: {np.percentile(control_pnl,5):,.0f} to {np.percentile(control_pnl,95):,.0f})")
print(f"Treatment: mean approvals={treat_appr.mean():,.0f}, mean monthly net P&L=INR {treat_pnl.mean():,.0f} "
      f"(90% CI: {np.percentile(treat_pnl,5):,.0f} to {np.percentile(treat_pnl,95):,.0f})")
pnl_uplift = treat_pnl.mean() - control_pnl.mean()
prob_positive = (treat_pnl > control_pnl).mean()
print(f"Expected monthly P&L uplift from rollout: INR {pnl_uplift:,.0f}  "
      f"(P[treatment beats control] = {prob_positive:.1%})")
