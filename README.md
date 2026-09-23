# ab-testing-monte-carlo-simulation
# A/B Testing & Monte Carlo Portfolio Simulation Framework for Lending Strategies

An experimentation and simulation framework for evaluating a new ML-based
underwriting strategy against an existing rules-only strategy — combining
statistical hypothesis testing, segment-level uplift analysis, and Monte
Carlo portfolio simulation to quantify both statistical and business impact
before a full rollout.

## Overview

Before any lending strategy change is deployed at scale, a risk team needs to
know: **is the uplift real (not noise), where does it come from, and what
does it mean for the P&L?** This project answers all three by simulating a
randomized control/treatment experiment on a loan-applicant population and
layering statistical testing, segmentation, and Monte Carlo simulation on
top of the results.

## Problem Statement

Rolling out a new decisioning strategy (e.g. an ML-hybrid scorecard) without
rigorous testing risks either under-approving (leaving growth on the table)
or over-approving (increasing losses) at scale. This project simulates an
A/B test comparing:
- **Control:** existing rules-only underwriting strategy
- **Treatment:** new ML-hybrid decision strategy

and quantifies the uplift with statistical rigor before projecting its
financial impact at production volume.

## Data

A synthetic pool of 40,000 applicants was generated (same generative logic
as the credit-risk-engine project, re-fit independently), covering bureau,
financial, transactional, employment, and regional attributes, with a
simulated ground-truth default outcome.

## Methodology

1. **Model training** — trained an XGBoost scorecard on a held-out training split; the remaining applicants form the experiment cohort.
2. **Randomized assignment** — each applicant in the experiment cohort is randomly assigned to `control` (rules-only decision) or `treatment` (ML-hybrid decision) with 50/50 probability.
3. **Significance testing** — used a two-proportion z-test to test whether the treatment's approval rate and NPA rate differ significantly from control.
4. **Segment-level (heterogeneous treatment effect) analysis** — broke the uplift down by region, employment type, and bureau-score band to find where the strategy helps most/least.
5. **Monte Carlo portfolio simulation** — resampled the experiment population 2,000 times at a simulated monthly volume of 50,000 applicants, estimating the distribution of net portfolio P&L (revenue from performing loans minus losses from defaults) under each strategy.

## Results

### Overall experiment (n = 16,000)

| Metric | Control | Treatment | Delta | Significance |
|---|---|---|---|---|
| Approval rate | 34.9% | 61.6% | **+26.6 pp** | z = 33.7, p < 0.001 |
| NPA rate (approved) | 2.9% | 4.8% | +1.9 pp | z = 3.98, p < 0.001 |

### Segment-level uplift (selected)

| Segment | Uplift |
|---|---|
| Bureau band 650–700 | **+72.3 pp** (largest — a previously thin-approval segment) |
| Bureau band 700–750 | +25.2 pp |
| Salaried | +27.7 pp |
| Gig | +20.8 pp (smallest — highest-risk segment, more conservatively treated) |

### Monte Carlo portfolio simulation (50,000 applicants/month, 2,000 runs)

| Strategy | Mean Approvals | Mean Monthly Net P&L | 90% CI |
|---|---|---|---|
| Control | 17,463 | ₹23.1 Cr | ₹22.8–23.5 Cr |
| Treatment | 30,789 | ₹36.7 Cr | ₹36.2–37.2 Cr |

**Expected monthly P&L uplift: ~₹13.6 Cr**, with a **100% probability** of the treatment strategy outperforming control across all 2,000 simulated months.

## Tech Stack

Python · Pandas · NumPy · SciPy · Scikit-learn · XGBoost

## How to Run

```bash
pip install pandas numpy scipy scikit-learn xgboost
python project2_ab_testing_simulation.py
```

## Future Improvements

- Add sequential testing / early-stopping rules to reduce experiment duration
- Model yield and LGD (loss given default) assumptions as distributions rather than fixed constants, for a fuller Monte Carlo picture
- Extend segmentation to interaction effects (e.g. region × bureau band) using a proper CATE/uplift-modeling approach (e.g. causal forests)
- Add a cost-of-capital-adjusted ROI metric alongside raw P&L
