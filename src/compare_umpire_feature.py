"""
One-factor test: does adding umpire strikeout tendency improve the K model?

Builds two feature sets -- baseline (current FEATURE_COLS) and +umpire (baseline plus
umpire_k_index, an as-of-date league-relative index of the home-plate umpire's
game-total-strikeout tendency, see rolling.build_umpire_k_form) -- trains an NB2 model
on each with the identical train/holdout split and methodology as model_ks.py, and
compares CRPS + Brier@4.5/5.5/6.5 head to head on the 2025 holdout.

Deliberately a standalone script, not a change to model_ks.py itself: if the feature
doesn't help, nothing about the shipped model needs to change, and this file (plus
data/processed/model_features_ks_umpire.parquet) can just be deleted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import features_ks
from features_ks import EXPOSURE_COL, FEATURE_COLS, TARGET_COL, UMPIRE_FEATURE_COL
from model_ks import (HOLDOUT_SEASON, PROP_LINES, TRAIN_SEASONS, _standardize,
                       crps_nbinom, fit_nb2, prob_over)


def fit_and_score(train: pd.DataFrame, holdout: pd.DataFrame, regressor_cols: list[str]):
    (X_train_std, X_holdout_std), _, _ = _standardize(train[regressor_cols], holdout[regressor_cols])
    y_train, y_holdout = train[TARGET_COL].values, holdout[TARGET_COL].values
    exp_train, exp_holdout = train[EXPOSURE_COL].values, holdout[EXPOSURE_COL].values

    fit = fit_nb2(y_train, X_train_std, exp_train)
    alpha = fit.params["alpha"]
    beta = fit.params.drop("alpha", errors="ignore")
    X_holdout_const = sm.add_constant(X_holdout_std, has_constant="add")[beta.index]
    mu = np.exp(X_holdout_const.values @ beta.values) * exp_holdout

    crps = crps_nbinom(mu, alpha, y_holdout).mean()
    brier = {}
    for line in PROP_LINES:
        actual_over = (y_holdout > line).astype(float)
        p = prob_over(mu, alpha, line)
        brier[line] = float(np.mean((p - actual_over) ** 2))
    return crps, brier, fit


def main():
    print("building baseline features (no umpire)...")
    baseline_df = features_ks.build_features(include_umpire=False)
    print("building umpire-augmented features...")
    umpire_df = features_ks.build_features(include_umpire=True)

    b_train = baseline_df[baseline_df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    b_holdout = baseline_df[baseline_df["season"] == HOLDOUT_SEASON].reset_index(drop=True)
    u_train = umpire_df[umpire_df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    u_holdout = umpire_df[umpire_df["season"] == HOLDOUT_SEASON].reset_index(drop=True)

    print(f"\nbaseline: train={len(b_train):,} holdout={len(b_holdout):,}")
    print(f"+umpire : train={len(u_train):,} holdout={len(u_holdout):,}")
    if len(b_train) != len(u_train) or len(b_holdout) != len(u_holdout):
        print("[warn] row counts differ between baseline and +umpire feature sets -- "
              "some starts' games are missing officials data. Comparison is still valid "
              "(each model evaluated on its own holdout), but note the difference.")

    baseline_regressors = [c for c in FEATURE_COLS if c != EXPOSURE_COL]
    umpire_regressors = baseline_regressors + [UMPIRE_FEATURE_COL]

    crps_b, brier_b, fit_b = fit_and_score(b_train, b_holdout, baseline_regressors)
    crps_u, brier_u, fit_u = fit_and_score(u_train, u_holdout, umpire_regressors)

    print(f"\n{'=' * 74}\nUMPIRE FEATURE A/B TEST -- {HOLDOUT_SEASON} holdout\n{'=' * 74}")
    print(f"\nCRPS (lower is better):")
    print(f"  baseline (current model): {crps_b:.4f}")
    print(f"  +umpire_k_index         : {crps_u:.4f}  "
          f"({(1 - crps_u / crps_b) * 100:+.2f}% {'improvement' if crps_u < crps_b else 'change'})")

    print(f"\nBrier score at prop lines (lower is better):")
    for line in PROP_LINES:
        delta = (1 - brier_u[line] / brier_b[line]) * 100
        print(f"  {line}: baseline={brier_b[line]:.4f}  +umpire={brier_u[line]:.4f}  "
              f"({delta:+.2f}% {'better' if brier_u[line] < brier_b[line] else 'worse/flat'})")

    ump_coef = fit_u.params.get(UMPIRE_FEATURE_COL, float("nan"))
    ump_p = fit_u.pvalues.get(UMPIRE_FEATURE_COL, float("nan"))
    print(f"\numpire_k_index coefficient (standardized): {ump_coef:+.4f}  (p={ump_p:.3f})")

    beats_crps = crps_u < crps_b
    beats_all_lines = all(brier_u[l] < brier_b[l] for l in PROP_LINES)
    beats_any_line = any(brier_u[l] < brier_b[l] for l in PROP_LINES)

    print(f"\n{'-' * 74}")
    if beats_crps and beats_all_lines:
        print("VERDICT: umpire feature helps on CRPS AND every prop line. Keep it.")
    elif not beats_crps and not beats_any_line:
        print("VERDICT: umpire feature is flat or worse on every metric. Drop it.")
    else:
        print("VERDICT: mixed -- helps on some metrics, not all. Marginal, not a clean win; "
              "lean toward dropping given the added data-pull cost and model complexity.")
    print(f"{'-' * 74}")


if __name__ == "__main__":
    main()
