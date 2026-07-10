"""
Track 2: pitcher strikeout distribution model.

Negative Binomial GLM (NB2), predicting each start's strikeout COUNT with trailing
IP/start used as an exposure/offset (log-exposure), not a regular covariate -- this
is the textbook way to separate "how many innings is he expected to throw" from "how
many Ks per inning is he good for," rather than letting workload differences get
tangled up in the rate coefficients. Poisson is fit alongside purely to justify NB2
via a dispersion check (count data is very commonly overdispersed relative to Poisson).

A full NB2 (all FEATURE_COLS) is compared against a NAIVE NB2 (mu = trailing-3-start
K rate x trailing IP/start, no other covariates, its own separately-fit dispersion)
so "does the richer feature set beat a dumb extrapolation" is a fair, like-for-like
distributional comparison, not full-model-vs-a-point-estimate.

Train: 2021-2024. Validate: 2025, held out completely.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from features_ks import EXPOSURE_COL, FEATURE_COLS, PROCESSED, TARGET_COL

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

TRAIN_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
REGRESSOR_COLS = [c for c in FEATURE_COLS if c != EXPOSURE_COL]
CRPS_MAX_K = 25
PROP_LINES = [4.5, 5.5, 6.5]


def load_split():
    df = pd.read_parquet(PROCESSED / "model_features_ks.parquet")
    train = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    holdout = df[df["season"] == HOLDOUT_SEASON].reset_index(drop=True)
    return train, holdout


def _standardize(train_X: pd.DataFrame, *others: pd.DataFrame):
    mu, sd = train_X.mean(), train_X.std().replace(0, 1)
    out = [((train_X - mu) / sd)]
    for o in others:
        out.append((o - mu) / sd)
    return out, mu, sd


def fit_nb2(y: np.ndarray, X: pd.DataFrame, exposure: np.ndarray):
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.NegativeBinomial(y, X_const, exposure=exposure, loglike_method="nb2")
    return model.fit(disp=False, maxiter=200)


def fit_poisson(y: np.ndarray, X: pd.DataFrame, exposure: np.ndarray):
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, X_const, family=sm.families.Poisson(), exposure=exposure)
    return model.fit()


def nb2_np(mu: np.ndarray, alpha: float):
    """NB2 parameterization (mean=mu, var=mu + alpha*mu^2) -> scipy's (n, p)."""
    p = 1.0 / (1.0 + alpha * mu)
    n = np.full_like(p, 1.0 / alpha)
    return n, p


def crps_nbinom(mu: np.ndarray, alpha: float, actual: np.ndarray, k_max: int = CRPS_MAX_K) -> np.ndarray:
    n, p = nb2_np(mu, alpha)
    ks = np.arange(0, k_max + 1)
    # cdf shape: (n_obs, k_max+1)
    cdf = np.array([stats.nbinom.cdf(ks, n_i, p_i) for n_i, p_i in zip(n, p)])
    indicator = (ks[None, :] >= actual[:, None]).astype(float)
    return ((cdf - indicator) ** 2).sum(axis=1)


def prob_over(mu: np.ndarray, alpha: float, line: float) -> np.ndarray:
    n, p = nb2_np(mu, alpha)
    threshold = int(np.floor(line))  # P(K > 4.5) = P(K >= 5) = 1 - CDF(4)
    return 1 - stats.nbinom.cdf(threshold, n, p)


def run():
    train, holdout = load_split()
    print(f"train: {len(train):,} starts ({TRAIN_SEASONS})  |  holdout: {len(holdout):,} starts ({HOLDOUT_SEASON})")
    print(f"SO mean={train[TARGET_COL].mean():.2f} var={train[TARGET_COL].var():.2f} "
          f"(var/mean={train[TARGET_COL].var()/train[TARGET_COL].mean():.2f})")

    (X_train_std, X_holdout_std), mu_s, sd_s = _standardize(train[REGRESSOR_COLS], holdout[REGRESSOR_COLS])

    y_train = train[TARGET_COL].values
    exp_train = train[EXPOSURE_COL].values
    y_holdout = holdout[TARGET_COL].values
    exp_holdout = holdout[EXPOSURE_COL].values

    # --- dispersion check: Poisson vs NB2 ---
    pois_fit = fit_poisson(y_train, X_train_std, exp_train)
    pearson_resid = (y_train - pois_fit.fittedvalues) / np.sqrt(pois_fit.fittedvalues)
    dispersion = float((pearson_resid ** 2).sum() / pois_fit.df_resid)
    print(f"\nPoisson Pearson dispersion statistic: {dispersion:.3f} (1.0 = Poisson is fine; "
          f"materially > 1 means overdispersed -> use NB2)")

    # --- full NB2 model ---
    full_fit = fit_nb2(y_train, X_train_std, exp_train)
    alpha_full = full_fit.params["alpha"] if "alpha" in full_fit.params.index else full_fit.params[-1]
    beta_full = full_fit.params.drop("alpha", errors="ignore")
    X_holdout_const = sm.add_constant(X_holdout_std, has_constant="add")[beta_full.index]
    mu_full = np.exp(X_holdout_const.values @ beta_full.values) * exp_holdout

    print("\nfull NB2 coefficients (on standardized features, so magnitude = relative importance):")
    print(full_fit.params.drop("alpha", errors="ignore").sort_values(key=abs, ascending=False).to_string())

    # --- naive NB2: mu = naive_pred_k directly, intercept-only multiplicative correction ---
    naive_X_train = pd.DataFrame({"const": 1.0}, index=train.index)
    naive_exp_train = train["naive_pred_k"].values.clip(min=0.1)
    naive_model = sm.NegativeBinomial(y_train, naive_X_train, exposure=naive_exp_train, loglike_method="nb2")
    naive_fit = naive_model.fit(disp=False)
    alpha_naive = naive_fit.params["alpha"]
    naive_exp_holdout = holdout["naive_pred_k"].values.clip(min=0.1)
    mu_naive = np.exp(naive_fit.params["const"]) * naive_exp_holdout

    # --- holdout validation: CRPS + Brier @ prop lines ---
    crps_full = crps_nbinom(mu_full, alpha_full, y_holdout)
    crps_naive = crps_nbinom(mu_naive, alpha_naive, y_holdout)

    print(f"\n{'=' * 78}\nSTRIKEOUT MODEL BACKTEST -- {HOLDOUT_SEASON} holdout (n={len(holdout):,} starts)\n{'=' * 78}")
    print(f"\nCRPS (lower is better, 0 = perfect):")
    print(f"  full NB2 model : {crps_full.mean():.4f}")
    print(f"  naive baseline : {crps_naive.mean():.4f}")
    print(f"  full model {'BEATS' if crps_full.mean() < crps_naive.mean() else 'DOES NOT BEAT'} naive "
          f"({(1 - crps_full.mean()/crps_naive.mean())*100:+.1f}% {'improvement' if crps_full.mean() < crps_naive.mean() else 'change'})")

    print(f"\nBrier score at prop lines (P(K > line) vs actual, lower is better, 0.25 = coin flip):")
    line_rows = []
    for line in PROP_LINES:
        actual_over = (y_holdout > line).astype(float)
        p_full = prob_over(mu_full, alpha_full, line)
        p_naive = prob_over(mu_naive, alpha_naive, line)
        bs_full = float(np.mean((p_full - actual_over) ** 2))
        bs_naive = float(np.mean((p_naive - actual_over) ** 2))
        line_rows.append({"line": line, "over_rate": actual_over.mean(),
                           "brier_full": bs_full, "brier_naive": bs_naive,
                           "full_beats_naive": bs_full < bs_naive})
        print(f"  {line}: full={bs_full:.4f}  naive={bs_naive:.4f}  "
              f"{'model wins' if bs_full < bs_naive else 'naive wins'}  (actual over-rate: {actual_over.mean():.1%})")

    line_df = pd.DataFrame(line_rows)
    beats_on_all = bool(line_df["full_beats_naive"].all())
    beats_on_crps = crps_full.mean() < crps_naive.mean()

    print(f"\n{'-'*78}")
    if beats_on_crps and beats_on_all:
        print("VERDICT: full model beats naive on CRPS AND every prop line. Worth building daily_ks.py.")
    elif beats_on_crps or line_df["full_beats_naive"].any():
        print("VERDICT: mixed -- beats naive on some metrics, not all. Marginal, not a clean win.")
    else:
        print("VERDICT: full model does NOT beat naive. The extra features (whiff%, opponent K%, "
              "rest, 30-day K/9) aren't adding value over a simple trailing-rate extrapolation.")
    print(f"{'-'*78}")

    holdout_out = holdout[["mlbID", "game_pk", "season", "official_date", "Name", TARGET_COL, "naive_pred_k"]].copy()
    holdout_out["mu_full"] = mu_full
    holdout_out["mu_naive"] = mu_naive
    holdout_out["crps_full"] = crps_full
    holdout_out["crps_naive"] = crps_naive
    holdout_out.to_parquet(OUTPUT / "ks_holdout_predictions.parquet", index=False)
    line_df.to_csv(OUTPUT / "ks_prop_line_brier.csv", index=False)

    joblib.dump({"beta": beta_full, "alpha": alpha_full, "mu_scale": mu_s, "sd_scale": sd_s,
                 "regressor_cols": REGRESSOR_COLS, "exposure_col": EXPOSURE_COL}, OUTPUT / "model_ks.joblib")

    return {"crps_full": crps_full, "crps_naive": crps_naive, "line_df": line_df,
            "beats_on_all": beats_on_all, "beats_on_crps": beats_on_crps,
            "full_fit": full_fit, "dispersion": dispersion}


if __name__ == "__main__":
    run()
