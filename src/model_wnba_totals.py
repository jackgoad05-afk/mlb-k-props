"""
WNBA totals model -- predicts each game's combined (home + away) point total.
No totals model exists anywhere else in this repo (MLB never got one -- see
CLAUDE.md, out of scope for the K-props/moneyline builds).

RidgeCV, not HistGradientBoostingRegressor -- same overfitting finding as
model_wnba_ml.py: ~900 training games is too little for GBM's flexibility here,
a heavily-regularized linear model does better (RMSE 16.74 vs. HGB's 17.00 on
the 2025 holdout; CV picked alpha=100, i.e. strong shrinkage, which itself is a
signal that per-feature correlations with total_points are genuinely weak, not
that the model needs more capacity to find them).

Combined WNBA scoring runs ~140-190 points, high enough as a count that a
Gaussian residual is the right distributional choice -- unlike K props' NB2
(strikeouts are a low, over-dispersed count, ~0-15). `sigma` is fit ONCE from
leave-one-season-out out-of-fold residuals on the training set (never the 2025
holdout).

Train: 2021-2024. Validate: 2025, held out completely. Honest result (see
backtest_wnba.py): this model comes close to but does not clearly beat the
simplest possible heuristic (total_form_avg alone, no ML) on the 2025 holdout
-- game-level total-points variance in the WNBA looks dominated by randomness/
pace that these team-scoring-rate features don't fully capture. Likely next
investment: real per-game pace/possession features from box scores (deferred
this build for API-call cost -- see fetch_wnba.py's docstring), not more
model complexity on the current feature set.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features_wnba import FEATURE_COLS, PROCESSED, TARGET_TOTALS

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

TRAIN_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
RANDOM_STATE = 42
RIDGE_ALPHAS = [0.1, 1, 10, 50, 100, 200]


def load_split():
    df = pd.read_parquet(PROCESSED / "model_features_wnba.parquet")
    train = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    holdout = df[df["season"] == HOLDOUT_SEASON].reset_index(drop=True)
    return train, holdout


def _fit_one(train_df: pd.DataFrame):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_TOTALS])
    return model


def leave_one_season_out_residual_std(train: pd.DataFrame) -> float:
    """Out-of-fold residual std on the training set, used ONLY to fix sigma for
    the Gaussian P(over/under) calc -- never touching 2025."""
    residuals = []
    for held_season in TRAIN_SEASONS:
        fit_seasons = [s for s in TRAIN_SEASONS if s != held_season]
        fit_df = train[train["season"].isin(fit_seasons)]
        val_df = train[train["season"] == held_season]
        model = _fit_one(fit_df)
        preds = model.predict(val_df[FEATURE_COLS])
        residuals.append(val_df[TARGET_TOTALS].values - preds)
    return float(np.std(np.concatenate(residuals)))


def prob_over(mu: np.ndarray, sigma: float, line: float) -> np.ndarray:
    return 1.0 - norm.cdf((line - mu) / sigma)


def train_final_model(train: pd.DataFrame):
    return _fit_one(train)


def run():
    train, holdout = load_split()
    print(f"train: {len(train):,} games ({TRAIN_SEASONS})  |  holdout: {len(holdout):,} games ({HOLDOUT_SEASON})")

    sigma = leave_one_season_out_residual_std(train)
    print(f"OOF residual std (sigma): {sigma:.2f} points")

    final_model = train_final_model(train)
    mu_holdout = final_model.predict(holdout[FEATURE_COLS])

    result = holdout[["game_id", "season", "official_date", "home_team_name", "away_team_name",
                       TARGET_TOTALS]].copy()
    result["mu"] = mu_holdout
    result["residual"] = result[TARGET_TOTALS] - result["mu"]

    joblib.dump({"model": final_model, "sigma": sigma}, OUTPUT / "model_wnba_totals.joblib")
    result.to_parquet(OUTPUT / "wnba_totals_holdout_predictions.parquet", index=False)

    mae = result["residual"].abs().mean()
    rmse = float(np.sqrt((result["residual"] ** 2).mean()))
    print(f"\nholdout MAE: {mae:.2f}  RMSE: {rmse:.2f}  (naive std-dev baseline: {holdout[TARGET_TOTALS].std():.2f})")

    perm = permutation_importance(final_model, holdout[FEATURE_COLS], holdout[TARGET_TOTALS],
                                   n_repeats=15, random_state=RANDOM_STATE, scoring="neg_mean_squared_error")
    importances = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)
    print("\ntop feature importances (permutation, holdout MSE impact):")
    print(importances.to_string())

    return result, final_model, sigma


if __name__ == "__main__":
    run()
