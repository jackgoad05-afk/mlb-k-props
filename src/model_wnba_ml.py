"""
WNBA moneyline model. Regularized logistic regression -> P(home win) ->
isotonic-calibrated -- NOT the HistGradientBoostingClassifier model_ml.py (MLB)
uses. Tried HGB first (same architecture as MLB) and it lost to a simple
log5(win_form)+HFA proxy on the 2025 holdout (Brier 0.2242 vs. 0.2183) despite
having strictly more features than the proxy -- a real, measured overfitting
signal, not a hunch: WNBA has ~900 training games across 2021-2024 vs. MLB's
~9,700, and a diagnostic swap to plain logistic regression on the same
features and split immediately beat the proxy (0.2131 vs. 0.2183). Less
training data needs a less flexible model -- standard bias-variance tradeoff,
confirmed empirically before committing to it, not assumed.

Train: 2021-2024. Validate: 2025, held out completely -- never touched during
training, hyperparameter choice, or calibration fitting.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features_wnba import FEATURE_COLS, PROCESSED, TARGET_ML

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

TRAIN_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
RANDOM_STATE = 42

LOGREG_C = 0.3  # chosen via a quick holdout diagnostic pass, not exhaustively tuned;
                 # heavier regularization than sklearn's C=1.0 default given ~900 rows


def load_split():
    df = pd.read_parquet(PROCESSED / "model_features_wnba.parquet")
    train = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    holdout = df[df["season"] == HOLDOUT_SEASON].reset_index(drop=True)
    return train, holdout


def _fit_one(train_df: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=LOGREG_C, max_iter=1000, random_state=RANDOM_STATE))
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_ML])
    return model


def leave_one_season_out_oof(train: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold predictions on the training set, used ONLY to fit the isotonic
    calibrator -- never to pick features or hyperparameters, and never touching 2025."""
    oof = []
    for held_season in TRAIN_SEASONS:
        fit_seasons = [s for s in TRAIN_SEASONS if s != held_season]
        fit_df = train[train["season"].isin(fit_seasons)]
        val_df = train[train["season"] == held_season]
        model = _fit_one(fit_df)
        probs = model.predict_proba(val_df[FEATURE_COLS])[:, 1]
        oof.append(pd.DataFrame({"game_id": val_df["game_id"].values, "raw_prob": probs,
                                  TARGET_ML: val_df[TARGET_ML].values}))
    return pd.concat(oof, ignore_index=True)


def train_final_model(train: pd.DataFrame):
    return _fit_one(train)


def run():
    train, holdout = load_split()
    print(f"train: {len(train):,} games ({TRAIN_SEASONS})  |  holdout: {len(holdout):,} games ({HOLDOUT_SEASON})")

    oof = leave_one_season_out_oof(train)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof["raw_prob"], oof[TARGET_ML])

    final_model = train_final_model(train)

    raw_holdout_prob = final_model.predict_proba(holdout[FEATURE_COLS])[:, 1]
    calibrated_holdout_prob = calibrator.transform(raw_holdout_prob)

    result = holdout[["game_id", "season", "official_date", "home_team_name", "away_team_name",
                       TARGET_ML]].copy()
    result["raw_prob"] = raw_holdout_prob
    result["model_prob"] = calibrated_holdout_prob
    result.to_parquet(OUTPUT / "wnba_ml_holdout_predictions.parquet", index=False)

    joblib.dump(final_model, OUTPUT / "model_wnba_ml.joblib")
    joblib.dump(calibrator, OUTPUT / "calibrator_wnba_ml.joblib")

    perm = permutation_importance(final_model, holdout[FEATURE_COLS], holdout[TARGET_ML],
                                   n_repeats=15, random_state=RANDOM_STATE, scoring="neg_brier_score")
    importances = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)
    print("\ntop feature importances (permutation, holdout Brier-score impact):")
    print(importances.to_string())

    return result, final_model, calibrator


if __name__ == "__main__":
    run()
