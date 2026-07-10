"""
Session 2: moneyline model.

Gradient-boosted classifier -> P(home win) -> isotonic-calibrated.
Calibration matters more than raw accuracy for a betting model: a well-calibrated
60% needs to actually win ~60% of the time, or every downstream edge/ROI number
built on top of it is wrong in a way that's invisible until you've lost money on it.

Spec calls for LightGBM; using sklearn's HistGradientBoostingClassifier instead
because LightGBM's macOS wheel needs libomp (Homebrew) which isn't available on
this machine. Same histogram-based gradient boosting family. See CLAUDE.md.

Train: 2021-2024. Validate: 2025, held out completely -- never touched during
training, hyperparameter choice, or calibration fitting.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression

from features import FEATURE_COLS, TARGET_COL, PROCESSED

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

TRAIN_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
RANDOM_STATE = 42

HGB_PARAMS = dict(
    max_iter=500,
    learning_rate=0.03,
    max_depth=4,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)


def load_split():
    df = pd.read_parquet(PROCESSED / "model_features_ml.parquet")
    train = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    holdout = df[df["season"] == HOLDOUT_SEASON].reset_index(drop=True)
    return train, holdout


def _fit_one(train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])
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
        oof.append(pd.DataFrame({"game_pk": val_df["game_pk"].values, "raw_prob": probs,
                                  TARGET_COL: val_df[TARGET_COL].values}))
    return pd.concat(oof, ignore_index=True)


def train_final_model(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Final model for scoring the holdout: trained on all 4 training seasons, with
    HGB's own internal (shuffled, in-training-set) early stopping split."""
    return _fit_one(train)


def run():
    train, holdout = load_split()
    print(f"train: {len(train):,} games ({TRAIN_SEASONS})  |  holdout: {len(holdout):,} games ({HOLDOUT_SEASON})")

    oof = leave_one_season_out_oof(train)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof["raw_prob"], oof[TARGET_COL])

    final_model = train_final_model(train)

    raw_holdout_prob = final_model.predict_proba(holdout[FEATURE_COLS])[:, 1]
    calibrated_holdout_prob = calibrator.transform(raw_holdout_prob)

    result = holdout[["game_pk", "season", "official_date", "home_team_name", "away_team_name",
                       TARGET_COL]].copy()
    result["raw_prob"] = raw_holdout_prob
    result["model_prob"] = calibrated_holdout_prob
    result.to_parquet(OUTPUT / "ml_holdout_predictions.parquet", index=False)

    joblib.dump(final_model, OUTPUT / "model_ml.joblib")
    joblib.dump(calibrator, OUTPUT / "calibrator_ml.joblib")

    # Post-hoc diagnostic only (computed on the holdout after scoring is final; does not
    # feed back into feature/hyperparameter choices, so it doesn't compromise the holdout).
    perm = permutation_importance(final_model, holdout[FEATURE_COLS], holdout[TARGET_COL],
                                   n_repeats=15, random_state=RANDOM_STATE, scoring="neg_brier_score")
    importances = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)
    print("\ntop feature importances (permutation, holdout Brier-score impact):")
    print(importances.head(15).to_string())

    return result, final_model, calibrator


if __name__ == "__main__":
    run()
