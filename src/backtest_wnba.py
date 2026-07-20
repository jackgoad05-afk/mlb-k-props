"""
Holdout (2025) backtest for both WNBA models against a proxy market -- no free
historical WNBA odds dataset exists (checked; unlike MLB's Track 1 GitHub
scrape, nothing equivalent turned up), so this is a proxy-vs-model comparison,
NOT a real-market backtest. Labeled "(proxy)" throughout, same discipline as
backtest.py's MLB proxy-market disclaimer -- don't quote these numbers as real
profitability.

Moneyline proxy: log5(home_win_form, away_win_form) + empirical home-court-
advantage odds ratio, same method as backtest.py's MLB proxy. Reuses the
rolling win_form columns already in model_features_wnba.parquet (an as-of-date,
shrunk-toward-prior-season win%) rather than rebuilding an isolated shrinkage
pipeline -- a simplification worth naming: this proxy isn't fully independent
of the model's own inputs (both draw on the same rolling win_form), so a close
result here is a weaker signal than MLB's proxy comparison was. Still useful
for a sanity check and directionally informative.

Totals proxy: total_form_avg itself (the simple heuristic already computed as
a model feature -- "average of both teams' own rolling scoring+allowed") used
directly as a point prediction, with its OWN separately-fit OOF residual std
-- a fair distributional comparison, not full-model-vs-a-point-estimate (same
fairness discipline CLAUDE.md documents for model_ks.py's naive baseline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from features_wnba import PROCESSED, TARGET_ML, TARGET_TOTALS
from model_wnba_ml import HOLDOUT_SEASON as ML_HOLDOUT, TRAIN_SEASONS as ML_TRAIN
from model_wnba_ml import run as run_ml
from model_wnba_totals import run as run_totals

TOTALS_LINES = [155.5, 165.5, 175.5]


def log5(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    return (p_a - p_a * p_b) / (p_a + p_b - 2 * p_a * p_b)


def apply_hfa(p_home_no_hfa: np.ndarray, odds_ratio_hfa: float) -> np.ndarray:
    odds = p_home_no_hfa / (1 - p_home_no_hfa)
    odds_adj = odds * odds_ratio_hfa
    return odds_adj / (1 + odds_adj)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def moneyline_proxy(df: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    home_wp = train[TARGET_ML].mean()
    hfa_or = home_wp / (1 - home_wp)
    raw = log5(df["home_win_form"].values, df["away_win_form"].values)
    return apply_hfa(raw, hfa_or)


def totals_proxy_sigma(train: pd.DataFrame) -> float:
    resid = train[TARGET_TOTALS].values - train["total_form_avg"].values
    return float(np.std(resid))


def run():
    df = pd.read_parquet(PROCESSED / "model_features_wnba.parquet")
    train = df[df["season"].isin(ML_TRAIN)].reset_index(drop=True)
    holdout = df[df["season"] == ML_HOLDOUT].reset_index(drop=True)

    print("=== moneyline: model vs. proxy (log5 + HFA, NOT real odds) ===")
    ml_result, _, _ = run_ml()
    ml_result = ml_result.merge(holdout[["game_id", "home_win_form", "away_win_form"]], on="game_id")
    proxy_prob = moneyline_proxy(ml_result, train)

    bs_model = brier_score(ml_result["model_prob"].values, ml_result[TARGET_ML].values)
    bs_proxy = brier_score(proxy_prob, ml_result[TARGET_ML].values)
    print(f"Brier -- model: {bs_model:.4f}  |  proxy (log5+HFA): {bs_proxy:.4f}  "
          f"({'model beats proxy' if bs_model < bs_proxy else 'model does NOT beat proxy'})")
    print(f"model prob std dev: {ml_result['model_prob'].std():.3f}  |  "
          f"proxy prob std dev: {proxy_prob.std():.3f}")

    print("\n=== totals: model vs. proxy (total_form_avg heuristic, NOT real odds) ===")
    totals_result, _, sigma_model = run_totals()
    proxy_sigma = totals_proxy_sigma(train)
    proxy_mu = holdout["total_form_avg"].values

    mae_model = totals_result["residual"].abs().mean()
    rmse_model = float(np.sqrt((totals_result["residual"] ** 2).mean()))
    resid_proxy = holdout[TARGET_TOTALS].values - proxy_mu
    mae_proxy = np.abs(resid_proxy).mean()
    rmse_proxy = float(np.sqrt((resid_proxy ** 2).mean()))
    print(f"MAE  -- model: {mae_model:.2f}  |  proxy: {mae_proxy:.2f}")
    print(f"RMSE -- model: {rmse_model:.2f}  |  proxy: {rmse_proxy:.2f}  "
          f"({'model beats proxy' if rmse_model < rmse_proxy else 'model does NOT beat proxy'})")

    print("\nBrier at representative total lines (model sigma={:.2f}, proxy sigma={:.2f}):".format(
        sigma_model, proxy_sigma))
    actual_over = {line: (holdout[TARGET_TOTALS].values > line).astype(float) for line in TOTALS_LINES}
    for line in TOTALS_LINES:
        p_model = 1 - norm.cdf((line - totals_result["mu"].values) / sigma_model)
        p_proxy = 1 - norm.cdf((line - proxy_mu) / proxy_sigma)
        bs_m = brier_score(p_model, actual_over[line])
        bs_p = brier_score(p_proxy, actual_over[line])
        print(f"  {line}: model {bs_m:.4f}  |  proxy {bs_p:.4f}  "
              f"({'model beats proxy' if bs_m < bs_p else 'model does NOT beat proxy'})")


if __name__ == "__main__":
    run()
