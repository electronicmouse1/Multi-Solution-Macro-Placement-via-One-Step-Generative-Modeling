"""Within-design ranking metrics for predictor evaluation (and for proxy baselines)."""
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def ranking_metrics(pred, true, k=5, scale=None):
    """pred, true: 1-D arrays of COSTS for one design (lower is better)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    m = ~(np.isnan(pred) | np.isnan(true))
    pred, true = pred[m], true[m]
    n = len(true)
    if n < 4 or np.std(true) == 0 or np.std(pred) == 0:
        return None
    iu = np.triu_indices(n, 1)
    dt = (true[:, None] - true[None, :])[iu]
    dp = (pred[:, None] - pred[None, :])[iu]
    keep = dt != 0
    pair_acc = float(np.mean(np.sign(dt[keep]) == np.sign(dp[keep]))) if keep.any() else float("nan")
    kk = min(k, n)
    top_true = set(np.argsort(true)[:kk])
    top_pred = set(np.argsort(pred)[:kk])
    best_pred = int(np.argmin(pred))
    s = scale if scale else (np.percentile(true, 75) - np.percentile(true, 25)) / 1.349 or np.std(true)
    return {"n": n,
            "spearman": float(spearmanr(pred, true).correlation),
            "kendall": float(kendalltau(pred, true).correlation),
            "pearson": float(pearsonr(pred, true)[0]),
            "pair_acc": pair_acc,
            f"top{kk}_recall": len(top_true & top_pred) / kk,
            "best_in_pred_topk": float(int(np.argmin(true)) in top_pred),
            "regret_z": float((true[best_pred] - true.min()) / s) if s else float("nan"),
            "pred_best_percentile": float((true < true[best_pred]).mean())}


def mean_over(rows, key):
    v = [r[key] for r in rows if r is not None and r.get(key) is not None and np.isfinite(r[key])]
    return (float(np.mean(v)), len(v), int(np.sum(np.array(v) > 0))) if v else (float("nan"), 0, 0)
