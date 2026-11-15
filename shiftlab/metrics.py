"""Per-class metrics and patient-level (cluster) bootstrap intervals."""
from __future__ import annotations

import numpy as np

from .config import CLASSES

K = len(CLASSES)


def confusion(y, p) -> np.ndarray:
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(p, int)), 1)
    return cm


def from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1)
    pred = cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(sup > 0, tp / sup, np.nan)
        ppv = np.where(pred > 0, tp / pred, 0.0)
        f1 = np.where(sup > 0, 2 * tp / (sup + pred), np.nan)
    out = {"acc": float(tp.sum() / cm.sum())}
    for i, c in enumerate(CLASSES):
        out[f"se_{c}"] = float(se[i])
        out[f"ppv_{c}"] = float(ppv[i])
        out[f"f1_{c}"] = float(f1[i])
    out["macro_f1"] = float(np.nanmean(f1))
    out["macro_f1_NSV"] = float(np.nanmean(f1[:3]))
    out["bal_acc"] = float(np.nanmean(se))
    return out


def score(y, p) -> dict:
    return from_cm(confusion(y, p))


def cluster_bootstrap(y, p, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Resample patients, not beats. Beats within a patient are not independent,
    so a beat-level bootstrap would understate the uncertainty badly."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    cms = np.stack([confusion(y[groups == g], p[groups == g]) for g in ids])
    keys = list(from_cm(cms.sum(0)).keys())
    draws = np.empty((n_boot, len(keys)))
    for b in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        m = from_cm(cms[pick].sum(0))
        draws[b] = [m[k] for k in keys]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return {k: [float(a), float(b)] for k, a, b in zip(keys, lo, hi)}


def per_group(y, p, groups) -> dict:
    return {str(g): {**score(y[groups == g], p[groups == g]),
                     "n": int((groups == g).sum()),
                     "support": confusion(y[groups == g], p[groups == g]).sum(1).tolist()}
            for g in np.unique(groups)}


def patient_mean(y, p, groups, min_support: int = 20) -> dict:
    """Class F1 computed within each patient, then averaged over the patients who
    have at least `min_support` beats of that class. Every patient counts once,
    so one arrhythmia-heavy record cannot decide the score."""
    out = {}
    per = per_group(y, p, groups)
    for i, c in enumerate(CLASSES):
        vals = [v[f"f1_{c}"] for v in per.values() if v["support"][i] >= min_support]
        out[f"pm_f1_{c}"] = float(np.mean(vals)) if vals else float("nan")
        out[f"pm_n_{c}"] = len(vals)
    return out
