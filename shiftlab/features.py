"""Three beat representations that differ only in how much patient identity they keep.

raw       absolute amplitude (mV) and absolute RR intervals (s).
pnorm     per-record amplitude scaling and RR intervals relative to the patient's
          own rhythm. Uses only unlabeled statistics of the record.
template  pnorm + the beat's deviation from the record's median beat, i.e. every
          beat is described relative to that patient's dominant morphology.

pnorm/template statistics are computed per record from unlabeled signal, so they
are legitimate at test time (transductive over the record, never over labels).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

MODES = ("raw", "pnorm", "template")


def _per_record(data, fn):
    out = None
    for r in np.unique(data["rec"]):
        m = data["rec"] == r
        v = fn(m)
        if out is None:
            out = np.zeros((len(m),) + v.shape[1:], dtype=np.float32)
        out[m] = v
    return out


def represent(data: dict, mode: str):
    """Return (seq, tab): seq is (n, channels, WIN) for the CNN, tab is (n, k)."""
    assert mode in MODES + ("context", "relative")
    wave = data["wave"]
    pre, post, local, med = (data[k] for k in ("rr_pre", "rr_post", "rr_local", "rr_med"))

    if mode == "raw":
        seq = wave[:, None, :]
        tab = np.stack([pre, post, local], 1)
        return seq.astype(np.float32), tab.astype(np.float32)

    # robust per-record amplitude scale: median peak-to-peak of the beats
    def scale(m):
        w = wave[m]
        s = np.median(w.max(1) - w.min(1))
        w = (w - w.mean(1, keepdims=True)) / s
        return w

    wn = _per_record(data, scale)
    tab = [pre, post, local,
           pre / local, post / local, pre / post,
           pre / med, post / med, local / med]

    if mode == "pnorm":
        return wn[:, None, :].astype(np.float32), np.stack(tab, 1).astype(np.float32)

    def resid(m):
        w = wn[m]
        return w - np.median(w, 0, keepdims=True)

    res = _per_record(data, resid)
    tmpl = wn - res
    num = (wn * tmpl).sum(1)
    den = np.linalg.norm(wn, axis=1) * np.linalg.norm(tmpl, axis=1) + 1e-8
    tab += [num / den, np.sqrt((res ** 2).mean(1))]
    seq = np.stack([wn, res], 1)
    if mode in ("context", "relative"):
        tab += rhythm_context(data)
        if mode == "relative":
            seq = res[:, None, :]  # deviation from the patient's own template only
    return seq.astype(np.float32), np.stack(tab, 1).astype(np.float32)


def rhythm_context(data: dict) -> list:
    """Wider rhythm context around each beat, all relative to the patient's own
    recent rhythm. Lags are taken over the kept beats of a record (the handful
    of dropped Q beats are ignored)."""
    pre, post, med = data["rr_pre"], data["rr_post"], data["rr_med"]
    n = len(pre)
    pre2, post2, base, sd = (np.zeros(n, np.float32) for _ in range(4))
    for r in np.unique(data["rec"]):
        idx = np.flatnonzero(data["rec"] == r)
        p, q = pre[idx], post[idx]
        pre2[idx] = np.concatenate([[p[0]], p[:-1]])
        post2[idx] = np.concatenate([q[1:], [q[-1]]])
        # rhythm baseline from the 10 intervals *before* the current one
        b = np.empty(len(p), np.float32)
        v = np.empty(len(p), np.float32)
        for i in range(len(p)):
            w = p[max(0, i - 10):i] if i > 0 else p[:1]
            b[i] = np.median(w)
            v[i] = w.std() / (w.mean() + 1e-6)
        base[idx], sd[idx] = b, v
    return [pre / base, post / base, pre2 / base, post2 / base,
            (pre - post) / base, (pre + post) / (2 * base), sd, base / med]


def flat(seq: np.ndarray, tab: np.ndarray, down: int = 4) -> np.ndarray:
    """Tabular view for tree models: decimated waveform channels + RR features."""
    ds = resample_poly(seq, 1, down, axis=2).reshape(len(seq), -1)
    return np.concatenate([ds, tab], 1).astype(np.float32)
