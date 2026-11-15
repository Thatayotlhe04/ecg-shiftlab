"""Raw WFDB records -> beat-level arrays.

Every beat keeps its record id so that every split downstream can be made
(and audited) at the patient level.
"""
from __future__ import annotations

import os

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

from . import config as C


def _mlii(record) -> np.ndarray:
    """Modified limb lead II, wherever it sits in the file (114 stores it second)."""
    names = [n.upper() for n in record.sig_name]
    if "MLII" not in names:
        raise ValueError(f"{record.record_name}: no MLII lead")
    return record.p_signal[:, names.index("MLII")].astype(np.float64)


def _bandpass(x: np.ndarray) -> np.ndarray:
    sos = butter(3, C.BANDPASS, btype="band", fs=C.FS, output="sos")
    return sosfiltfilt(sos, x)


def load_record(raw_dir: str, name: str) -> dict:
    rec = wfdb.rdrecord(os.path.join(raw_dir, name))
    ann = wfdb.rdann(os.path.join(raw_dir, name), "atr")
    sig = _bandpass(_mlii(rec))

    # Keep every annotated beat (including Q) for RR computation; label later.
    is_beat = np.array([s in C.AAMI for s in ann.symbol])
    pos = np.asarray(ann.sample)[is_beat]
    sym = np.asarray(ann.symbol)[is_beat]

    rr = np.diff(pos) / C.FS  # seconds
    med = float(np.median(rr))
    pre = np.concatenate([[med], rr])
    post = np.concatenate([rr, [med]])
    # causal local rhythm: mean of the previous LOCAL_RR_BEATS intervals
    local = np.empty_like(pre)
    for i in range(len(pre)):
        lo = max(0, i - C.LOCAL_RR_BEATS + 1)
        local[i] = pre[lo:i + 1].mean()

    keep = (pos - C.PRE >= 0) & (pos + C.POST <= len(sig))
    keep &= np.array([C.AAMI[s] in C.CLASSES for s in sym])
    idx = np.flatnonzero(keep)

    wave = np.stack([sig[p - C.PRE:p + C.POST] for p in pos[idx]]).astype(np.float32)
    y = np.array([C.CLASSES.index(C.AAMI[s]) for s in sym[idx]], dtype=np.int8)
    return dict(
        wave=wave,
        y=y,
        sym=sym[idx],
        pos=pos[idx].astype(np.int32),
        rr_pre=pre[idx].astype(np.float32),
        rr_post=post[idx].astype(np.float32),
        rr_local=local[idx].astype(np.float32),
        rr_med=np.full(len(idx), med, dtype=np.float32),
        rec=np.full(len(idx), name),
    )


def build(raw_dir: str, out_path: str) -> dict:
    parts = [load_record(raw_dir, r) for r in C.RECORDS]
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["subject"] = np.array([C.SUBJECT[r] for r in data["rec"]])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, out_path)
    return data


def load(path: str = "data/processed/beats.npz") -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
