"""Model development that never touches DS2.

Candidates are compared by patient-wise cross-validation inside DS1 only
(11 folds of 2 records). Selection criterion, fixed before running: macro-F1
over N, S, V, F. The winner is evaluated on DS2 exactly once by
`python -m shiftlab.experiments v2`.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from shiftlab import config as C, data as D, features as F, metrics as E, models as M

REG = dict(num_leaves=15, min_child_samples=100, colsample_bytree=0.3)
CANDIDATES = {
    "template": ("template", {}, True),
    "context": ("context", {}, True),
    "context-reg": ("context", REG, True),
    "relative": ("relative", {}, True),
    "relative-reg": ("relative", REG, True),
    "scalars-only": ("context", REG, False),  # rhythm + template-similarity scalars, no waveform
}

d = D.load()
ds1 = np.isin(d["rec"], C.DS1)
y, rec = d["y"], d["rec"]
fold = np.full(len(y), -1)
for i, r in enumerate(C.DS1):
    fold[rec == r] = i // 2

out = {}
for name, (mode, params, use_wave) in CANDIDATES.items():
    t0 = time.time()
    s, t = F.represent(d, mode)
    if not use_wave:
        s = s[:, :, :0]
    pred = np.zeros(len(y), dtype=int)
    for k in range(11):
        tr, te = ds1 & (fold != k), fold == k
        assert not set(rec[tr]) & set(rec[te])
        pred[te] = M.predict_gbm(M.fit_gbm(s[tr], t[tr], y[tr], **params), s[te], t[te]).argmax(1)
    out[name] = E.score(y[ds1], pred[ds1])
    m = out[name]
    print(f"{name:14s} macroF1 {m['macro_f1']:.3f} | N {m['f1_N']:.3f}  S {m['f1_S']:.3f} "
          f"(Se {m['se_S']:.2f} PPV {m['ppv_S']:.2f})  V {m['f1_V']:.3f}  F {m['f1_F']:.3f}  [{time.time()-t0:.0f}s]",
          flush=True)
best = max(out, key=lambda k: out[k]["macro_f1"])
out["selected"] = best
json.dump(out, open("results/dev_ds1.json", "w"), indent=1)
print("selected:", best)
