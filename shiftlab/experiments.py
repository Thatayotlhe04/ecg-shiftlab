"""All experiments. Each one writes a JSON file to results/.

    python -m shiftlab.experiments benchmark   # E1  DS1 -> DS2, the headline
    python -m shiftlab.experiments leakage     # E2  same test beats, graded leakage
    python -m shiftlab.experiments cv          # E3  patient-wise vs beat-wise CV, per patient
    python -m shiftlab.experiments subject     # E4  the 201/202 same-subject overlap
    python -m shiftlab.experiments identity    # E5  how recognisable is a patient?
    python -m shiftlab.experiments unseen      #     E1, GBM rows only (minutes, no CNNs)
    python -m shiftlab.experiments v2          #     DS1-selected candidate, DS2 once
    python -m shiftlab.experiments all         # E1-E5
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from . import config as C
from . import data as D
from . import features as F
from . import metrics as E
from . import models as M

RES = "results"
MODES = F.MODES


def _save(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {name}.json", flush=True)


def _assert_disjoint(d, tr, te, level="subject"):
    """Hard guard: a patient-independent split may not share a subject."""
    shared = set(d[level][tr]) & set(d[level][te])
    assert not shared, f"patient leakage: {sorted(shared)}"


def _reps(d):
    return {m: F.represent(d, m) for m in MODES}


LABEL = {"raw": "raw", "pnorm": "normalised", "template": "relative"}
RULE = "\u2500"


def _head(*lines, cols, width=None):
    for ln in lines:
        print(ln)
    print()
    print(cols)
    print(RULE * (width or len(cols)), flush=True)


# ----------------------------------------------------------------------------- E1
def benchmark(d, reps, cnn_seeds=(0, 1, 2)):
    y, rec = d["y"], d["rec"]
    tr, te = np.isin(rec, C.DS1), np.isin(rec, C.DS2)
    _assert_disjoint(d, tr, te, level="rec")  # the standard split is record-disjoint only
    out, preds = {}, {}

    def pack(p, extra=None):
        r = {"metrics": E.score(y[te], p), "ci95": E.cluster_bootstrap(y[te], p, rec[te]),
             "confusion": E.confusion(y[te], p).tolist(), "per_record": E.per_group(y[te], p, rec[te])}
        r.update(extra or {})
        return r

    for mode in MODES:
        s, t = reps[mode]
        g = M.fit_gbm(s[tr], t[tr], y[tr])
        p = M.predict_gbm(g, s[te], t[te]).argmax(1)
        out[f"gbm/{mode}"] = pack(p)
        preds[f"gbm/{mode}"] = p
        print(f"gbm/{mode}", {k: round(v, 3) for k, v in out[f'gbm/{mode}']['metrics'].items()
                              if k.startswith(('f1', 'macro'))}, flush=True)

        prob = {False: [], True: []}
        for sd in cnn_seeds:
            net = M.fit_cnn(s[tr], t[tr], y[tr], seed=sd)
            for ada in (False, True):
                prob[ada].append(M.predict_cnn(net, s[te], t[te], groups=rec[te], adabn=ada))
        for ada in (False, True):
            key = f"cnn{'+adabn' if ada else ''}/{mode}"
            seeds = [E.score(y[te], q.argmax(1)) for q in prob[ada]]
            p = np.mean(prob[ada], 0).argmax(1)  # seed ensemble
            out[key] = pack(p, {"seed_mean": {k: float(np.mean([s_[k] for s_ in seeds])) for k in seeds[0]},
                                "seed_sd": {k: float(np.std([s_[k] for s_ in seeds])) for k in seeds[0]}})
            preds[key] = p
            print(key, {k: round(v, 3) for k, v in out[key]['seed_mean'].items()
                        if k.startswith(('f1', 'macro'))}, flush=True)

    np.savez_compressed(os.path.join(RES, "benchmark_preds.npz"), y=y[te], rec=rec[te],
                        **{k.replace("/", "__").replace("+", "_"): v for k, v in preds.items()})
    _save("benchmark", out)


def unseen(d, reps):
    """GBM rows of E1 on their own: train on DS1, test on the 22 unseen DS2 records."""
    y, rec = d["y"], d["rec"]
    tr, te = np.isin(rec, C.DS1), np.isin(rec, C.DS2)
    _assert_disjoint(d, tr, te, level="rec")
    _head("train: DS1, 22 records \u00b7 test: DS2, 22 unseen records",
          "95% intervals: 2,000 bootstrap resamples over patients",
          cols="features     acc     F1 S                F1 V", width=56)
    out = {}
    for mode in MODES:
        s, t = reps[mode]
        p = M.predict_gbm(M.fit_gbm(s[tr], t[tr], y[tr]), s[te], t[te]).argmax(1)
        m_, ci = E.score(y[te], p), E.cluster_bootstrap(y[te], p, rec[te])
        out[mode] = {"metrics": m_, "ci95": ci}
        f = lambda k_: f"{m_[k_]:.2f} [{ci[k_][0]:.2f}, {ci[k_][1]:.2f}]"
        print(f"{LABEL[mode]:<13}{m_['acc']:.3f}   {f('f1_S')}   {f('f1_V')}", flush=True)
    _save("unseen", out)


# ----------------------------------------------------------------------------- E2
def _leak_split(d):
    rec = d["rec"]
    rng = np.random.default_rng(123)
    ds1 = np.flatnonzero(np.isin(rec, C.DS1))
    ds2 = np.flatnonzero(np.isin(rec, C.DS2))
    half = rng.random(len(ds2)) < 0.5
    test, pool = ds2[half], ds2[~half]  # same patients, disjoint beats
    assert not set(test) & set(pool)
    return ds1, test, pool


def _leak_train(ds1, pool, f, n_total, seed):
    r = np.random.default_rng(seed)
    k = int(round(f * n_total))
    return np.concatenate([r.choice(ds1, n_total - k, replace=False), r.choice(pool, k, replace=False)]), k


def leakage(d, reps, n_total=40000, fracs=(0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5), seeds=(0, 1, 2)):
    """Fix the test beats. Vary only how many training beats come from the
    test patients, holding the training-set size constant."""
    y, rec = d["y"], d["rec"]
    ds1, test, pool = _leak_split(d)
    out = {"n_total": n_total, "n_test": int(len(test)), "curve": {}}
    _head(f"test beats fixed: {len(test):,} (half of every test patient's record)",
          f"training set fixed: {n_total:,} beats \u00b7 {len(seeds)} resamples per row",
          cols="features   from test patients      F1 S   F1 V")
    for mode in ("raw", "template"):
        s, t = reps[mode]
        rows = []
        for f in fracs:
            for sd in seeds:
                idx, k = _leak_train(ds1, pool, f, n_total, sd)
                g = M.fit_gbm(s[idx], t[idx], y[idx], seed=sd)
                p = M.predict_gbm(g, s[test], t[test]).argmax(1)
                rows.append({"frac": f, "seed": sd, "n_leaked": k,
                             "n_leaked_by_class": np.bincount(y[idx[n_total - k:]], minlength=4).tolist(),
                             **E.score(y[test], p), **E.patient_mean(y[test], p, rec[test])})
            last = [r_ for r_ in rows if r_["frac"] == f]
            m_ = {k_: np.mean([r_[k_] for r_ in last]) for k_ in ("f1_S", "f1_V")}
            print(f"{LABEL[mode]:<11}{f:>5.1%}  {last[0]['n_leaked']:>6,} beats      "
                  f"{m_['f1_S']:.2f}   {m_['f1_V']:.2f}", flush=True)
        out["curve"][mode] = rows
    _save("leakage", out)


def leakage_cnn(d, reps, n_total=40000):
    y, rec = d["y"], d["rec"]
    ds1, test, pool = _leak_split(d)
    out = {}
    for mode in ("raw", "template"):
        s, t = reps[mode]
        for f in (0, 0.5):
            idx, _ = _leak_train(ds1, pool, f, n_total, 0)
            net = M.fit_cnn(s[idx], t[idx], y[idx], seed=0)
            p = M.predict_cnn(net, s[test], t[test]).argmax(1)
            out[f"{mode}/{f}"] = {**E.score(y[test], p), **E.patient_mean(y[test], p, rec[test])}
            print("cnn", mode, f, round(out[f"{mode}/{f}"]["macro_f1"], 3), flush=True)
    _save("leakage_cnn", out)


# ----------------------------------------------------------------------------- E3
def cv(d, reps, k=11, seed=0):
    """Out-of-fold predictions for every beat under two fold constructions that
    differ only in what a fold is made of: whole subjects, or shuffled beats."""
    y, rec, sub = d["y"], d["rec"], d["subject"]
    rng = np.random.default_rng(seed)
    subjects = np.unique(sub)
    order = rng.permutation(len(subjects))
    fold_of_subject = {s_: i % k for i, s_ in zip(range(len(subjects)), subjects[order])}
    folds = {"patient": np.array([fold_of_subject[s_] for s_ in sub]),
             "beat": rng.integers(0, k, len(y))}
    out = {}
    _head(f"cross-validation \u00b7 {k} folds \u00b7 {len(subjects)} patients \u00b7 {len(y):,} beats",
          cols="features   test folds made of   acc     F1 S   F1 V")
    for mode in ("raw", "template"):
        s, t = reps[mode]
        for kind, fold in folds.items():
            pred = np.zeros(len(y), dtype=int)
            for i in range(k):
                tr, te = fold != i, fold == i
                if kind == "patient":
                    _assert_disjoint(d, tr, te)
                g = M.fit_gbm(s[tr], t[tr], y[tr])
                pred[te] = M.predict_gbm(g, s[te], t[te]).argmax(1)
            out[f"{mode}/{kind}"] = {"metrics": E.score(y, pred),
                                     "ci95": E.cluster_bootstrap(y, pred, sub),
                                     "per_record": E.per_group(y, pred, rec)}
            m_ = out[f"{mode}/{kind}"]["metrics"]
            name = "whole patients" if kind == "patient" else "shuffled beats"
            print(f"{LABEL[mode]:<11}{name:<21}{m_['acc']:.3f}   {m_['f1_S']:.2f}   {m_['f1_V']:.2f}", flush=True)
    pr = out["template/patient"]["per_record"]
    vals = np.array([v["f1_S"] for v in pr.values() if v["support"][1] >= 20])
    print(f"\nrelative features, supraventricular F1 per unseen patient")
    print(f"{len(vals)} patients with 20+ such beats: median {np.median(vals):.2f}, "
          f"{int((vals < 0.5).sum())} below 0.50")
    _save("cv", out)


# ----------------------------------------------------------------------------- E4
def subject(d, reps, seeds=(0, 1, 2, 3, 4)):
    """Record 202 (DS2) is the same person as record 201 (DS1)."""
    y, rec = d["y"], d["rec"]
    te202 = rec == "202"
    te_rest = np.isin(rec, C.DS2) & ~te202
    out = {}
    _head("records 201 and 202 are the same patient (MIT-BIH directory)",
          f"standard split: 201 in training, 202 in test \u00b7 {len(seeds)} seeds",
          cols="features   201 in training   F1 S on 202   F1 V on 202")
    for mode in ("raw", "template"):
        s, t = reps[mode]
        for name, train_recs in (("with_201", C.DS1), ("without_201", [r for r in C.DS1 if r != "201"])):
            tr = np.isin(rec, train_recs)
            rows = []
            for sd in seeds:
                g = M.fit_gbm(s[tr], t[tr], y[tr], seed=sd)
                rows.append({"on_202": E.score(y[te202], M.predict_gbm(g, s[te202], t[te202]).argmax(1)),
                             "on_rest": E.score(y[te_rest], M.predict_gbm(g, s[te_rest], t[te_rest]).argmax(1))})
            out[f"{mode}/{name}"] = rows
            g_ = lambda k_: np.mean([r_["on_202"][k_] for r_ in rows])
            print(f"{LABEL[mode]:<11}{'yes' if name == 'with_201' else 'no':<18}"
                  f"{g_('f1_S'):>11.2f}{g_('f1_V'):>14.2f}", flush=True)
    out["support_202"] = E.confusion(y[te202], y[te202]).sum(1).tolist()
    _save("subject", out)


# ----------------------------------------------------------------------------- E5
def identity(d, reps, per_record=600, seed=0):
    """Can a linear probe tell which patient a *normal* beat came from?
    44-way, chance = 2.3%. High accuracy means beats carry a patient signature
    that a classifier can exploit whenever train and test share patients."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    y, rec = d["y"], d["rec"]
    rng = np.random.default_rng(seed)
    idx = []
    for r in C.RECORDS:
        cand = np.flatnonzero((rec == r) & (y == 0))
        idx.append(rng.choice(cand, min(per_record, len(cand)), replace=False))
    idx = np.concatenate(idx)
    half = rng.random(len(idx)) < 0.5
    out = {"chance": 1 / len(C.RECORDS)}
    for mode in MODES:
        s, t = reps[mode]
        # morphology only: RR features are excluded so the probe sees waveform shape
        X = F.flat(s[idx], np.zeros((len(idx), 0), dtype=np.float32))
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[half], rec[idx][half])
        out[mode] = float((lda.predict(X[~half]) == rec[idx][~half]).mean())
        print("identity", mode, round(out[mode], 3), flush=True)
    _save("identity", out)


# ----------------------------------------------------------------------------- v2
def v2(d, reps=None):
    """The candidate selected by patient-wise CV inside DS1 (scripts/dev_ds1.py),
    evaluated on DS2 once. Appended to benchmark.json."""
    sel = json.load(open(os.path.join(RES, "dev_ds1.json")))["selected"]
    assert sel == "context-reg", "update v2() if the DS1 selection changes"
    y, rec = d["y"], d["rec"]
    tr, te = np.isin(rec, C.DS1), np.isin(rec, C.DS2)
    s, t = F.represent(d, "context")
    g = M.fit_gbm(s[tr], t[tr], y[tr], num_leaves=15, min_child_samples=100, colsample_bytree=0.3)
    p = M.predict_gbm(g, s[te], t[te]).argmax(1)
    out = json.load(open(os.path.join(RES, "benchmark.json")))
    out["gbm-v2/context"] = {"metrics": E.score(y[te], p), "ci95": E.cluster_bootstrap(y[te], p, rec[te]),
                             "confusion": E.confusion(y[te], p).tolist(),
                             "per_record": E.per_group(y[te], p, rec[te])}
    print("gbm-v2/context", {k: round(v, 3) for k, v in out["gbm-v2/context"]["metrics"].items()}, flush=True)
    _save("benchmark", out)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = D.load()
    if which == "v2":
        return v2(d)
    reps = _reps(d)
    jobs = {"benchmark": benchmark, "unseen": unseen, "leakage": leakage, "leakage_cnn": leakage_cnn, "cv": cv, "subject": subject, "identity": identity}
    names = [n for n in jobs if n not in ("unseen", "leakage_cnn")] if which == "all" else [which]
    for name in names:
        t0 = time.time()
        jobs[name](d, reps)
        print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
