"""Print the README tables straight from results/*.json, so the prose can be
checked against the numbers the code produced."""
import json

import numpy as np

R = lambda n: json.load(open(f"results/{n}.json"))
b, cv, lk, sub, idn, dev = R("benchmark"), R("cv"), R("leakage"), R("subject"), R("identity"), R("dev_ds1")
f = lambda x: f"{x:.2f}"
ci = lambda v, k: f"{v['metrics'][k]:.2f} [{v['ci95'][k][0]:.2f}, {v['ci95'][k][1]:.2f}]"

print("### DS1 -> DS2\n\n| Model | Representation | Accuracy | F1 N | F1 S [95% CI] | F1 V [95% CI] | F1 F |\n|---|---|---|---|---|---|---|")
names = {"raw": "raw", "pnorm": "patient-normalised", "template": "patient-relative", "context": "DS1-selected"}
for k, v in b.items():
    m, r = k.split("/")
    m = {"gbm": "GBM", "cnn": "CNN", "cnn+adabn": "CNN + AdaBN", "gbm-v2": "GBM"}[m]
    x = v["metrics"]
    print(f"| {m} | {names[r]} | {x['acc']:.3f} | {f(x['f1_N'])} | {ci(v, 'f1_S')} | {ci(v, 'f1_V')} | {f(x['f1_F'])} |")

print("\n### Patient-wise vs beat-wise 11-fold CV, all 44 records\n\n| Representation | Folds made of | Accuracy | Macro-F1 [95% CI] | F1 S | F1 V | F1 F |\n|---|---|---|---|---|---|---|")
for k, v in cv.items():
    r, kind = k.split("/")
    x = v["metrics"]
    print(f"| {names[r]} | {'whole patients' if kind == 'patient' else 'shuffled beats'} | {x['acc']:.3f} | {ci(v, 'macro_f1')} | {f(x['f1_S'])} | {f(x['f1_V'])} | {f(x['f1_F'])} |")
a, s = cv["template/patient"]["per_record"], cv["template/beat"]["per_record"]
for i, c in ((1, "S"), (2, "V")):
    recs = [r for r in a if a[r]["support"][i] >= 20]
    pa = np.array([a[r][f"f1_{c}"] for r in recs]); pb = np.array([s[r][f"f1_{c}"] for r in recs])
    print(f"\n{c}: {len(recs)} patients with >=20 beats. unseen: median {np.median(pa):.2f}, min {pa.min():.2f}, "
          f"{int((pa < 0.5).sum())} below 0.5. seen: median {np.median(pb):.2f}, min {pb.min():.2f}")

print("\n### Leakage curve (GBM, mean of 3 resamples)\n\n| Share from test patients | Beats | of which S | raw: S | raw: V | relative: S | relative: V | relative: S per-patient |\n|---|---|---|---|---|---|---|---|")
mu = lambda mode, fr, k: np.mean([r[k] for r in lk["curve"][mode] if r["frac"] == fr])
for fr in sorted({r["frac"] for r in lk["curve"]["raw"]}):
    print(f"| {fr:.1%} | {int(mu('raw', fr, 'n_leaked'))} | {np.mean([r['n_leaked_by_class'][1] for r in lk['curve']['raw'] if r['frac'] == fr]):.0f} | "
          f"{f(mu('raw', fr, 'f1_S'))} | {f(mu('raw', fr, 'f1_V'))} | {f(mu('template', fr, 'f1_S'))} | {f(mu('template', fr, 'f1_V'))} | {f(mu('template', fr, 'pm_f1_S'))} |")
print("\nCNN:", {k: {m: round(v[m], 2) for m in ("acc", "f1_S", "f1_V")} for k, v in R("leakage_cnn").items()})

print("\n### Record 202 with and without its twin 201 in training (GBM, 5 seeds)\n\n| Representation | 201 in training | Accuracy on 202 | F1 S on 202 | F1 V on 202 |\n|---|---|---|---|---|")
for k, rows in sub.items():
    if k.startswith("support"):
        continue
    r, w = k.split("/")
    g = lambda m: np.mean([x["on_202"][m] for x in rows])
    print(f"| {names[r]} | {'yes' if w == 'with_201' else 'no'} | {g('acc'):.3f} | {f(g('f1_S'))} | {f(g('f1_V'))} |")
print("\nidentity probe:", {k: round(v, 3) for k, v in idn.items()})
print("DS1 dev:", {k: round(v["macro_f1"], 3) for k, v in dev.items() if k != "selected"}, "->", dev["selected"])
pr = b["gbm/template"]["per_record"]
tot = sum(v["support"][1] for v in pr.values())
print("232 share of DS2 S:", pr["232"]["support"][1], "/", tot, "S sens on 232:", {k: round(b[k]["per_record"]["232"]["se_S"], 2) for k in b})
