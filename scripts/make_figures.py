"""Render figures/*.png from results/*.json. Dark, quiet, two hues at most."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, INK2, MUTE, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#8a897f", "#2e2e2c"
BLUE, ORANGE = "#3987e5", "#d95926"   # validated pair for the dark surface
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})
R = lambda n: json.load(open(f"results/{n}.json"))
os.makedirs("figures", exist_ok=True)


def head(fig, title, sub):
    h = fig.get_figheight()
    fig.text(0.04, 1 - 0.18 / h, title, fontsize=14, fontweight="bold", va="top")
    fig.text(0.04, 1 - 0.50 / h, sub, fontsize=10, color=INK2, va="top", linespacing=1.4)


def leakage():
    d = R("leakage")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.7), sharey=True)
    xs = sorted({r["frac"] for r in d["curve"]["raw"]})
    pos = np.arange(len(xs))
    for ax, (key, name) in zip(axes, [("f1_S", "S, beats pooled"), ("pm_f1_S", "S, averaged per patient"),
                                      ("f1_V", "V, beats pooled")]):
        ends = []
        for mode, col, lab in (("raw", ORANGE, "raw beats"), ("template", BLUE, "patient-relative")):
            rows = d["curve"][mode]
            mu = np.array([np.mean([r[key] for r in rows if r["frac"] == f]) for f in xs])
            lo = np.array([np.min([r[key] for r in rows if r["frac"] == f]) for f in xs])
            hi = np.array([np.max([r[key] for r in rows if r["frac"] == f]) for f in xs])
            ax.fill_between(pos, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(pos, mu, color=col, lw=2, marker="o", ms=6, mec=BG, mew=1.5, label=lab)
            ends.append(mu)
        for j, (x, dx, ha) in enumerate(((pos[0], -7, "right"), (pos[-1], 7, "left"))):
            v = sorted(m[0 if j == 0 else -1] for m in ends)
            if v[1] - v[0] < 0.06:   # one label when the two series coincide
                ax.annotate(f"{np.mean(v):.2f}", (x, np.mean(v)), xytext=(dx, 0), textcoords="offset points",
                            ha=ha, va="center", color=INK2, fontsize=9)
            else:
                for val in v:
                    ax.annotate(f"{val:.2f}", (x, val), xytext=(dx, 0), textcoords="offset points",
                                ha=ha, va="center", color=INK2, fontsize=9)
        ax.set_title(name)
        ax.set_xticks(pos, ["0", "0.5%", "1%", "2%", "5%", "10%", "25%", "50%"])
        ax.set_xlim(-0.9, len(xs) - 0.1)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("F1 on the same held-out beats")
    axes[1].set_xlabel("share of the 40,000 training beats drawn from the test patients")
    axes[0].legend(loc="lower right", fontsize=9)
    head(fig, "A trace of the test patients in training is enough",
         "Same test beats, same training-set size; only the share of training beats taken from the test patients changes.\n"
         "0.5% is 200 beats, about 9 per test patient. GBM, three resamples (band = range).")
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.07, right=0.97, wspace=0.08)
    fig.savefig("figures/leakage_curve.png", dpi=200)
    plt.close(fig)


def per_patient():
    d = R("cv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    for ax, (ci, key, name) in zip(axes, [(1, "f1_S", "Supraventricular (S)"), (2, "f1_V", "Ventricular (V)")]):
        a, b = d["template/patient"]["per_record"], d["template/beat"]["per_record"]
        recs = [r for r in a if a[r]["support"][ci] >= 20]
        recs.sort(key=lambda r: a[r][key])
        y = np.arange(len(recs))
        for i, r in enumerate(recs):
            ax.plot([a[r][key], b[r][key]], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter([b[r][key] for r in recs], y, s=46, color=BLUE, ec=BG, lw=1.5, zorder=3,
                   label="patient seen in training (beat-wise folds)")
        ax.scatter([a[r][key] for r in recs], y, s=46, color=ORANGE, ec=BG, lw=1.5, zorder=3,
                   label="patient unseen (patient-wise folds)")
        ax.set_yticks(y, [f"{r}  n={a[r]['support'][ci]}" for r in recs], fontsize=8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(f"{name}: records with at least 20 such beats")
        ax.set_xlabel("F1 within the record")
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0.035, 0.885), ncol=2, fontsize=9,
               handletextpad=0.3, columnspacing=1.6)
    head(fig, "The gap is not uniform: some patients collapse, others are untouched",
         "Every beat predicted out-of-fold by the same model (GBM, patient-relative features); "
         "only the fold construction differs. 11 folds, 43 subjects.")
    fig.legends[0].set_bbox_to_anchor((0.035, 1 - 0.78 / fig.get_figheight()))
    fig.subplots_adjust(top=0.78, bottom=0.09, left=0.09, right=0.97, wspace=0.28)
    fig.savefig("figures/per_patient_gap.png", dpi=200)
    plt.close(fig)


def benchmark():
    d = R("benchmark")
    order = [k for k in ("gbm/raw", "gbm/pnorm", "gbm/template", "cnn/raw", "cnn/pnorm", "cnn/template",
                         "cnn+adabn/raw", "cnn+adabn/pnorm", "cnn+adabn/template", "gbm-v2/context") if k in d]
    names = {"raw": "raw", "pnorm": "patient-normalised", "template": "patient-relative",
             "context": "DS1-selected"}
    lab = lambda k: f"{k.split('/')[0].upper().replace('+ADABN', ' + AdaBN').replace('-V2', '')} · {names[k.split('/')[1]]}"
    fig, axes = plt.subplots(1, 4, figsize=(11, 5.3), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, c in zip(axes, ["N", "S", "V", "F"]):
        for yi, k in zip(y, order):
            m, ci = d[k]["metrics"][f"f1_{c}"], d[k]["ci95"][f"f1_{c}"]
            ax.plot(ci, [yi, yi], color=MUTE, lw=1.5, solid_capstyle="round", zorder=2)
            ax.scatter([m], [yi], s=44, color=BLUE, ec=BG, lw=1.5, zorder=3)
            ax.annotate(f"{m:.2f}", (1.0, yi), xytext=(0, 7), textcoords="offset points", ha="right",
                        fontsize=8, color=INK2)
        ax.set_title({"N": "Normal", "S": "Supraventricular", "V": "Ventricular", "F": "Fusion"}[c])
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(order) - 0.3)
    axes[0].set_yticks(y, [lab(k) for k in order], fontsize=9)
    fig.text(0.5, 0.03, "F1 on DS2 (22 unseen records). Line = 95% interval from resampling patients.",
             ha="center", color=INK2, fontsize=9)
    head(fig, "Unseen patients: ventricular beats transfer, supraventricular beats do not",
         "Train on DS1, test on DS2 (de Chazal split). Intervals are wide because the unit of evidence is the patient, not the beat.")
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.25, right=0.97, wspace=0.1)
    fig.savefig("figures/benchmark_ds2.png", dpi=200)
    plt.close(fig)


def support():
    """Where do the DS2 arrhythmic beats live? One patient can own a class."""
    d = R("benchmark")["gbm/template"]["per_record"]
    recs = sorted(d)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, (ci, name) in zip(axes, [(1, "Supraventricular (S)"), (2, "Ventricular (V)")]):
        n = np.array([d[r]["support"][ci] for r in recs])
        o = np.argsort(-n)
        ax.bar(np.arange(len(recs)), n[o] / n.sum(), width=0.62, color=BLUE)
        ax.set_xticks(np.arange(len(recs)), [recs[i] for i in o], rotation=90, fontsize=8)
        ax.set_title(f"{name}: share of DS2 beats by record")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        if ci == 1:
            ax.annotate(f"record {recs[o[0]]}: {n[o[0]] / n.sum():.0%} of all S beats", (0, n[o[0]] / n.sum()),
                        xytext=(14, -4), textcoords="offset points", color=INK2, fontsize=9, va="top")
    head(fig, "A benchmark class can be one patient",
         "Any aggregate S score on DS2 is mostly a statement about a single record.")
    fig.subplots_adjust(top=0.70, bottom=0.15, left=0.06, right=0.98, wspace=0.16)
    fig.savefig("figures/class_support.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for f in (leakage, per_patient, benchmark, support):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
