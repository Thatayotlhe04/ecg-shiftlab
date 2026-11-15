"""Five 1080x1350 cards for a LinkedIn post, plus a PDF carousel.
Every number is read from results/*.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, INK, INK2, MUTE, RULE = "#141413", "#f2f1ee", "#c3c2b7", "#8a897f", "#2f2f2c"
BLUE, ORANGE = "#3987e5", "#d95926"
plt.rcParams.update({"font.family": "Inter", "text.color": INK, "figure.facecolor": BG,
                     "savefig.facecolor": BG, "axes.facecolor": BG})
R = lambda n: json.load(open(f"results/{n}.json"))
OUT = "figures/linkedin"
os.makedirs(OUT, exist_ok=True)
W, H, DPI = 7.2, 9.0, 150  # 1080 x 1350
N = 5


def card(i, headline, deck):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    ax.text(0.075, 0.945, "ECG SHIFTLAB", fontsize=10.5, color=MUTE, fontweight="semibold")
    ax.text(0.925, 0.945, f"{i:02d} / {N:02d}", fontsize=10.5, color=MUTE, ha="right")
    ax.plot([0.075, 0.925], [0.928, 0.928], color=RULE, lw=1)
    ax.text(0.075, 0.895, headline, fontsize=29, fontweight="bold", va="top", linespacing=1.12,
            fontfamily="Inter Display")
    n_lines = headline.count("\n") + 1
    ax.text(0.075, 0.895 - 0.058 * n_lines - 0.012, deck, fontsize=13.5, color=INK2, va="top", linespacing=1.45)
    ax.plot([0.075, 0.925], [0.072, 0.072], color=RULE, lw=1)
    ax.text(0.075, 0.045, "MIT-BIH Arrhythmia Database", fontsize=9.5, color=MUTE)
    ax.text(0.925, 0.045, "github.com/Thatayotlhe04/ecg-shiftlab", fontsize=9.5, color=MUTE, ha="right")
    return fig, ax


def save(fig, i):
    path = f"{OUT}/card_{i}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def c1():
    cv = R("cv")
    seen, unseen = cv["template/beat"]["metrics"], cv["template/patient"]["metrics"]
    fig, ax = card(1, "99.5% accurate.\nUntil the patient is new.",
                   "Same model. Same heartbeats. One change: the test folds are\n"
                   "whole patients instead of shuffled beats.")
    ax.text(0.56, 0.615, "Patients seen\nin training", fontsize=12, color=INK2, ha="center", va="bottom", linespacing=1.3)
    ax.text(0.81, 0.615, "Patients\nnever seen", fontsize=12, color=INK2, ha="center", va="bottom", linespacing=1.3)
    ax.plot([0.47, 0.65], [0.602, 0.602], color=BLUE, lw=3, solid_capstyle="round")
    ax.plot([0.72, 0.90], [0.602, 0.602], color=ORANGE, lw=3, solid_capstyle="round")
    rows = [("Accuracy", f"{seen['acc']:.1%}", f"{unseen['acc']:.1%}"),
            ("Ventricular\nectopic beats, F1", f"{seen['f1_V']:.2f}", f"{unseen['f1_V']:.2f}"),
            ("Supraventricular\nectopic beats, F1", f"{seen['f1_S']:.2f}", f"{unseen['f1_S']:.2f}")]
    for k, (name, a, b) in enumerate(rows):
        y = 0.52 - k * 0.125
        ax.text(0.075, y, name, fontsize=14, color=INK2, va="center", linespacing=1.3)
        ax.text(0.56, y, a, fontsize=34, fontweight="bold", ha="center", va="center", fontfamily="Inter Display")
        ax.text(0.81, y, b, fontsize=34, fontweight="bold", ha="center", va="center", fontfamily="Inter Display")
        ax.plot([0.075, 0.925], [y - 0.0625, y - 0.0625], color=RULE, lw=1)
    ax.text(0.075, 0.125, "Gradient-boosted trees, 11-fold cross-validation, every beat predicted out of fold.\n"
            "A 95% accurate model that misses most of one arrhythmia class.", fontsize=11, color=MUTE,
            va="center", linespacing=1.5)
    return save(fig, 1)


def c2():
    d = R("leakage")["curve"]["raw"]
    xs = sorted({r["frac"] for r in d})
    mu = [np.mean([r["f1_S"] for r in d if r["frac"] == f]) for f in xs]
    fig, ax = card(2, "200 heartbeats are\nenough to fake it.",
                   "Test beats fixed. Training set fixed at 40,000 beats. Only the share\n"
                   "of training beats that come from the test patients changes.")
    p = fig.add_axes([0.13, 0.215, 0.76, 0.435])
    pos = np.arange(len(xs))
    for s in ("top", "right", "left"):
        p.spines[s].set_visible(False)
    p.spines["bottom"].set_color(RULE)
    p.grid(axis="y", color=RULE, lw=0.8), p.set_axisbelow(True)
    p.plot(pos, mu, color=ORANGE, lw=3, marker="o", ms=10, mec=BG, mew=2.5, solid_capstyle="round")
    p.set_ylim(0, 1.05), p.set_xlim(-0.5, len(xs) - 0.5)
    p.set_xticks(pos, ["0", "0.5%", "1%", "2%", "5%", "10%", "25%", "50%"])
    p.tick_params(colors=MUTE, length=0, labelsize=11.5)
    p.set_ylabel("Supraventricular F1 on the same test beats", color=INK2, fontsize=11.5, labelpad=10)
    p.set_xlabel("Share of training beats taken from the test patients", color=INK2, fontsize=11.5, labelpad=10)
    p.annotate(f"{mu[0]:.2f}", (0, mu[0]), xytext=(14, -4), textcoords="offset points", fontsize=17, fontweight="bold")
    p.annotate("no leakage", (0, mu[0]), xytext=(14, -22), textcoords="offset points", fontsize=10.5, color=INK2)
    p.annotate(f"{mu[1]:.2f}", (1, mu[1]), xytext=(4, -34), textcoords="offset points", fontsize=17, fontweight="bold")
    p.annotate("200 leaked beats,\nabout 9 per patient", (1, mu[1]), xytext=(4, -70), textcoords="offset points",
               fontsize=10.5, color=INK2, linespacing=1.4)
    p.annotate(f"{mu[-1]:.2f}", (pos[-1], mu[-1]), xytext=(-6, -32), textcoords="offset points", fontsize=17,
               fontweight="bold", ha="right")
    ax.text(0.075, 0.105, "A random beat-level split sits at the right-hand end of this curve.", fontsize=11, color=MUTE)
    return save(fig, 2)


def c3():
    cv = R("cv")
    a, b = cv["template/patient"]["per_record"], cv["template/beat"]["per_record"]
    recs = sorted([r for r in a if a[r]["support"][1] >= 20], key=lambda r: a[r]["f1_S"])
    low = sum(a[r]["f1_S"] < 0.5 for r in recs)
    fig, ax = card(3, "The average hides it.",
                   f"Supraventricular beats, one row per patient. {low} of {len(recs)} patients score\n"
                   "below 0.5 when the model has never seen them.")
    ax.scatter([0.085], [0.725], s=90, color=ORANGE), ax.text(0.105, 0.725, "patient never seen", fontsize=11.5, color=INK2, va="center")
    ax.scatter([0.40], [0.725], s=90, color=BLUE), ax.text(0.42, 0.725, "patient seen in training", fontsize=11.5, color=INK2, va="center")
    p = fig.add_axes([0.19, 0.205, 0.71, 0.485])
    for s in ("top", "right", "left"):
        p.spines[s].set_visible(False)
    p.spines["bottom"].set_color(RULE)
    p.grid(axis="x", color=RULE, lw=0.8), p.set_axisbelow(True)
    y = np.arange(len(recs))
    for i, r in enumerate(recs):
        p.plot([a[r]["f1_S"], b[r]["f1_S"]], [i, i], color="#3a3a37", lw=2.5, solid_capstyle="round", zorder=1)
    p.scatter([b[r]["f1_S"] for r in recs], y, s=110, color=BLUE, ec=BG, lw=2, zorder=3)
    p.scatter([a[r]["f1_S"] for r in recs], y, s=110, color=ORANGE, ec=BG, lw=2, zorder=3)
    p.set_yticks(y, [f"patient {r}" for r in recs])
    p.set_xlim(-0.04, 1.04), p.set_ylim(-0.7, len(recs) - 0.3)
    p.tick_params(colors=MUTE, length=0, labelsize=10.5)
    p.set_xlabel("F1 for supraventricular beats within that patient", color=INK2, fontsize=11.5, labelpad=10)
    ax.text(0.075, 0.105, "Patients with at least 20 supraventricular beats. Same model in both conditions.", fontsize=11, color=MUTE)
    return save(fig, 3)


def c4():
    s = R("subject")
    g = lambda k: np.mean([x["on_202"]["f1_S"] for x in s[k]])
    w, wo = g("template/with_201"), g("template/without_201")
    fig, ax = card(4, "The standard split has\none person on both sides.",
                   "Records 201 and 202 are the same patient. The benchmark partition\n"
                   "used across the literature trains on 201 and tests on 202.")
    ax.text(0.075, 0.635, "SUPRAVENTRICULAR F1 ON RECORD 202", fontsize=10.5, color=MUTE, fontweight="semibold")
    for x, val, col, lab in ((0.075, w, BLUE, "with record 201\nin training, as published"),
                             (0.52, wo, ORANGE, "with record 201\nremoved from training")):
        ax.text(x, 0.50, f"{val:.2f}", fontsize=92, fontweight="bold", va="center", fontfamily="Inter Display")
        ax.plot([x, x + 0.33], [0.405, 0.405], color=col, lw=3, solid_capstyle="round")
        ax.text(x, 0.385, lab, fontsize=13, color=INK2, va="top", linespacing=1.4)
    ax.plot([0.075, 0.925], [0.275, 0.275], color=RULE, lw=1)
    ax.text(0.075, 0.25, "The pooled benchmark score barely moves, because record 202\n"
            "is small. For that patient, the reported number and the honest\n"
            "number are not the same measurement.", fontsize=13, color=INK2, va="top", linespacing=1.5)
    ax.text(0.075, 0.105, "Same-subject note: MIT-BIH database directory, PhysioNet. Mean of 5 seeds.", fontsize=11, color=MUTE)
    return save(fig, 4)


def c5():
    b, cv = R("benchmark")["gbm/template"], R("cv")["template/patient"]["per_record"]
    lo, hi = b["ci95"]["f1_V"]
    worst = min(v["f1_V"] for v in cv.values() if v["support"][2] >= 20)
    nrec = sum(v["support"][2] >= 20 for v in cv.values())
    fig, ax = card(5, "What held up.",
                   "Ventricular ectopic beats transfer to patients the model\nhas never seen.")
    ax.text(0.075, 0.635, "VENTRICULAR F1, 22 UNSEEN RECORDS", fontsize=10.5, color=MUTE, fontweight="semibold")
    ax.text(0.075, 0.535, f"{b['metrics']['f1_V']:.2f}", fontsize=92, fontweight="bold", va="center", fontfamily="Inter Display")
    ax.text(0.47, 0.555, f"95% interval {lo:.2f} to {hi:.2f},\nresampled over patients", fontsize=13, color=INK2, va="center", linespacing=1.4)
    ax.text(0.47, 0.495, f"No patient below {worst:.2f} ({nrec} patients)", fontsize=13, color=INK2, va="center")
    ax.plot([0.075, 0.925], [0.44, 0.44], color=RULE, lw=1)
    ax.text(0.075, 0.415, "HOW IT WAS MEASURED", fontsize=10.5, color=MUTE, fontweight="semibold", va="top")
    lines = ["Patients, not beats, are the unit of evidence: every interval\nresamples whole patients.",
             "Model settings were fixed before the test patients were touched.",
             "Every number here is printed from result files in the repository."]
    y = 0.375
    for t in lines:
        ax.text(0.075, y, "–", fontsize=13, color=MUTE, va="top")
        ax.text(0.105, y, t, fontsize=13, color=INK2, va="top", linespacing=1.45)
        y -= 0.036 * (t.count("\n") + 1) + 0.018
    ax.text(0.075, 0.125, "The inter-patient protocol is from de Chazal et al., 2004. This project measures\n"
            "the size of the effect; it does not claim to have discovered it.", fontsize=11, color=MUTE,
            va="center", linespacing=1.5)
    return save(fig, 5)


if __name__ == "__main__":
    paths = [f() for f in (c1, c2, c3, c4, c5)]
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    ims[0].save(f"{OUT}/ecg_shiftlab_carousel.pdf", save_all=True, append_images=ims[1:], resolution=DPI)
    print(paths, ims[0].size)
