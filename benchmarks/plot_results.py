"""
Turn the persisted comparison results into figures for the README.

Reads benchmarks/results/*.json (written by run_comparison.py) and writes PNGs to
benchmarks/plots/:
  - comparison.png   AUC (with 95% CI) and F1 per model, sorted
  - roc.png          ROC curve per model, AUC in the legend
  - distributions.png  score distribution, grounded vs hallucinated, per model

    python -m benchmarks.plot_results
"""

import json
import pathlib

import numpy as np

RESULTS = pathlib.Path(__file__).parent / "results"
PLOTS = pathlib.Path(__file__).parent / "plots"

# fixed display order (within-provider, left to right)
ORDER = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4.1-mini",
         "gemini/gemini-2.5-flash-lite", "gemini/gemini-2.5-flash",
         "anthropic/claude-haiku-4-5"]

# one distinct colour per model, grouped into provider hue families
MODEL_COLOR = {
    "gpt-3.5-turbo": "#0a6e57",
    "gpt-4o-mini": "#13a884",
    "gpt-4.1-mini": "#73cdb6",
    "gemini/gemini-2.5-flash-lite": "#8ab4f8",
    "gemini/gemini-2.5-flash": "#1a73e8",
    "anthropic/claude-haiku-4-5": "#e8870c",
}
GROUNDED, HALLUC = "#2a9d8f", "#e76f51"


def short(model):
    return model.split("/")[-1]


def color(model):
    return MODEL_COLOR.get(model, "#888888")


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#cfcfcf", "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": "#ededed", "grid.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "font.family": "DejaVu Sans",
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 12,
        "axes.labelsize": 11, "axes.labelcolor": "#222",
        "legend.fontsize": 9, "legend.frameon": False,
        "xtick.color": "#555", "ytick.color": "#555", "text.color": "#222",
    })


def load():
    out = [json.loads(f.read_text()) for f in RESULTS.glob("*.json")]
    out.sort(key=lambda r: ORDER.index(r["model"]) if r["model"] in ORDER else 99)
    return out


def auc_ci(scores, labels, rng, B=2000):
    from benchmarks import metrics
    s = np.array(scores); y = np.array(labels); n = len(y)
    boot = [metrics.roc_auc(s[i].tolist(), y[i].tolist())
            for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(boot, [2.5, 97.5])


def roc_points(scores, labels):
    s = np.array(scores); y = np.array(labels)
    P, N = y.sum(), (1 - y).sum()
    fpr, tpr = [0.0], [0.0]
    for thr in sorted(set(s), reverse=True):
        pred = s >= thr
        tpr.append((pred & (y == 1)).sum() / max(P, 1))
        fpr.append((pred & (y == 0)).sum() / max(N, 1))
    fpr.append(1.0); tpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()

    data = load()
    if not data:
        raise SystemExit("no results/*.json yet")
    PLOTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    n_cases = data[0]["n"]

    # ---- 1. comparison: horizontal AUC bars (sorted, with 95% CI) + F1 markers ----
    srt = sorted(data, key=lambda r: r["auc"])  # ascending -> best on top
    names = [short(r["model"]) for r in srt]
    aucs = [r["auc"] for r in srt]
    cols = [color(r["model"]) for r in srt]
    err = np.array([[a - auc_ci(r["scores"], r["labels"], rng)[0],
                     auc_ci(r["scores"], r["labels"], rng)[1] - a]
                    for r, a in zip(srt, aucs)]).T
    y = np.arange(len(srt))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(y, aucs, color=cols, height=0.62, zorder=3,
            xerr=err, error_kw=dict(ecolor="#555", elinewidth=1.3, capsize=4, zorder=4))
    ax.scatter([r["f1"] for r in srt], y, marker="D", s=34, color="#222", zorder=5)
    for yi, a in zip(y, aucs):
        ax.text(a + err[1][yi] + 0.012, yi, f"{a:.2f}", va="center", fontsize=10, color="#222")
    ax.axvline(0.5, color="#bbb", ls="--", lw=1, zorder=1)
    ax.text(0.5, len(srt) - 0.4, "chance", color="#999", fontsize=8, ha="center")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0.45, 1.0); ax.set_xlabel("Groundedness AUC  (bars, 95% CI)   •   F1 (◆)")
    ax.set_title("Hallucination-detection skill by model")
    ax.set_axisbelow(True); ax.grid(axis="y", visible=False)
    fig.text(0.5, 0.005, f"HaluEval QA · n={n_cases} · colour = provider",
             ha="center", fontsize=8, color="#888")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(PLOTS / "comparison.png", dpi=200); plt.close(fig)

    # ---- 2. ROC ----
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for r in data:
        fpr, tpr = roc_points(r["scores"], r["labels"])
        ax.plot(fpr, tpr, lw=2.4, alpha=0.92, color=color(r["model"]),
                label=f"{short(r['model'])}  ·  AUC {r['auc']:.2f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#bbb")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Groundedness ROC")
    ax.legend(loc="lower right", title=f"HaluEval QA · n={n_cases}", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "roc.png", dpi=200); plt.close(fig)

    # ---- 3. score distributions (small multiples, shared style, density) ----
    ncol = 3
    nrow = (len(data) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.0 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, r in zip(axes.flat, data):
        s = np.array(r["scores"]); yl = np.array(r["labels"])
        ax.hist(s[yl == 1], bins=bins, density=True, alpha=0.7, color=GROUNDED,
                edgecolor="white", linewidth=0.6)
        ax.hist(s[yl == 0], bins=bins, density=True, alpha=0.6, color=HALLUC,
                edgecolor="white", linewidth=0.6)
        ax.set_title(f"{short(r['model'])}   (AUC {r['auc']:.2f})", fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(data):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("groundedness score")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUNDED, alpha=0.7),
               plt.Rectangle((0, 0), 1, 1, color=HALLUC, alpha=0.6)]
    fig.legend(handles, ["grounded answers", "hallucinated answers"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Score separation: grounded vs hallucinated", y=1.04, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "distributions.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print("wrote", *sorted(p.name for p in PLOTS.glob("*.png")))


if __name__ == "__main__":
    main()
