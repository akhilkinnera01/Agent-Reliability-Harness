"""
Turn the persisted comparison results into figures for the README.

Reads benchmarks/results/*.json (written by run_comparison.py) and writes PNGs to
benchmarks/plots/:
  - roc.png          ROC curve per model (groundedness), AUC in the legend
  - distributions.png  score distribution, grounded vs hallucinated, per model
  - comparison.png   AUC / F1 bars per model, grouped by provider

    python -m benchmarks.plot_results
"""

import json
import pathlib

import numpy as np

RESULTS = pathlib.Path(__file__).parent / "results"
PLOTS = pathlib.Path(__file__).parent / "plots"

# fixed display order (within-provider evolution / size, left to right)
ORDER = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4.1-mini",
         "gemini/gemini-2.5-flash-lite", "gemini/gemini-2.5-flash",
         "anthropic/claude-haiku-4-5"]
PROVIDER_COLOR = {"openai": "#10a37f", "google": "#4285f4", "anthropic": "#d97706"}


def provider(model: str) -> str:
    if "gpt" in model:
        return "openai"
    if "gemini" in model:
        return "google"
    return "anthropic"


def short(model: str) -> str:
    return model.split("/")[-1]


def load() -> list:
    out = []
    for f in RESULTS.glob("*.json"):
        out.append(json.loads(f.read_text()))
    out.sort(key=lambda r: ORDER.index(r["model"]) if r["model"] in ORDER else 99)
    return out


def auc_ci(data, rng, B=2000):
    """Bootstrap 95% CI for each model's AUC; yields (lo, hi, auc)."""
    from benchmarks import metrics
    for r in data:
        s = np.array(r["scores"]); y = np.array(r["labels"]); n = len(y)
        boot = [metrics.roc_auc(s[i].tolist(), y[i].tolist())
                for i in (rng.integers(0, n, n) for _ in range(B))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        yield lo, hi, r["auc"]


def roc_points(scores, labels):
    """Sweep thresholds high->low, return (fpr, tpr) arrays."""
    s = np.array(scores)
    y = np.array(labels)
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

    data = load()
    if not data:
        raise SystemExit("no results/*.json yet")
    PLOTS.mkdir(exist_ok=True)

    # 1. ROC
    plt.figure(figsize=(6, 6))
    for r in data:
        fpr, tpr = roc_points(r["scores"], r["labels"])
        c = PROVIDER_COLOR[provider(r["model"])]
        plt.plot(fpr, tpr, label=f"{short(r['model'])} (AUC {r['auc']:.2f})", color=c, alpha=0.85)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Groundedness ROC — HaluEval QA (n=%d)" % data[0]["n"])
    plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(PLOTS / "roc.png", dpi=130); plt.close()

    # 2. score distributions (grid)
    n = len(data)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, r in zip(axes.flat, data):
        s = np.array(r["scores"]); y = np.array(r["labels"])
        bins = np.linspace(0, 1, 11)
        ax.hist(s[y == 1], bins=bins, alpha=0.6, label="grounded", color="#2a9d8f")
        ax.hist(s[y == 0], bins=bins, alpha=0.6, label="hallucinated", color="#e76f51")
        ax.set_title(short(r["model"]), fontsize=9)
        ax.set_xlabel("groundedness score"); ax.legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Score separation: grounded vs hallucinated")
    fig.tight_layout()
    fig.savefig(PLOTS / "distributions.png", dpi=130); plt.close(fig)

    # 3. AUC / F1 bars grouped by model, with bootstrap 95% CI error bars on AUC
    labels = [short(r["model"]) for r in data]
    auc = [r["auc"] for r in data]
    f1 = [r["f1"] for r in data]
    colors = [PROVIDER_COLOR[provider(r["model"])] for r in data]
    rng = np.random.default_rng(0)
    yerr = [[], []]
    for r in auc_ci(data, rng):
        lo, hi, a = r
        yerr[0].append(a - lo); yerr[1].append(hi - a)
    x = np.arange(len(data)); w = 0.38
    plt.figure(figsize=(max(7, len(data) * 1.3), 5))
    plt.bar(x - w / 2, auc, w, label="AUC (95% CI)", color=colors,
            yerr=yerr, capsize=4, ecolor="#333")
    plt.bar(x + w / 2, f1, w, label="F1", color=colors, alpha=0.5)
    for i, (a, f) in enumerate(zip(auc, f1)):
        plt.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    plt.xticks(x, labels, rotation=30, ha="right", fontsize=8)
    plt.ylim(0, 1.05); plt.ylabel("score")
    plt.title("Groundedness AUC / F1 by model (color = provider)")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS / "comparison.png", dpi=130); plt.close()

    print("wrote", *[p.name for p in PLOTS.glob("*.png")])


if __name__ == "__main__":
    main()
