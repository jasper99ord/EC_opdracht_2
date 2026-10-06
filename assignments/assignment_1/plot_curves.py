import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "__data__" / "ea_selection_compare"

METHODS = ("fitness", "age", "random")
METRIC = "best_so_far"  
THRESHOLD = 13.0  


def load_curves(method: str, metric: str = METRIC) -> np.ndarray:
    """Return an array of shape (n_seeds, n_gens + 1) for one method."""
    rows = []
    for path in sorted(DATA.glob(f"curve_{method}_seed*.json")):
        log = json.loads(path.read_text())
        if not isinstance(log, dict):  
            print(f"skipping old-format file {path.name}")
            continue
        rows.append(log[metric])
    if not rows:
        msg = f"no runs found for method={method!r} in {DATA}"
        raise FileNotFoundError(msg)
    return np.array(rows, dtype=float)


def gens_to_threshold(curve: np.ndarray, threshold: float) -> float:
    """First generation at which the curve drops below threshold (nan if never)."""
    hits = np.flatnonzero(curve < threshold)
    return float(hits[0]) if hits.size else float("nan")


def main() -> None:
    curves = {m: load_curves(m) for m in METHODS}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHODS:
        arr = curves[method]
        gens = np.arange(arr.shape[1])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1)
        ax.plot(gens, mean, label=f"{method} (n={arr.shape[0]})")
        ax.fill_between(gens, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("generation")
    ax.set_ylabel("best-so-far fitness (lower is better)")
    ax.set_title("Mean ± 1 std over independent runs")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = DATA / "selection_compare.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    lines = []
    for method in METHODS:
        arr = curves[method]
        final = arr[:, -1]
        ttt = np.array([gens_to_threshold(c, THRESHOLD) for c in arr])
        lines.append(
            f"{method:8s} n={arr.shape[0]}  final best-so-far: "
            f"{final.mean():.3f} ± {final.std(ddof=1):.3f}  "
            f"(per seed: {np.round(final, 2).tolist()})  "
            f"gens to < {THRESHOLD}: {ttt.tolist()}"
        )
    u = mannwhitneyu(curves["fitness"][:, -1], curves["age"][:, -1], alternative="two-sided")
    lines.append(
        f"Mann-Whitney U (final best-so-far, fitness vs age): "
        f"U={u.statistic:.1f}, p={u.pvalue:.4f}"
    )
    summary = "\n".join(lines)
    print(summary)
    (DATA / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
