import numpy as np
import math_statistics as ms
from data_processing import time_to_hours

def _hour_ticks(ax):
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=45, fontsize=8)

def plot_timeseries(ax, times, values, raw=None, bounds=None, types=None):
    ax.clear()
    xs = [time_to_hours(t) for t in times]
    if raw is not None:
        rt, rv = raw
        rx = [time_to_hours(t) for t in rt]
        for i, v in enumerate(rv):
            if np.isnan(v):
                ax.axvline(rx[i], color="gray", alpha=.35, lw=6)
        ax.plot(rx, rv, color="lightgray", marker=".", lw=1, label="вихідні дані", zorder=1)
        if types:
            for kind, col, mk, lab in (("technical", "red", "X", "технічна аномалія"),
                                       ("real", "orange", "D", "реальна подія")):
                idx = [i for i, k in types.items() if k == kind]
                if idx:
                    ax.scatter([rx[i] for i in idx], [rv[i] for i in idx], c=col, marker=mk,
                               s=70, zorder=4, label=lab)
    ax.plot(xs, values, color="tab:blue", marker="o", ms=3, lw=1.4, label="поточні дані", zorder=2)
    if bounds:
        ax.axhline(bounds[0], color="green", ls="--", lw=.8, label="межі аномалій")
        ax.axhline(bounds[1], color="green", ls="--", lw=.8)
    v = np.asarray(values, float)
    if len(v) and not np.isnan(v).any():
        imax, imin = np.where(v == v.max())[0], np.where(v == v.min())[0]
        ax.scatter([xs[i] for i in imax], v[imax], marker="^", s=110, c="darkgreen", zorder=5,
                   label=f"максимум ({v.max():.2f})")
        ax.scatter([xs[i] for i in imin], v[imin], marker="v", s=110, c="purple", zorder=5,
                   label=f"мінімум ({v.min():.2f})")
    ax.set_title("Часовий ряд за добу"); ax.set_ylabel("показник"); ax.set_xlim(-.3, 24)
    _hour_ticks(ax); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper left")

def plot_histogram(ax, x, edges, title, gamma=None):
    """гістаграма відносних частот + крива нормального закону (f(x)·h) + CI для x̄"""
    ax.clear()
    x = [float(v) for v in x]
    counts, rel = ms.histogram(x, edges)
    h = edges[1] - edges[0]
    ax.bar(edges[:-1], rel, width=h, align="edge", edgecolor="black", alpha=.6,
           label="відносні частоти nᵢ/n")
    m, s = ms.mean(x), ms.std(x)
    grid = np.linspace(min(edges[0], m - 3 * s), max(edges[-1], m + 3 * s), 300)
    ax.plot(grid, [ms.norm_pdf(g, m, s) * h for g in grid], "r-", lw=2,
            label=f"N(x̄={m:.2f}; S={s:.2f}) · h")
    ax.axvline(m, color="k", ls="--", lw=1, label="x̄")
    if gamma:
        rows, _ = ms.characteristics(x, gamma)
        ax.axvspan(rows[0][1], rows[0][3], color="gold", alpha=.35,
                   label=f"довірчий інтервал x̄ (γ={gamma})")
    ax.set_xticks(edges); ax.set_xticklabels([f"{e:.1f}" for e in edges], rotation=45, fontsize=7)
    ax.set_title(f"{title}: k={len(edges) - 1}, h={h:.3g}")
    top = max(max(rel), max(ms.norm_pdf(g, m, s) * h for g in grid))
    ax.set_ylim(0, top * 1.45)
    ax.set_ylabel("відносна частота"); ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.3)

def plot_cdf(ax, x, gamma):
    ax.clear()
    x = [float(v) for v in x]
    n = len(x)
    xs, F = ms.ecdf(x)
    m, s = ms.mean(x), ms.std(x)
    eps = ms.dkw_eps(n, gamma)
    Fa = np.array(F)
    ax.step(xs, Fa, where="post", color="tab:blue", lw=1.8, label="емпірична Fₙ(x)")
    ax.fill_between(xs, np.clip(Fa - eps, 0, 1), np.clip(Fa + eps, 0, 1), step="post",
                    color="tab:blue", alpha=.18, label=f"Fₙ ± ε, ε={eps:.3f} (γ={gamma})")
    grid = np.linspace(min(xs) - s, max(xs) + s, 400)
    ax.plot(grid, [ms.norm_cdf(g, m, s) for g in grid], "r-", lw=2,
            label=f"теоретична N(x̄={m:.2f}; S={s:.2f})")
    ax.set_title("Функція розподілу"); ax.set_ylabel("F(x)"); ax.set_xlabel("x")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
