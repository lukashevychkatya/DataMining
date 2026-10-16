"""visualization.py - графіки """
import numpy as np
import math_statistics as ms
import distributions as dist
from data_processing import time_to_hours
LAW_COLORS = {"Нормальний": "tab:red", "Рівномірний": "tab:green", "Експоненціальний": "tab:purple"}
def _hour_ticks(ax):
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=45, fontsize=8)

# ЛР1
def plot_timeseries(ax, times, values, raw=None, bounds=None, types=None):
    """Часовий ряд"""
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
    """Стовпці - це відносні частоти nᵢ/n; червона крива -нормальна щільність f(x)·h; пунктир - x̄;
    жовта смуга - довірчий інтервал для x̄"""
    ax.clear()
    x = [float(v) for v in x]
    counts, rel = ms.histogram(x, edges)
    h = edges[1] - edges[0]
    ax.bar(edges[:-1], rel, width=h, align="edge", edgecolor="black", alpha=.6,
           label="стовпці: відносні частоти nᵢ/n")
    m, s = ms.mean(x), ms.std(x)
    grid = np.linspace(min(edges[0], m - 3 * s), max(edges[-1], m + 3 * s), 300)
    ax.plot(grid, [ms.norm_pdf(g, m, s) * h for g in grid], "r-", lw=2,
            label=f"червона крива: нормальна щільність f(x)·h, x̄={m:.2f}, S={s:.2f}")
    ax.axvline(m, color="k", ls="--", lw=1, label="x̄ (середнє)")
    if gamma:
        rows, _ = ms.characteristics(x, gamma)
        ax.axvspan(rows[0][1], rows[0][3], color="gold", alpha=.35,
                   label=f"жовта смуга: довірчий інтервал x̄ (γ={gamma})")
    ax.set_xticks(edges); ax.set_xticklabels([f"{e:.1f}" for e in edges], rotation=45, fontsize=7)
    top = max(max(rel), max(ms.norm_pdf(g, m, s) * h for g in grid))
    ax.set_ylim(0, top * 1.55)
    ax.set_title(f"{title}: k={len(edges) - 1}, h=(max−min)/k={h:.3g}")
    ax.set_ylabel("відносна частота"); ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.3)

def plot_cdf(ax, x, gamma):
    """Сині східці емпірична Fₙ(x): кожне значення піднімає графік на 1/n;
    блакитна смуга Fₙ ± ε; червона крива теоретична нормальна F(x) = Φ((x−x̄)/S)"""
    ax.clear()
    x = [float(v) for v in x]
    n = len(x)
    xs, F = ms.ecdf(x)
    m, s = ms.mean(x), ms.std(x)
    eps = ms.dkw_eps(n, gamma)
    xs = xs + [xs[-1] + 0.1 * (xs[-1] - xs[0])] # продовжуємо вправо, щоб було видно рівень 1
    Fa = np.array(F + [1.0])
    ax.step(xs, Fa, where="post", color="tab:blue", lw=1.8,
            label=f"сині східці: емпірична Fₙ(x); кожен стрибок = 1/n = {1 / n:.3f}")
    ax.fill_between(xs, np.clip(Fa - eps, 0, 1), np.clip(Fa + eps, 0, 1), step="post",
                    color="tab:blue", alpha=.18, label=f"блакитна смуга: Fₙ ± ε, ε={eps:.3f} (γ={gamma})")
    grid = np.linspace(min(xs) - s, max(xs) + s, 400)
    ax.plot(grid, [ms.norm_cdf(g, m, s) for g in grid], "r-", lw=2,
            label=f"червона крива: теоретична Φ((x−x̄)/S), x̄={m:.2f}, S={s:.2f}")
    ax.set_title("Функція розподілу"); ax.set_ylabel("F(x)"); ax.set_xlabel("x")
    ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower right")

# ЛР2
def plot_laws(ax_h, ax_c, x, edges, params_by_law):
    """Гістограма з щільностями та Fₙ з теоретичними F для кількох законів"""
    x = [float(v) for v in x]
    ax_h.clear(); ax_c.clear()
    counts, rel = ms.histogram(x, edges)
    h = edges[1] - edges[0]
    ax_h.bar(edges[:-1], rel, width=h, align="edge", edgecolor="black", alpha=.5, label="відносні частоти")
    lo, hi = min(x), max(x)
    grid = np.linspace(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo), 400)
    xs, F = ms.ecdf(x)
    xs = xs + [xs[-1] + 0.1 * (xs[-1] - xs[0])]
    ax_c.step(xs, F + [1.0], where="post", color="tab:blue", lw=1.8, label="емпірична Fₙ(x)")
    for law, p in params_by_law.items():
        ax_h.plot(grid, dist.pdf(law, grid, p) * h, color=LAW_COLORS[law], lw=2, label=f"{law}: f(x)·h")
        ax_c.plot(grid, dist.cdf(law, grid, p), color=LAW_COLORS[law], lw=2, label=f"{law}: F(x)")
    ax_h.set_title("Гістограма та щільності трьох законів"); ax_h.set_ylabel("відносна частота")
    ax_c.set_title("Fₙ(x) та теоретичні F(x)"); ax_c.set_ylabel("F(x)")
    for a in (ax_h, ax_c):
        a.grid(alpha=.3); a.legend(fontsize=7)

def plot_sim_sample(ax, sample, law, p, k=20):
    """Гістограма змодельованої вибірки (щільність nᵢ/(n·h)) і теоретична f(x)"""
    ax.clear()
    x = [float(v) for v in sample]
    edges = ms.edges_by_k(x, k)
    h = edges[1] - edges[0]
    counts, rel = ms.histogram(x, edges)
    ax.bar(edges[:-1], [r / h for r in rel], width=h, align="edge", edgecolor="black", alpha=.6,
           label="гістограма вибірки (nᵢ/(n·h))")
    grid = np.linspace(min(x), max(x), 400)
    ax.plot(grid, dist.pdf(law, grid, p), "r-", lw=2, label="теоретична щільність f(x)")
    ax.set_title(f"Змодельована вибірка: {law}, N={len(x)}"); ax.set_ylabel("щільність")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

def plot_t_hist(ax, t, tcrit, N, param):
    """Гістограма R значень t, щільність N(0,1) і межі ±t_кр """
    ax.clear()
    t = [float(v) for v in t]
    edges = ms.edges_by_k(t, 15)
    h = edges[1] - edges[0]
    counts, rel = ms.histogram(t, edges)
    ax.bar(edges[:-1], [r / h for r in rel], width=h, align="edge", edgecolor="black", alpha=.6,
           label="гістограма t")
    g = np.linspace(min(-4, edges[0]), max(4, edges[-1]), 300)
    ax.plot(g, [ms.norm_pdf(v, 0, 1) for v in g], "k-", lw=2, label="N(0, 1)")
    ax.axvline(tcrit, color="red", ls="--", label=f"±t_кр = ±{tcrit:.3f}"); ax.axvline(-tcrit, color="red", ls="--")
    ax.set_title(f"Параметр {param}: значення t, N = {N}"); ax.set_xlabel("t")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

def plot_delta(ax, curves):
    """δ(N) у логарифмічному масштабі: суцільна - похибка за експериментом,
    пунктир - півширина довірчого інтервалу t·σ{θ̂}/|θ₀|"""
    ax.clear()
    for name, c in curves.items():
        line, = ax.plot(c["N"], c["δ"], "o-", label=f"{name}: δ(N)")
        ax.plot(c["N"], c["інтервал"], "--", color=line.get_color(), label=f"{name}: півширина інтервалу")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (обсяг вибірки)"); ax.set_ylabel("відносна похибка, %")
    ax.set_title("δ(N): як зменшується похибка зі зростанням N"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)