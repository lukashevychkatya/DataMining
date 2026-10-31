"""twodim.py — ЛР3: генерація, оцінки, інтервали, гістограма, щільність, χ², t-тест для r."""
import math
import numpy as np
from scipy.stats import t as student_t, chi2
VARIANT_SEED_FACTOR = 100

def generate(mx, my, sx, sy, r, N, seed):
    check_params(sx, sy, r, N)
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal(N), rng.standard_normal(N)
    x = mx + sx * z1
    y = my + sy * (r * z1 + math.sqrt(1 - r * r) * z2)
    return x, y

def check_params(sx, sy, r, N):
    if sx <= 0 or sy <= 0:
        raise ValueError("σx і σy мають бути додатними ")
    if abs(r) >= 1:
        raise ValueError("|r| має бути меншим за 1 ")
    if N < 10:
        raise ValueError("Обсяг вибірки N має бути >= 10 ")

def read_pairs(path):
    if path.lower().endswith((".xlsx", ".xls")):
        import pandas as pd
        df = pd.read_excel(path, header=None).dropna()
        x, y = df.iloc[:, 0].astype(float).values, df.iloc[:, 1].astype(float).values
    else:
        rows = []
        for ln in open(path, encoding="utf-8-sig"):
            ln = ln.strip()
            if not ln:
                continue
            sep = ";" if ";" in ln else ("\t" if "\t" in ln else None)
            parts = (ln.split(sep) if sep else ln.split())
            parts = [p.replace(",", ".") for p in parts[:2]]
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue
        if not rows:
            raise ValueError("У файлі немає двох числових стовпців.")
        x, y = np.array([r_[0] for r_ in rows]), np.array([r_[1] for r_ in rows])
    if len(x) < 10:
        raise ValueError("Вибірка має містити щонайменше 10 пар.")
    return x, y


def estimates(x, y):
    """x̄, ȳ, Sx², Sy², Kxy, rxy — формули з методички (дільник N−1)."""
    N = len(x)
    xm, ym = x.sum() / N, y.sum() / N
    Sx2 = ((x - xm) ** 2).sum() / (N - 1)
    Sy2 = ((y - ym) ** 2).sum() / (N - 1)
    Kxy = ((x - xm) * (y - ym)).sum() / (N - 1)
    r = ((x - xm) * (y - ym)).sum() / math.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    return {"x̄": xm, "ȳ": ym, "Sx²": Sx2, "Sy²": Sy2, "Kxy": Kxy, "rxy": r}


def intervals(x, y, gamma):
    """Рядки таблиці: (показник, θн, θ, θв). Середні — t-інтервал; дисперсії — χ²-інтервал."""
    N, alpha = len(x), 1 - gamma
    e = estimates(x, y)
    tq = float(student_t.ppf(1 - alpha / 2, N - 1))
    rows = []
    for name, sname, mean in (("mx", "x̄", e["x̄"]), ("my", "ȳ", e["ȳ"])):
        S = math.sqrt(e["Sx²" if name == "mx" else "Sy²"])
        rows.append((name, mean - tq * S / math.sqrt(N), mean, mean + tq * S / math.sqrt(N)))
    for name, key in (("σx²", "Sx²"), ("σy²", "Sy²")):
        lo = (N - 1) * e[key] / chi2.ppf(1 - alpha / 2, N - 1)
        hi = (N - 1) * e[key] / chi2.ppf(alpha / 2, N - 1)
        rows.append((name, lo, e[key], hi))
    return rows, e

def covers(rows, true):
    """Чи накриває інтервал заданий параметр варіанта"""
    tv = {"mx": true["mx"], "my": true["my"], "σx²": true["sx"] ** 2, "σy²": true["sy"] ** 2}
    return {name: lo <= tv[name] <= hi for name, lo, _, hi in rows if name in tv}

def r_test(r, N, gamma):
    """t = r·√(N−2)/√(1−r²) з ν = N−2; H₀: r = 0"""
    alpha = 1 - gamma
    t = r * math.sqrt(N - 2) / math.sqrt(1 - r * r)
    tc = float(student_t.ppf(1 - alpha / 2, N - 2))
    return t, tc, abs(t) > tc

def bin_edges(x, y, k):
    ex = np.linspace(x.min(), x.max(), k + 1)
    ey = np.linspace(y.min(), y.max(), k + 1)
    return ex, ey

def histogram2d(x, y, k):
    ex, ey = bin_edges(x, y, k)
    n, _, _ = np.histogram2d(x, y, bins=[ex, ey])     # n[i, j]: i — x, j — y
    n = n.T                                            # рядок j (y), стовпець i (x) — як у таблиці
    cx, cy = (ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2
    return n, n / len(x), cx, cy, ex, ey

def density(x, y, mx, my, sx, sy, r):
    """f(x,y) двовимірного нормального """
    def f(X, Y):
        Q = ((X - mx) / sx) ** 2 - 2 * r * (X - mx) * (Y - my) / (sx * sy) + ((Y - my) / sy) ** 2
        return np.exp(-Q / (2 * (1 - r * r))) / (2 * math.pi * sx * sy * math.sqrt(1 - r * r))
    return f

def chi2_test(x, y, k, mx, my, sx, sy, r, alpha):
    N = len(x)
    n, p_emp, cx, cy, ex, ey = histogram2d(x, y, k)
    hx, hy = ex[1] - ex[0], ey[1] - ey[0]
    f = density(x, y, mx, my, sx, sy, r)
    CX, CY = np.meshgrid(cx, cy) # [j, i] = (cx[i], cy[j])
    Np = f(CX, CY) * hx * hy * N
    obs = n.astype(float)
    small = Np < 5
    obs_g = list(obs[~small]); exp_g = list(Np[~small])
    if small.any(): # об'єднуємо малі класи
        obs_g.append(obs[small].sum()); exp_g.append(Np[small].sum())
    obs_g, exp_g = np.array(obs_g), np.array(exp_g)
    stat = float(((obs_g - exp_g) ** 2 / exp_g).sum())
    groups = len(obs_g)
    nu = groups - 1 - 5  # 5 оцінених параметрів
    crit = float(chi2.ppf(1 - alpha, nu))
    return {"χ²": stat, "ν": nu, "крит": crit, "груп": groups, "малих": int(small.sum()),
            "адекватно": stat <= crit}, n, Np, cx, cy, ex, ey

def merge_neighbors(obs, exp, min_exp=5.0):
    go, ge, co, ce = [], [], 0.0, 0.0
    for o, e in zip(obs, exp):
        co += o; ce += e
        if ce >= min_exp:
            go.append(co); ge.append(ce); co = ce = 0.0
    if ce > 0:
        if ge:
            go[-1] += co; ge[-1] += ce
        else:
            go.append(co); ge.append(ce)
    return np.array(go), np.array(ge)

def chi2_test(x, y, k, mx, my, sx, sy, r, alpha):
    N = len(x)
    n, p_emp, cx, cy, ex, ey = histogram2d(x, y, k)
    hx, hy = ex[1] - ex[0], ey[1] - ey[0]
    f = density(x, y, mx, my, sx, sy, r)
    CX, CY = np.meshgrid(cx, cy)
    Np = f(CX, CY) * hx * hy * N
    obs_g, exp_g = merge_neighbors(n.ravel().astype(float), Np.ravel())
    stat = float(((obs_g - exp_g) ** 2 / exp_g).sum())
    groups = len(obs_g)
    nu = groups - 1 - 5
    crit = float(chi2.ppf(1 - alpha, nu))
    return {"χ²": stat, "ν": nu, "крит": crit, "груп": groups,
            "малих": int((Np < 5).sum()), "адекватно": stat <= crit}, n, Np, cx, cy, ex, ey