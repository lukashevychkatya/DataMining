import math
import numpy as np
from bisect import bisect_right
from scipy.stats import t as student_t, norm

# точкові оцінки
def mean(x):
    return sum(x) / len(x)                                   # x̄
def variance(x): m = mean(x); return sum((v - m) ** 2 for v in x) / (len(x) - 1)   # S²
def std(x):
    return math.sqrt(variance(x))                            # S
def skewness(x): m, s = mean(x), std(x); return sum((v - m) ** 3 for v in x) / (len(x) * s ** 3)      # A
def kurtosis(x): m, s = mean(x), std(x); return sum((v - m) ** 4 for v in x) / (len(x) * s ** 4) - 3  # E

# похибки (СКВ) оцінок
def sigma_mean(s, n):
    return s / math.sqrt(n)
def sigma_var(s2, n):
    return s2 * math.sqrt(2 / (n - 1))
def sigma_std(s, n):
    return s / math.sqrt(2 * (n - 1))
def sigma_skew(n):
    return math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
def sigma_kurt(n):
    return math.sqrt(24 * n * (n - 2) * (n - 3) / ((n - 1) ** 2 * (n + 3) * (n + 5)))

def characteristics(x, gamma):
    """таблиця /назва/θн/θ/θв/ σθ/ і квантиль t. інтервал: θ ± t·σθ"""
    x, n = [float(v) for v in x], len(x)
    if n < 4:                  raise ValueError("Потрібно щонайменше 4 значення")
    if not 0 < gamma < 1:      raise ValueError("Довірча ймовірність γ має бути в (0; 1)")
    if max(x) == min(x):       raise ValueError("Усі значення однакові (S = 0)")
    t = float(student_t.ppf((1 + gamma) / 2, n - 1))
    m, s2, s = mean(x), variance(x), std(x)
    est = [("x̄ (середнє)", m, sigma_mean(s, n)), ("S² (дисперсія)", s2, sigma_var(s2, n)),
           ("S (СКВ)", s, sigma_std(s, n)), ("A (асиметрія)", skewness(x), sigma_skew(n)),
           ("E (ексцес)", kurtosis(x), sigma_kurt(n))]
    return [(name, th - t * sg, th, th + t * sg, sg) for name, th, sg in est], t

# варіаційний ряд і гістограма
def variational_series(x): return sorted(float(v) for v in x)
def sturges_k(n):          return max(1, round(1 + 3.322 * math.log10(n)))

def edges_by_k(x, k):
    """межі k рівних класів: крок h = (xmax − xmin)/k"""
    h = (max(x) - min(x)) / k
    return [min(x) + i * h for i in range(k + 1)]

def edges_by_step(x, step):
    if step <= 0: raise ValueError("Крок має бути додатнім")
    start = math.floor(min(x) / step) * step
    k = math.floor((max(x) - start) / step) + 1
    return [start + i * step for i in range(k + 1)]

def histogram(x, edges):
    k, counts = len(edges) - 1, [0] * (len(edges) - 1)
    for v in x:
        counts[min(max(bisect_right(edges, v) - 1, 0), k - 1)] += 1
    return counts, [c / len(x) for c in counts]

# функції розподілу
def norm_pdf(x, m, s):
    return math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
def norm_cdf(x, m, s):
    return 0.5 * (1 + math.erf((x - m) / (s * math.sqrt(2))))

def ecdf(x):
    """емпірична Fn- відсортовані значення і частка значень <= x"""
    xs = variational_series(x)
    return xs, [(i + 1) / len(xs) for i in range(len(xs))]

def dkw_eps(n, gamma):
    """ε = √(ln(2/α)/(2N)), α = 1 − γ"""
    return math.sqrt(math.log(2 / (1 - gamma)) / (2 * n))

# ЛР4: кореляція та лінійна регресія
def drop_outliers_3s(x, y):
    """Правило 3σ: пара (xᵢ, yᵢ) відкидається, якщо |xᵢ − x̄| > 3Sₓ або |yᵢ − ȳ| > 3Sᵧ.
    Повертає (x, y, відкинуті пари)"""
    x, y = [float(v) for v in x], [float(v) for v in y]
    if len(x) != len(y):
        raise ValueError("X і Y мають бути однакової довжини")
    if len(x) < 4:
        raise ValueError("Потрібно щонайменше 4 пари значень")
    mx, sx, my, sy = mean(x), std(x), mean(y), std(y)
    bad = {i for i in range(len(x)) if abs(x[i] - mx) > 3 * sx or abs(y[i] - my) > 3 * sy}
    keep = [i for i in range(len(x)) if i not in bad]
    return [x[i] for i in keep], [y[i] for i in keep], [(x[i], y[i]) for i in sorted(bad)]

def pearson(x, y):
    """r̂ = Σ(xᵢ−x̄)(yᵢ−ȳ) / √(Σ(xᵢ−x̄)² · Σ(yᵢ−ȳ)²)"""
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        raise ValueError("S = 0: коефіцієнт кореляції не визначений")
    return sxy / math.sqrt(sxx * syy)

def ranks(x):
    """Ранги за зростанням; однакові значення дістають середнє арифметичне позицій"""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r, i = [0.0] * len(x), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r

def spearman(x, y):
    """коефіцієнт Пірсона для рангів"""
    return pearson(ranks(x), ranks(y))

def corr_test(r, n, alpha):
    """t = r√(N−2)/√(1−r²); |t| > t_{1−α/2; N−2} → H₀: r = 0 відхиляємо. Повертає (t, tкр, p-value, значущий)"""
    nu = n - 2
    t = math.copysign(math.inf, r) if abs(r) >= 1 else r * math.sqrt(nu) / math.sqrt(1 - r * r)
    tcrit = float(student_t.ppf(1 - alpha / 2, nu))
    return t, tcrit, float(2 * student_t.sf(abs(t), nu)), abs(t) > tcrit

def fisher_interval(r, n, gamma):
    """z = ½·ln((1+r)/(1−r)); z ± u_{1−α/2}/√(N−3); назад r = (e^{2z}−1)/(e^{2z}+1)"""
    if abs(r) >= 1:
        return r, r
    u = float(norm.ppf((1 + gamma) / 2))
    z = 0.5 * math.log((1 + r) / (1 - r))
    d = u / math.sqrt(n - 3)
    back = lambda v: (math.exp(2 * v) - 1) / (math.exp(2 * v) + 1)
    return back(z - d), back(z + d)

def chaddock(r):
    """зв'язок за шкалою Чеддока"""
    a = abs(r)
    for lim, name in ((0.1, "практично відсутній"), (0.3, "слабкий"), (0.5, "помірний"),
                      (0.7, "помітний"), (0.9, "високий")):
        if a < lim:
            return name
    return "дуже високий"

def linear_regression(x, y, gamma):
    """МНК: ŷ = â + b̂x. Повертає словник оцінок, їхніх СКВ, довірчих інтервалів, t-тестів і R²"""
    x, y = [float(v) for v in x], [float(v) for v in y]
    n = len(x)
    if n < 3:
        raise ValueError("Для регресії потрібно щонайменше 3 пари")
    mx, my = mean(x), mean(y)
    sxx = sum((v - mx) ** 2 for v in x)            # = Sₓ²·(N−1)
    syy = sum((v - my) ** 2 for v in y)
    if sxx == 0:
        raise ValueError("Усі xᵢ однакові: регресію відтворити неможливо")
    if syy == 0:
        raise ValueError("Усі yᵢ однакові (Sᵧ = 0)")
    b = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sxx
    a = my - b * mx
    yhat = [a + b * u for u in x]
    sse = sum((v - h) ** 2 for v, h in zip(y, yhat))
    nu = n - 2
    s_res = math.sqrt(sse / nu)
    s_b = s_res / math.sqrt(sxx)
    s_a = s_res * math.sqrt(1 / n + mx ** 2 / sxx)
    tq = float(student_t.ppf((1 + gamma) / 2, nu))

    def test(est, sg):
        t = est / sg if sg else math.copysign(math.inf, est)
        return t, float(2 * student_t.sf(abs(t), nu)), abs(t) > tq
    t_a, p_a, ok_a = test(a, s_a)
    t_b, p_b, ok_b = test(b, s_b)
    return {"a": a, "b": b, "s_res": s_res, "s_a": s_a, "s_b": s_b, "nu": nu, "tq": tq,
            "a_lo": a - tq * s_a, "a_hi": a + tq * s_a, "b_lo": b - tq * s_b, "b_hi": b + tq * s_b,
            "t_a": t_a, "p_a": p_a, "ok_a": ok_a, "t_b": t_b, "p_b": p_b, "ok_b": ok_b,
            "R2": 1 - sse / syy, "x_mean": mx, "y_mean": my}

def norm2_pdf(x, y, mx, my, sx, sy, r):
    """щільність двовимірного нормального закону"""
    if abs(r) >= 1:
        raise ValueError("|r| = 1: щільність виродена, 3D-графік неможливий")
    zx, zy = (x - mx) / sx, (y - my) / sy
    q = (zx ** 2 - 2 * r * zx * zy + zy ** 2) / (1 - r * r)
    return np.exp(-0.5 * q) / (2 * math.pi * sx * sy * math.sqrt(1 - r * r))