"""simulation.py - моделювання вибірок методом зворотної функції, T-тест, R повторень
Оцінювання викликається з distributions.py (не дублюється)"""
import math
import numpy as np
from scipy.stats import t as student_t
import distributions as dist
import math_statistics as ms

DEFAULT_N_SET = (50, 400, 1000)
DELTA_GRID = (20, 50, 100, 200, 400, 1000, 2000, 5000)

# перевірка введення
def check_experiment(law, theta0, R, alpha, Ns):
    dist.check_params(law, theta0)
    if not 200 <= R <= 500:
        raise ValueError("Кількість повторень R має бути в межах [200; 500]")
    if not 0 < alpha < 1:
        raise ValueError("Рівень значущості α має бути в інтервалі (0; 1)")
    if not Ns or any(n < 5 for n in Ns):
        raise ValueError("Обсяги вибірок N мають бути цілими числами ≥ 5")

def parse_n_set(text):
    try:
        return [int(v) for v in text.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise ValueError("Обсяги N введіть цілими числами через кому, напр. 50, 400, 1000...")

# одна вибірка
def simulate(law, p, N, rng):
    """N значень: u = rng.random(N) (рівномірне на [0; 1)), x = F⁻¹(u; θ)"""
    return dist.inv(law, rng.random(N), p)

def sample_vs_theory(law, p, x):
    """Вибіркові x̄, S², A, E та теоретичні значення"""
    x = [float(v) for v in x]
    th = dist.theory_moments(law, p)
    return [("математичне сподівання / x̄", th[0], ms.mean(x)), ("дисперсія / S²", th[1], ms.variance(x)),
            ("асиметрія A", th[2], ms.skewness(x)), ("ексцес E", th[3], ms.kurtosis(x))]

# T-тест і R повторень
def run_experiment(law, theta0, Ns, R, alpha, seed, progress=None):
    """Для кожного N: R разів — вибірка → оцінки → t = (θ₀ − θ̂)/σ{θ̂}; |t| > t_{1−α/2; N−1} → відхилення H₀
    Повертає (rows[param] = [dict по N], tvals[(param, N)] = масив t)"""
    check_experiment(law, theta0, R, alpha, Ns)
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in dist.PARAMS[law]}
    tvals = {}
    for j, N in enumerate(Ns):
        X = dist.inv(law, rng.random((R, N)), theta0)          # матриця R×N вибірок
        est = dist.estimate(law, X)
        tcrit = float(student_t.ppf(1 - alpha / 2, N - 1))
        sig_th = dist.sigma_theory(law, theta0, N)
        for name, (th, sg) in est.items():
            t = (theta0[name] - th) / sg
            tvals[(name, N)] = t
            rows[name].append({"N": N, "ν": N - 1, "tкр": tcrit, "середня θ̂": float(th.mean()),
                               "СКВ θ̂": float(th.std(ddof=1)), "σ{θ̂} теор.": float(sig_th[name]),
                               "середнє t": float(t.mean()), "СКВ t": float(t.std(ddof=1)),
                               "відхилень H₀, %": float(100 * np.mean(np.abs(t) > tcrit))})
        if progress:
            progress(j + 1, len(Ns), f"N = {N}")
    return rows, tvals

# δ(N) і необхідний обсяг
def delta_curve(law, theta0, R, alpha, seed, grid=DELTA_GRID, progress=None):
    """δ(N) = середня відносна похибка оцінки, % : mean|θ̂ − θ₀| / |θ₀| · 100.
    Додатково — теоретична півширина довірчого інтервалу t·σ{θ̂}/|θ₀|, %"""
    rng = np.random.default_rng(seed + 1)
    out = {k: {"N": [], "δ": [], "інтервал": []} for k in dist.PARAMS[law]}
    for j, N in enumerate(grid):
        X = dist.inv(law, rng.random((R, N)), theta0)
        est = dist.estimate(law, X)
        tq = float(student_t.ppf(1 - alpha / 2, N - 1))
        sig = dist.sigma_theory(law, theta0, N)
        for name, (th, _) in est.items():
            den = abs(theta0[name]) or 1.0
            out[name]["N"].append(N)
            out[name]["δ"].append(float(100 * np.mean(np.abs(th - theta0[name])) / den))
            out[name]["інтервал"].append(float(100 * tq * sig[name] / den))
        if progress:
            progress(j + 1, len(grid), f"δ(N): N = {N}")
    return out

def required_n(law, theta0, alpha, targets=(10, 5, 2, 1)):
    """Найменше N, при якому півширина довірчого інтервалу t·σ{θ̂}/|θ₀| ≤ δ*, %
    Оскільки σ{θ̂} ∝ 1/√N, розв'язуємо ітерацією N = (t·c/δ*)^2, c = σ{θ̂}·√N/|θ₀|"""
    res = {}
    for name in dist.PARAMS[law]:
        den = abs(theta0[name]) or 1.0
        c = dist.sigma_theory(law, theta0, 1)[name] / den
        row = {}
        for d in targets:
            N = 20
            for _ in range(50):
                tq = float(student_t.ppf(1 - alpha / 2, max(N - 1, 1)))
                new = max(20, math.ceil((tq * c / (d / 100)) ** 2))
                if new == N:
                    break
                N = new
            row[d] = N
        res[name] = row
    return res