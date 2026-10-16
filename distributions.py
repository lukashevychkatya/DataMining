"""distributions.py - три закони (нормальний, рівномірний, експоненціальний):
        щільність f(x), функція розподілу F(x), обернена функція F⁻¹(u), оцінки параметрів,
        їхні СКВ σ{θ̂}, довірчі інтервали та відстань D = max|Fₙ − F|"""
import math
import numpy as np
from scipy.stats import norm, t as student_t
import math_statistics as ms

LAWS = ["Нормальний", "Рівномірний", "Експоненціальний"]
PARAMS = {"Нормальний": ["m", "σ"], "Рівномірний": ["a", "b"], "Експоненціальний": ["λ"]}
THEORY_AE = {"Нормальний": (0.0, 0.0), "Рівномірний": (0.0, -1.2), "Експоненціальний": (2.0, 6.0)}

# перевірка параметрів
def check_params(law, p):
    """p - словник {ім'я: значення}. Кидає ValueError для неприпустимих параметрів"""
    if law == "Нормальний" and p["σ"] <= 0:
        raise ValueError("Для нормального закону потрібно σ > 0.")
    if law == "Рівномірний" and p["a"] >= p["b"]:
        raise ValueError("Для рівномірного закону потрібно a < b.")
    if law == "Експоненціальний" and p["λ"] <= 0:
        raise ValueError("Для експоненціального закону потрібно λ > 0.")

# f(x), F(x), F^-1(u) (працюють і для чисел, і для масивів numpy)
def pdf(law, x, p):
    x = np.asarray(x, float)
    if law == "Нормальний":
        return np.exp(-0.5 * ((x - p["m"]) / p["σ"]) ** 2) / (p["σ"] * math.sqrt(2 * math.pi))
    if law == "Рівномірний":
        return np.where((x >= p["a"]) & (x <= p["b"]), 1.0 / (p["b"] - p["a"]), 0.0)
    return np.where(x >= 0, p["λ"] * np.exp(-p["λ"] * np.clip(x, 0, None)), 0.0)


def cdf(law, x, p):
    x = np.asarray(x, float)
    if law == "Нормальний":
        return norm.cdf((x - p["m"]) / p["σ"]) # Φ((x−m)/σ)
    if law == "Рівномірний":
        return np.clip((x - p["a"]) / (p["b"] - p["a"]), 0.0, 1.0)
    return np.where(x >= 0, 1.0 - np.exp(-p["λ"] * np.clip(x, 0, None)), 0.0)


def inv(law, u, p):
    """Метод зворотної функції: x = F⁻^-1(u), u - рівномірне на [0; 1)"""
    u = np.asarray(u, float)
    if law == "Нормальний":
        return p["m"] + p["σ"] * norm.ppf(u)      # m + σ·Φ⁻¹(u)
    if law == "Рівномірний":
        return p["a"] + (p["b"] - p["a"]) * u     # a + (b−a)·u
    return -np.log(1.0 - u) / p["λ"]              # −ln(1−u)/λ  (1−u > 0 завжди)

# теоретичні характеристики
def theory_moments(law, p):
    """(математичне сподівання, дисперсія, A, E)."""
    A, E = THEORY_AE[law]
    if law == "Нормальний":
        return p["m"], p["σ"] ** 2, A, E
    if law == "Рівномірний":
        return (p["a"] + p["b"]) / 2, (p["b"] - p["a"]) ** 2 / 12, A, E
    return 1 / p["λ"], 1 / p["λ"] ** 2, A, E

# оцінки параметрів та σ{θ̂} (x може бути 1-D вибіркою або матрицею RхN)
def estimate(law, x):
    """Повертає словник {параметр: (оцінка, σ{оцінки})}. Вісь вибірки - остання"""
    x = np.asarray(x, float)
    N = x.shape[-1]
    if N < 4:
        raise ValueError("Для оцінювання потрібно щонайменше 4 значення")
    mean = x.sum(axis=-1) / N                                        # x̄
    S = np.sqrt(((x - mean[..., None]) ** 2).sum(axis=-1) / (N - 1))  # S
    if law == "Нормальний":
        return {"m": (mean, S / math.sqrt(N)), "σ": (S, S / math.sqrt(2 * N))}
    if law == "Рівномірний":
        a, b = mean - math.sqrt(3) * S, mean + math.sqrt(3) * S
        sg = (b - a) * math.sqrt(2 / (15 * N))
        return {"a": (a, sg), "b": (b, sg)}
    if np.any(x < 0):
        raise ValueError("Експоненціальна модель не допускає від'ємних даних")
    lam = 1 / mean
    return {"λ": (lam, lam / math.sqrt(N))}

def sigma_theory(law, p, N):
    """Теоретичне σ{θ̂} при істинних параметрах p (для таблиці T-тесту)"""
    if law == "Нормальний":
        return {"m": p["σ"] / math.sqrt(N), "σ": p["σ"] / math.sqrt(2 * N)}
    if law == "Рівномірний":
        s = (p["b"] - p["a"]) * math.sqrt(2 / (15 * N))
        return {"a": s, "b": s}
    return {"λ": p["λ"] / math.sqrt(N)}

def intervals(law, x, gamma):
    """Рядки таблиці: (закон, параметр, θн, θ̂, θв, σ{θ̂}); інтервал θ̂ ± t·σ, ν = N-1"""
    x = np.asarray(x, float)
    tq = float(student_t.ppf((1 + gamma) / 2, len(x) - 1))
    rows = []
    for name, (th, sg) in estimate(law, x).items():
        rows.append((law, name, float(th - tq * sg), float(th), float(th + tq * sg), float(sg)))
    return rows

def estimates_dict(law, x):
    return {name: float(th) for name, (th, _) in estimate(law, x).items()}

# відстань D = max |Fₙ(x) − F(x)|
def distance_D(law, x, p):
    """Колмогоровська відстань між східчастою Fₙ та F: у кожній точці xᵢ перевіряємо
    обидва краї сходинки — i/n (після стрибка) і (i−1)/n (до стрибка)"""
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    F = cdf(law, xs, p)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))

# підказка за формою
def shape_table(x, gamma):
    """Порівняння довірчих інтервалів A та E з теоретичними значеннями трьох законів"""
    x = [float(v) for v in x]
    rows, _ = ms.characteristics(x, gamma)
    (_, a_lo, a, a_hi, _), (_, e_lo, e, e_hi, _) = rows[3], rows[4]
    out = []
    for law in LAWS:
        tA, tE = THEORY_AE[law]
        out.append((law, tA, a_lo <= tA <= a_hi, tE, e_lo <= tE <= e_hi))
    return (a_lo, a, a_hi), (e_lo, e, e_hi), out

def variation_coeff(x):
    """V = S / x̄ (для експоненціального закону V = 1)"""
    x = [float(v) for v in x]
    return ms.std(x) / ms.mean(x)