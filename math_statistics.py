import math
from bisect import bisect_right
from scipy.stats import t as student_t

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