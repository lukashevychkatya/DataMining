import re
import numpy as np
import pandas as pd
from math_statistics import mean, std
PERIOD_NAMES = ["Ніч (00:00–05:30)", "Ранок (06:00–11:30)", "День (12:00–17:30)", "Вечір (18:00–23:30)"]
TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*[;\t, ]\s*(.*?)\s*$")

def period_index(t):
    return min(int(t.split(":")[0]) // 6, 3)       # 0 ніч, 1 ранок, 2 день, 3 вечір
def time_to_hours(t):
    h, m = t.split(":")[:2]; return int(h) + int(m) / 60

# завантаження
def to_float(x):
    """число або NaN, чи було нечисловим. кома/крапка - обидва ок"""
    s = str(x).strip().replace("\u00a0", "").replace(" ", "").replace(",", ".")
    if s.lower() in ("", "nan", "none"): return np.nan, False
    try:    return float(s), False
    except ValueError: return np.nan, True

def _time(cell):
    if hasattr(cell, "strftime"): return cell.strftime("%H:%M")
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", str(cell))
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None

def list_sheets(path): return pd.ExcelFile(path).sheet_names
def load_file(path, sheet=None):
    """повертає times, values, bad_count, title. пропуск у values = NaN"""
    low, title = path.lower(), ""
    if low.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet or 0, header=None)
        if df.shape[1] < 2: raise ValueError("Потрібно щонайменше 2 стовпці (час, значення)")
        rows = list(zip(df[0], df[1]))
        title = str(df.iloc[0, 3]) if df.shape[1] > 3 and pd.notna(df.iloc[0, 3]) else ""
    elif low.endswith((".csv", ".txt")):
        for enc in ("utf-8-sig", "cp1251"):
            try:
                lines = open(path, encoding=enc).read().splitlines(); break
            except UnicodeDecodeError: pass
        rows = []
        for ln in filter(str.strip, lines):
            m = TIME_RE.match(ln)
            rows.append((f"{m.group(1)}:{m.group(2)}", m.group(3).split(";")[0]) if m else (None, ln.split(";")[0]))
    else:
        raise ValueError("Такий формат не підтримується (потрібно .xlsx, .csv або .txt)")

    times, vals, bad = [], [], 0
    for t, v in rows:
        t = _time(t) if t is not None and pd.notna(t) else None
        v, was_bad = to_float(v)
        if t is not None or not np.isnan(v):
            times.append(t); vals.append(v); bad += was_bad
    if not vals or np.isnan(vals).all(): raise ValueError("У файлі нема жодного числового значення")
    if None in times: # часу нема - крок 30 хв
        times = [f"{i * 30 // 60 % 24:02d}:{i * 30 % 60:02d}" for i in range(len(vals))]
    return times, np.array(vals, float), bad, title


# пропуски аномалії
def find_missing(values):
    return [int(i) for i in np.where(np.isnan(values))[0]]

def find_anomalies(values, lo, hi):
    """індекси значень поза [lo; hi] самі межі ще норма"""
    with np.errstate(invalid="ignore"):
        return [int(i) for i in np.where((values < lo) | (values > hi))[0]]

def suggest_types(anom_idx):
    s = set(anom_idx)
    return {i: "real" if (i - 1 in s or i + 1 in s) else "technical" for i in sorted(s)}

# заповнення і видалення
def period_means(times, values, good):
    """середні за ніч/ранок/день/вечір по коректних значеннях + загал середнє"""
    gm, per = float(values[good].mean()), np.array([period_index(t) for t in times])
    return [float(values[good & (per == p)].mean()) if (good & (per == p)).any() else gm
            for p in range(4)], gm

def clean(times, values, technical_idx, method):
    """method: 'period' | 'global' | 'delete'. повертає (times, values, log)"""
    values = values.astype(float).copy()
    miss, tech = np.isnan(values), np.zeros(len(values), bool)
    tech[list(technical_idx)] = True
    fix = miss | tech
    good = ~fix
    if not good.any(): raise ValueError("Немає жодного коректного значення")
    pm, gm = period_means(times, values, good)
    log = []
    for i in np.where(fix)[0]:
        new = None if method == "delete" else (pm[period_index(times[i])] if method == "period" else gm)
        log.append((times[i], values[i], new, "пропуск" if miss[i] else "технічна аномалія"))
        if new is not None: values[i] = new
    if method == "delete":
        return [t for t, f in zip(times, fix) if not f], values[good], log
    return list(times), values, log


# перетворення
def _complete(v):
    if np.isnan(v).any(): raise ValueError("У даних є пропуски. Спочатку заповніть чи видаліть їх")
    return v

def log_transform(values):
    """y = ln x. якщо є x <= 0 - спершу зсув c = 1 − min. повертає (y, c)"""
    v = _complete(values)
    c = 1.0 - v.min() if v.min() <= 0 else 0.0
    return np.log(v + c), c

def standardize(values):
    """y = (x − x̄)/S."""
    v = _complete(values)
    if std(list(v)) == 0: raise ValueError("S = 0 — стандартизація неможлива")
    return (v - mean(list(v))) / std(list(v))

def shift_by(values, c):
    """y = x + c."""
    return _complete(values) + c

# межі аномалій і журнал подій з аркуша Опис варіантів
def read_variant_info(path, sheet):
    """для аркуша «Варіант N» повертає {'lo', 'hi', 'object', 'indicator', 'journal'} або None"""
    try:
        n = int(re.findall(r"\d+", sheet)[-1])
        df = pd.read_excel(path, sheet_name="Опис варіантів", header=None)
    except Exception:
        return None
    for _, r in df.iterrows():
        try:
            if int(r.iloc[0]) == n:
                return {"object": str(r.iloc[1]), "indicator": str(r.iloc[2]), "journal": str(r.iloc[4]),
                        "lo": float(r.iloc[-2]), "hi": float(r.iloc[-1])}
        except (ValueError, TypeError):
            continue
    return None