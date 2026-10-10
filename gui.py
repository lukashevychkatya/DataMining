import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import data_processing as dp
import math_statistics as ms
import visualization as viz

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ІАД || ЛР №1 || Первинний статистичний аналіз (ІАД) || Лукашевич К. ПП-31/1")
        self.geometry("1350x860")
        # стан
        self.path = None
        self.raw_t = self.raw_v = None
        self.cur_t = self.cur_v = None
        self.types = {}
        self.transformed = False
        self.stages = []
        self._build()

    # побудова інтерфейсу
    def _build(self):
        left = ttk.Frame(self, padding=6); left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=6); right.pack(side="right", fill="both", expand=True)

        box = ttk.LabelFrame(left, text="1. Дані"); box.pack(fill="x", pady=3)
        ttk.Button(box, text="Відкрити файл…", command=self.open_file).pack(fill="x")
        self.sheet_var = tk.StringVar()
        self.sheet_cb = ttk.Combobox(box, textvariable=self.sheet_var, state="readonly")
        self.sheet_cb.pack(fill="x"); self.sheet_cb.bind("<<ComboboxSelected>>", lambda e: self.load())

        box = ttk.LabelFrame(left, text="2. Аномалії"); box.pack(fill="x", pady=3)
        self.lo_var, self.hi_var = tk.StringVar(value="3"), tk.StringVar(value="50")
        r = ttk.Frame(box); r.pack(fill="x")
        ttk.Label(r, text="нижня").pack(side="left"); ttk.Entry(r, textvariable=self.lo_var, width=6).pack(side="left")
        ttk.Label(r, text=" верхня").pack(side="left"); ttk.Entry(r, textvariable=self.hi_var, width=6).pack(side="left")
        ttk.Button(box, text="Знайти аномалії", command=self.detect).pack(fill="x")
        self.tree = ttk.Treeview(box, columns=("t", "v", "k"), show="headings", height=5)
        for c, w, txt in (("t", 55, "час"), ("v", 50, "знач."), ("k", 110, "тип (2×клік — змінити)")):
            self.tree.heading(c, text=txt); self.tree.column(c, width=w)
        self.tree.pack(fill="x"); self.tree.bind("<Double-1>", self.toggle_type)
        for txt, m in (("Замінити середнім періоду", "period"), ("Замінити загальним середнім", "global"),
                       ("Видалити", "delete")):
            ttk.Button(box, text=txt, command=lambda m=m: self.apply_clean(m)).pack(fill="x")

        box = ttk.LabelFrame(left, text="3. Параметри"); box.pack(fill="x", pady=3)
        self.gamma_var, self.k_var, self.step_var = tk.StringVar(value="0.95"), tk.StringVar(), tk.StringVar(value="5")
        for lab, var in (("γ (довірча ймовірність)", self.gamma_var),
                         ("к-ть класів (порожньо = Стерджес)", self.k_var),
                         ("«ручний» крок h", self.step_var)):
            r = ttk.Frame(box); r.pack(fill="x"); ttk.Label(r, text=lab).pack(side="left")
            ttk.Entry(r, textvariable=var, width=7).pack(side="right")
        ttk.Button(box, text="Перерахувати та побудувати", command=self.refresh).pack(fill="x")

        box = ttk.LabelFrame(left, text="4. Перетворення"); box.pack(fill="x", pady=3)
        ttk.Button(box, text="Логарифмування ln x", command=self.do_log).pack(fill="x")
        ttk.Button(box, text="Стандартизація (x−x̄)/S", command=self.do_std).pack(fill="x")
        r = ttk.Frame(box); r.pack(fill="x"); self.c_var = tk.StringVar(value="0")
        ttk.Button(r, text="Зсув x+c, c =", command=self.do_shift).pack(side="left")
        ttk.Entry(r, textvariable=self.c_var, width=7).pack(side="left")
        ttk.Button(box, text="Скинути всі зміни", command=self.reset).pack(fill="x")

        self.info = tk.Label(left, justify="left", anchor="w", bg="#f4f4f4", relief="sunken", wraplength=250)
        self.info.pack(fill="x", pady=3)

        # графіки таблиці праворуч
        self.nb = ttk.Notebook(right); self.nb.pack(fill="both", expand=True)
        self.figs = {}
        for key, name in (("ts", "Часовий ряд"), ("hist", "Гістограми"), ("cdf", "Функція розподілу")):
            fr = ttk.Frame(self.nb); self.nb.add(fr, text=name)
            fig = Figure(figsize=(9, 4.4), dpi=100)
            cv = FigureCanvasTkAgg(fig, master=fr); cv.get_tk_widget().pack(fill="both", expand=True)
            self.figs[key] = (fig, cv)

        bot = ttk.Frame(right); bot.pack(fill="x")
        self.tab = ttk.Treeview(bot, columns=("c", "lo", "th", "hi", "sg"), show="headings", height=6)
        for c, txt, w in (("c", "Характеристика", 130), ("lo", "θн", 90), ("th", "θ", 90), ("hi", "θв", 90), ("sg", "σθ", 90)):
            self.tab.heading(c, text=txt); self.tab.column(c, width=w)
        self.tab.pack(side="left")
        self.txt = tk.Text(bot, height=9, width=70, font=("Consolas", 9)); self.txt.pack(side="left", fill="both", expand=True)

    # допоміжне
    def err(self, e):
        messagebox.showerror("Помилка", str(e))

    def _need_data(self):
        if self.raw_v is None:
            raise ValueError("Спочатку відкрийте файл з даними")

    def _gamma(self):
        try:
            g = float(self.gamma_var.get().replace(",", "."))
        except ValueError:
            raise ValueError("γ має бути числом")
        if not 0 < g < 1:
            raise ValueError("γ має бути в межах інтервалу (0; 1)")
        return g

    def _bounds(self):
        try:
            lo, hi = (float(s.get().replace(",", ".")) for s in (self.lo_var, self.hi_var))
        except ValueError:
            raise ValueError("Межі аномалій мають бути числами")
        if lo >= hi:
            raise ValueError("Нижня межа має бути меншою за верхню")
        return lo, hi

    # дії
    def open_file(self):
        p = filedialog.askopenfilename(filetypes=[("Дані", "*.xlsx *.xls *.csv *.txt"), ("Усі", "*.*")])
        if not p:
            return
        self.path = p
        try:
            if p.lower().endswith((".xlsx", ".xls", ".xlsm")):
                sheets = dp.list_sheets(p)
                self.sheet_cb["values"] = sheets
                self.sheet_var.set(next((s for s in sheets if "10" in s), sheets[-1]))
            else:
                self.sheet_cb["values"] = []; self.sheet_var.set("")
            self.load()
        except Exception as e:
            self.err(e)

    def load(self):
        try:
            t, v, bad, title = dp.load_file(self.path, self.sheet_var.get() or None)
        except Exception as e:
            return self.err(e)
        self.raw_t, self.raw_v = t, v
        self.bad = bad
        self.reset(silent=True)
        if bad:
            messagebox.showwarning("Увага", f"{bad} нечислових значень сприйнято як пропуски")
        self.detect()

    def detect(self):
        try:
            self._need_data(); lo, hi = self._bounds()
        except Exception as e:
            return self.err(e)
        idx = dp.find_anomalies(self.raw_v, lo, hi)
        self.types = dp.suggest_types(idx)
        self._fill_tree(); self.refresh()

    def _fill_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, k in sorted(self.types.items()):
            self.tree.insert("", "end", iid=str(i), values=(
                self.raw_t[i], f"{self.raw_v[i]:g}", "реальна подія" if k == "real" else "технічна помилка"))

    def toggle_type(self, _):
        sel = self.tree.selection()
        if sel:
            i = int(sel[0]); self.types[i] = "real" if self.types[i] == "technical" else "technical"
            self._fill_tree(); self.refresh()

    def apply_clean(self, method):
        try:
            self._need_data()
            if self.transformed:
                raise ValueError("Спочатку скиньте всі зміни»")
            tech = [i for i, k in self.types.items() if k == "technical"]
            t, v, log = dp.clean(self.raw_t, self.raw_v, tech, method)
        except Exception as e:
            return self.err(e)
        self.cur_t, self.cur_v = t, v
        name = {"period": "середнє періоду", "global": "загальне середнє", "delete": "видалення"}[method]
        self._log = [f"{a}: {'—' if np.isnan(b) else format(b, 'g')} → "
                     f"{'видалено' if c is None else format(c, '.2f')}  ({d})" for a, b, c, d in log]
        self._stage(f"після обробки ({name})")
        self.refresh()

    def _transform(self, fn, label):
        try:
            self._need_data()
            y = fn(np.asarray(self.cur_v, float))
        except Exception as e:
            return self.err(e)
        self.cur_v = y; self.transformed = True
        self._stage(label); self.refresh()

    def do_log(self):
        def f(v):
            y, sh = dp.log_transform(v)
            if sh:
                messagebox.showinfo("Зсув", f"Були значення <= 0, виконано зсув на c = {sh:g} перед ln")
            return y
        self._transform(f, "ln x")

    def do_std(self):
        self._transform(dp.standardize, "стандартизація")

    def do_shift(self):
        try:
            c = float(self.c_var.get().replace(",", "."))
        except ValueError:
            return self.err("c має бути числом")
        self._transform(lambda v: dp.shift_by(v, c), f"зсув c={c:g}")

    def reset(self, silent=False):
        if self.raw_v is None:
            return
        self.cur_t, self.cur_v = list(self.raw_t), self.raw_v.copy()
        self.transformed = False; self.stages = []; self._log = []
        self._stage("вихідні (без пропусків)")
        if not silent:
            self.refresh()

    def _stage(self, label):
        try:
            x = [v for v in self.cur_v if not np.isnan(v)]
            rows, _ = ms.characteristics(x, 0.95)
            self.stages.append((label, len(x), rows[0][2], rows[1][2], rows[3][2], rows[4][2]))
        except Exception:
            pass

    # оновлення виводу
    def refresh(self):
        if self.raw_v is None:
            return
        try:
            gamma = self._gamma(); lo, hi = self._bounds()
            x = [float(v) for v in self.cur_v if not np.isnan(v)]
            rows, tq = ms.characteristics(x, gamma)
            k = int(self.k_var.get()) if self.k_var.get().strip() else ms.sturges_k(len(x))
            if k < 1:
                raise ValueError("Кількість класів має бути >= 1")
            step = float(self.step_var.get().replace(",", "."))
        except ValueError as e:
            return self.err(e)

        miss = len(dp.find_missing(self.cur_v)); anom = len(dp.find_anomalies(self.cur_v, lo, hi))
        self.info.config(text=f"Записів: {len(self.cur_v)}   Пропусків: {miss}\n"
                              f"Аномалій (за межами [{lo:g}; {hi:g}]): {anom}\n"
                              f"Класифіковано: {sum(1 for v in self.types.values() if v == 'technical')} техн., "
                              f"{sum(1 for v in self.types.values() if v == 'real')} реальних")
        self.tab.delete(*self.tab.get_children())
        for name, a, b, c, sg in rows:
            self.tab.insert("", "end", values=(name, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}", f"{sg:.4f}"))

        # графіки
        raw = (self.raw_t, self.raw_v) if not self.transformed else None
        viz.plot_timeseries(self.figs["ts"][0].subplots() if not self.figs["ts"][0].axes else self.figs["ts"][0].axes[0],
                            self.cur_t, self.cur_v, raw=raw, bounds=(lo, hi) if raw else None,
                            types=self.types if raw else None)
        fh = self.figs["hist"][0]; fh.clear(); a1, a2 = fh.subplots(1, 2)
        try:
            viz.plot_histogram(a1, x, ms.edges_by_k(x, k), "Стерджес" if not self.k_var.get().strip() else "задано k", gamma)
            viz.plot_histogram(a2, x, ms.edges_by_step(x, step), "ручний крок", gamma)
        except Exception as e:
            return self.err(e)
        fh.tight_layout()
        fc = self.figs["cdf"][0]; fc.clear(); viz.plot_cdf(fc.subplots(), x, gamma)
        self.figs["ts"][0].tight_layout()
        for fig, cv in self.figs.values():
            cv.draw()

        # текстова панель
        s = [f"γ={gamma}, t={tq:.4f}, ν={len(x)-1}. Стерджес: k={ms.sturges_k(len(x))} (1+3.322·lg{len(x)}="
             f"{1 + 3.322 * np.log10(len(x)):.3f}); використано k={k}"]
        ed = ms.edges_by_k(x, k); cnt, rel = ms.histogram(x, ed)
        s.append("Класи: " + "; ".join(f"[{ed[i]:.1f};{ed[i+1]:.1f}) n={cnt[i]}" for i in range(k)))
        A, E = rows[3], rows[4]
        s.append(f"A: інтервал {'накриває' if A[1] <= 0 <= A[3] else 'НЕ накриває'} 0 → "
                 f"{'симетричний' if A[1] <= 0 <= A[3] else 'асиметричний (' + ('правобічний' if A[2] > 0 else 'лівобічний') + ')'}")
        s.append(f"E: інтервал {'накриває' if E[1] <= 0 <= E[3] else 'НЕ накриває'} 0 → "
                 f"{'нормальна гостроверхість' if E[1] <= 0 <= E[3] else ('гостровершинний' if E[2] > 0 else 'плосковершинний')}")
        if not self.transformed and len(self.cur_t) == len(x):
            pm = [[v for t, v in zip(self.cur_t, x) if dp.period_index(t) == p] for p in range(4)]
            s.append("Періоди: " + "; ".join(f"{dp.PERIOD_NAMES[p].split()[0]} {np.mean(q):.2f}" for p, q in enumerate(pm) if q))
        s.append("— етапи —")
        s += [f"{l}: n={n} x̄={m:.3f} S²={v:.2f} A={a:.3f} E={e:.3f}" for l, n, m, v, a, e in self.stages]
        s += getattr(self, "_log", [])
        self.txt.delete("1.0", "end"); self.txt.insert("1.0", "\n".join(s))
