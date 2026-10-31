"""gui_lab4.py - вклдака ЛР4 - коефіцієнти Пірсона, Спірмена, лінійна регресія, R², 3D-щільність"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math_statistics as ms
import data_processing as dp
import visualization as viz
from gui_lab2 import _table, _fmt
class RegTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.x = self.y = None
        top = ttk.Frame(self); top.pack(fill="x", pady=3)
        ttk.Button(top, text="Завантажити X, Y з файлу…", command=self.load_xy).pack(side="left")
        ttk.Button(top, text="Розрахувати", command=self.run).pack(side="left", padx=6)
        self.lab = ttk.Label(top, text="Дані не завантажено", foreground="#064"); self.lab.pack(side="left", padx=6)

        ttk.Label(self, text="Оцінювання зв'язку між X і Y", font=("Arial", 9, "bold")).pack(anchor="w")
        self.tcor = _table(self, [("c", "Коефіцієнт"), ("v", "Оцінка"), ("lo", "θн"), ("hi", "θв"), ("t", "t"),
                                  ("tc", "tкр"), ("p", "p-value"), ("h", "Висновок")],
                           (110, 70, 70, 70, 70, 70, 80, 300), 2)
        self.tcor.pack(fill="x")
        ttk.Label(self, text="Лінійна регресія ŷ = â + b̂X", font=("Arial", 9, "bold")).pack(anchor="w")
        self.treg = _table(self, [("p", "Параметр"), ("v", "Оцінка"), ("sg", "σ (СКВ)"), ("lo", "θн"), ("hi", "θв"),
                                  ("t", "t"), ("tc", "tкр"), ("p2", "p-value"), ("h", "Висновок")],
                           (80, 80, 80, 80, 80, 70, 70, 80, 300), 2)
        self.treg.pack(fill="x")
        self.r2_lab = ttk.Label(self, text="", font=("Arial", 9, "bold")); self.r2_lab.pack(anchor="w")

        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True)
        self.figs = {}
        for key, name in (("reg", "Кореляційне поле і регресія"), ("3d", "Щільність 3D")):
            fr = ttk.Frame(nb); nb.add(fr, text=name)
            fig = Figure(figsize=(8, 3.8), dpi=100)
            cv = FigureCanvasTkAgg(fig, master=fr); cv.get_tk_widget().pack(fill="both", expand=True)
            self.figs[key] = (fig, cv)
        self.txt = tk.Text(self, height=9, font=("Consolas", 9)); self.txt.pack(fill="x")
        # перерахунок при зміні γ без перезапуску
        app.gamma_var.trace_add("write", lambda *a: self.run(silent=True))

    # дані
    def set_xy(self, x, y):
        self.x, self.y = [float(v) for v in x], [float(v) for v in y]
        self.lab.config(text=f"Завантажено пар: {len(self.x)}")
        self.run()

    def load_xy(self):
        p = filedialog.askopenfilename(filetypes=[("Дані", "*.xlsx *.xls *.csv *.txt"), ("Усі", "*.*")])
        if not p:
            return
        try:
            if p.lower().endswith((".xlsx", ".xlsm", ".xls")):
                df = pd.read_excel(p, header=None, dtype=str)
            else:
                df = pd.read_csv(p, header=None, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
            if df.shape[1] < 2:
                raise ValueError("Потрібно щонайменше 2 стовпці (X, Y)")
            pairs = [(dp.to_float(a)[0], dp.to_float(b)[0]) for a, b in zip(df[0], df[1])]
            pairs = [(a, b) for a, b in pairs if not (np.isnan(a) or np.isnan(b))]   # заголовки й пропуски відпадають
            if not pairs:
                raise ValueError("У перших двох стовпцях немає числових пар")
            self.set_xy([a for a, _ in pairs], [b for _, b in pairs])
        except Exception as ex:
            messagebox.showerror("Помилка", str(ex))

    # розрахунок
    def run(self, silent=False):
        try:
            if self.x is None:
                raise ValueError("Спочатку завантажте дані X, Y")
            gamma = self.app._gamma(); alpha = 1 - gamma
            x, y, removed = ms.drop_outliers_3s(self.x, self.y)
            n = len(x)
            if n < 4:
                raise ValueError("Після відкидання аномалій лишилось менше 4 пар")
            r, rho = ms.pearson(x, y), ms.spearman(x, y)
            t_r, tc, p_r, ok_r = ms.corr_test(r, n, alpha)
            t_s, _, p_s, ok_s = ms.corr_test(rho, n, alpha)
            lo, hi = ms.fisher_interval(r, n, gamma)
            reg = ms.linear_regression(x, y, gamma)
        except ValueError as ex:
            if not silent:
                messagebox.showerror("Помилка", str(ex))
            return
        yn = lambda ok: "значущий" if ok else "незначущий"
        self.tcor.delete(*self.tcor.get_children())
        self.tcor.insert("", "end", values=("Пірсона r̂", f"{r:.4f}", f"{lo:.4f}", f"{hi:.4f}", _fmt(t_r), _fmt(tc),
                                            _fmt(p_r), f"H₀: r = 0 {'відхиляємо' if ok_r else 'не відхиляємо'} — {yn(ok_r)}"))
        self.tcor.insert("", "end", values=("Спірмена ρ̂", f"{rho:.4f}", "—", "—", _fmt(t_s), _fmt(tc),
                                            _fmt(p_s), f"H₀: ρ = 0 {'відхиляємо' if ok_s else 'не відхиляємо'} — {yn(ok_s)}"))
        self.treg.delete(*self.treg.get_children())
        for nm, key, sg in (("a (вільний член)", "a", "s_a"), ("b (нахил)", "b", "s_b")):
            ok = reg["ok_" + key]
            self.treg.insert("", "end", values=(nm, f"{reg[key]:.5g}", f"{reg[sg]:.4g}", f"{reg[key + '_lo']:.5g}",
                                                f"{reg[key + '_hi']:.5g}", _fmt(reg["t_" + key]), _fmt(reg["tq"]),
                                                _fmt(reg["p_" + key]),
                                                f"H₀: {key} = 0 {'відхиляємо' if ok else 'не відхиляємо'} — {yn(ok)}"))
        self.r2_lab.config(text=f"R² = {reg['R2']:.4f} ({100 * reg['R2']:.2f} %);  Sзал = {reg['s_res']:.4g};  "
                                f"ν = N − 2 = {reg['nu']};  γ = {gamma:g}, α = {alpha:g}")
        # графіки
        fig, cv = self.figs["reg"]; fig.clear()
        viz.plot_regression(fig.subplots(), x, y, reg, removed); fig.tight_layout(); cv.draw()
        fig, cv = self.figs["3d"]; fig.clear()
        try:
            viz.plot_density3d(fig.add_subplot(projection="3d"), x, y)
        except ValueError as ex:
            fig.clear(); fig.text(.1, .5, str(ex))
        cv.draw()
        # висновки
        direction = "прямий (додатний)" if r > 0 else "зворотний (від'ємний)"
        s = [f"N = {len(self.x)}, відкинуто за правилом 3σ: {len(removed)}, для аналізу: {n}.",
             f"1) r̂ = {r:.4f}, інтервал [{lo:.4f}; {hi:.4f}] (γ = {gamma:g}); t = {t_r:.3f}, tкр = t(1−α/2; {n - 2}) = {tc:.3f}, "
             f"p = {p_r:.4g}.",
             (f"   |t| > tкр → H₀: r = 0 відхиляємо: лінійний зв'язок значущий, {direction}, "
              f"за шкалою Чеддока {ms.chaddock(r)} (|r̂| = {abs(r):.3f})."
              if ok_r else f"   |t| ≤ tкр → H₀: r = 0 не відхиляємо: лінійний зв'язок статистично не підтверджено."),
             f"2) ρ̂ = {rho:.4f}, t = {t_s:.3f}, p = {p_s:.4g}: монотонний зв'язок {yn(ok_s)}, {ms.chaddock(rho)}."
             + ("" if n >= 10 else "  (N < 10 — перевірка значущості ρ̂ за t наближена)"),
             f"   |r̂ − ρ̂| = {abs(r - rho):.3f}: " + ("близькі → зв'язок лінійний, викидів немає." if abs(r - rho) < 0.1
                                                       else "суттєва різниця → можливі викиди або нелінійність."),
             f"3) ŷ = {reg['a']:.4f} + {reg['b']:.4f}·x; b̂: +1 до x змінює y в середньому на {reg['b']:.4f}.",
             f"   a: t = {reg['t_a']:.3f} → {yn(reg['ok_a'])}; b: t = {reg['t_b']:.3f} → {yn(reg['ok_b'])} (tкр = {reg['tq']:.3f}).",
             (f"   H₀: a = 0 {'відхиляємо' if reg['ok_a'] else 'не відхиляємо — лінія може проходити через початок координат'}."),
             f"4) R² = {reg['R2']:.4f}: модель пояснює {100 * reg['R2']:.1f} % розкиду Y, "
             f"{100 * (1 - reg['R2']):.1f} % — неврахованими чинниками. Перевірка: r̂² = {r * r:.4f}."]
        self.txt.delete("1.0", "end"); self.txt.insert("1.0", "\n".join(s))