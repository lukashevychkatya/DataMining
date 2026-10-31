"""gui_lab3.py — вкладка ЛР3: двовимірний розподіл."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import twodim as td
import viz3
VARIANT = 10
DEFAULTS = dict(mx="18", my="59", sx="7", sy="2.5", r="-0.45", N="250", gamma="0.95", k="7")
class Lab3Tab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.x = self.y = None
        self.true = None
        self.gen_count = 0
        self.v = {k: tk.StringVar(value=val) for k, val in DEFAULTS.items()}
        self.seed_var = tk.StringVar(value=str(VARIANT * td.VARIANT_SEED_FACTOR))
        left = ttk.LabelFrame(self, text="Параметри варіанта 10"); left.pack(side="left", fill="y", padx=4, pady=3)
        labels = [("mx", "mx"), ("my", "my"), ("sx", "σx"), ("sy", "σy"), ("r", "rxy"), ("N", "N"),
                  ("gamma", "γ (рівень довіри)"), ("k", "k (класів на вісь)")]
        for key, lab in labels:
            r_ = ttk.Frame(left); r_.pack(fill="x"); ttk.Label(r_, text=lab, width=20).pack(side="left")
            ttk.Entry(r_, textvariable=self.v[key], width=8).pack(side="right")
        r_ = ttk.Frame(left); r_.pack(fill="x"); ttk.Label(r_, text="seed (варіант×100)", width=20).pack(side="left")
        ttk.Entry(r_, textvariable=self.seed_var, width=8).pack(side="right")
        ttk.Button(left, text="Згенерувати вибірку", command=self.generate).pack(fill="x", pady=2)
        ttk.Button(left, text="Відкрити файл вибірки ", command=self.open_file).pack(fill="x", pady=2)
        ttk.Button(left, text="Зберегти вибірку ", command=self.save_csv).pack(fill="x", pady=2)
        ttk.Button(left, text="Обчислити і побудувати", command=self.compute).pack(fill="x", pady=6)
        self.status = ttk.Label(left, text="Спершу згенеруйте або відкрийте вибірку ", wraplength=190)
        self.status.pack(pady=4)

        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.nb = ttk.Notebook(right); self.nb.pack(fill="both", expand=True)
        self.figs = {}
        for key, name in (("sc", "Кореляційне поле"), ("hm", "Теплова карта"), ("b3", "3D-гістограма"), ("chi", "χ²: порівняння")):
            fr = ttk.Frame(self.nb); self.nb.add(fr, text=name)
            fig = Figure(figsize=(8, 4.6), dpi=100)
            cv = FigureCanvasTkAgg(fig, master=fr); cv.get_tk_widget().pack(fill="both", expand=True)
            self.figs[key] = (fig, cv)
        self.tab = ttk.Treeview(right, columns=("p", "lo", "th", "hi"), show="headings", height=6)
        for c, t_, w in (("p", "Показник", 150), ("lo", "θн", 110), ("th", "θ̂", 110), ("hi", "θв", 110)):
            self.tab.heading(c, text=t_); self.tab.column(c, width=w, anchor="center")
        self.tab.pack(fill="x")
        self.txt = tk.Text(right, height=9, font=("Consolas", 9)); self.txt.pack(fill="x")

    # допоміжне
    def _params(self):
        try:
            p = {k: float(v.get().replace(",", ".")) for k, v in self.v.items()}
            p["N"] = int(p["N"]); p["k"] = int(p["k"])
        except ValueError:
            raise ValueError("Усі параметри мають бути числами (N, k — цілі).")
        td.check_params(p["sx"], p["sy"], p["r"], p["N"])
        if not 0 < p["gamma"] < 1:
            raise ValueError("γ має бути в інтервалі (0; 1).")
        if p["k"] < 2:
            raise ValueError("k має бути ≥ 2.")
        return p

    def _fresh(self, key):
        """Очищена фігура з одним полем осей (щоб графіки не накладалися)"""
        fig = self.figs[key][0]; fig.clear()
        return fig.add_subplot(111)

    def _err(self, e):
        messagebox.showerror("Помилка", str(e))

    # дії
    def generate(self):
        try:
            p = self._params()
            seed = int(self.seed_var.get()) + self.gen_count     # новий seed при кожній генерації
            self.gen_count += 1
            self.x, self.y = td.generate(p["mx"], p["my"], p["sx"], p["sy"], p["r"], p["N"], seed)
            self.true = p
            self.status.config(text=f"Згенеровано N = {p['N']}, seed = {seed}")
        except ValueError as e:
            return self._err(e)
        self.compute()

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("Дані", "*.xlsx *.txt *.csv"), ("Усі", "*.*")])
        if not path:
            return
        try:
            self.x, self.y = td.read_pairs(path)
            self.true = None
            self.status.config(text=f"Завантажено N = {len(self.x)} з файлу (параметри варіанта не порівнюються)")
        except Exception as e:
            return self._err(e)
        self.compute()

    def save_csv(self):
        if self.x is None:
            return messagebox.showinfo("Збереження", "Спершу згенеруйте вибірку.")
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            np.savetxt(path, np.column_stack([self.x, self.y]), delimiter=";", header="x;y", comments="", fmt="%.6f")

    def compute(self):
        if self.x is None:
            return
        try:
            p = self._params()
            if self.true is None:      # файл: параметри беремо з оцінок (для χ² і density)
                e = td.estimates(self.x, self.y)
                p.update(mx=e["x̄"], my=e["ȳ"], sx=np.sqrt(e["Sx²"]), sy=np.sqrt(e["Sy²"]), r=e["rxy"])
            rows, e = td.intervals(self.x, self.y, p["gamma"])
            tres, tc, rej = td.r_test(e["rxy"], len(self.x), p["gamma"])
            chi, n, Np, cx, cy, ex, ey = td.chi2_test(self.x, self.y, p["k"], p["mx"], p["my"],
                                                     p["sx"], p["sy"], p["r"], 1 - p["gamma"])
        except ValueError as ex_:
            return self._err(ex_)
        # графіки (кожен раз з чистої фігури)
        P = n / len(self.x)
        viz3.plot_scatter(self._fresh("sc"), self.x, self.y, p["mx"], p["my"], p["sx"], p["sy"], p["r"])
        viz3.plot_heat(self._fresh("hm"), n, P, ex, ey)
        viz3.plot_bar3d(self.figs["b3"][0], n, P, ex, ey)
        viz3.plot_chi_compare(self.figs["chi"][0], n, Np, cx, cy)
        for key, (fig, cv) in self.figs.items():
            fig.tight_layout(); cv.draw()
        # таблиця
        self.tab.delete(*self.tab.get_children())
        for name, lo, th, hi in rows:
            self.tab.insert("", "end", values=(name, f"{lo:.4f}", f"{th:.4f}", f"{hi:.4f}"))
        cov = td.covers(rows, p) if self.true is not None else {}
        s = [f"Оцінки: x̄={e['x̄']:.3f}, ȳ={e['ȳ']:.3f}, Sx²={e['Sx²']:.3f}, Sy²={e['Sy²']:.3f}, "
             f"Kxy={e['Kxy']:.3f}, r̂={e['rxy']:.4f}",
             "Накривають задані параметри: " + (", ".join(k + (": так" if v else ": ні") for k, v in cov.items()) or "—"),
             f"Кореляція: t = {tres:.3f}, tкр = {tc:.3f} → "
             f"{'зв’язок значущий (H₀: r=0 відхиляється)' if rej else 'зв’язок не значущий'}",
             f"χ² = {chi['χ²']:.3f}, ν = {chi['ν']}, χ²крит = {chi['крит']:.3f} "
             f"(груп {chi['груп']}, малих класів об’єднано: {chi['малих']}) → "
             f"{'відтворення адекватне' if chi['адекватно'] else 'відтворення неадекватне'}"]
        self.txt.delete("1.0", "end"); self.txt.insert("1.0", "\n".join(s))