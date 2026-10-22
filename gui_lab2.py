"""gui_lab2.py -вкладки ЛР2: Закон розподілу і Моделювання"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math_statistics as ms
import distributions as dist
import simulation as sim
import visualization as viz

def _table(master, cols, widths, height):
    tv = ttk.Treeview(master, columns=[c for c, _ in cols], show="headings", height=height)
    for (c, txt), w in zip(cols, widths):
        tv.heading(c, text=txt); tv.column(c, width=w, anchor="center")
    return tv

def _fmt(v):
    return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)

class LawTab(ttk.Frame):
    """Визначення закону розподілу за даними: A, E, оцінки трьох законів, D, графіки"""
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.est = {}  # {закон: {параметр: значення}}
        self.best = None
        top = ttk.Frame(self); top.pack(fill="x", pady=3)
        ttk.Label(top, text="α (γ = 1 − α):").pack(side="left")
        ttk.Entry(top, textvariable=app.alpha_var, width=6).pack(side="left", padx=3)
        ttk.Button(top, text="Визначити закон розподілу", command=self.run).pack(side="left", padx=6)
        self.lab = ttk.Label(top, text="", foreground="#064"); self.lab.pack(side="left", padx=6)

        mid = ttk.Frame(self); mid.pack(fill="x")
        left = ttk.Frame(mid); left.pack(side="left", padx=4)
        self.shape_lab = ttk.Label(left, text="", font=("Arial", 8)); self.shape_lab.pack(anchor="w")
        self.tshape = _table(left, [("l", "Закон"), ("a", "A теор."), ("ao", "A в інтервалі?"),
                                    ("e", "E теор."), ("eo", "E в інтервалі?")], (120, 60, 100, 60, 100), 3)
        self.tshape.pack()
        right = ttk.Frame(mid); right.pack(side="left", padx=10)
        self.test = _table(right, [("l", "Закон"), ("p", "Параметр"), ("lo", "θн"), ("th", "θ̂"), ("hi", "θв"),
                                   ("sg", "σ{θ̂}"), ("d", "D")], (110, 60, 80, 80, 80, 70, 70), 6)
        self.test.pack()
        self.fig = Figure(figsize=(9, 3.6), dpi=100)
        self.cv = FigureCanvasTkAgg(self.fig, master=self); self.cv.get_tk_widget().pack(fill="both", expand=True)
        self.txt = tk.Text(self, height=6, font=("Consolas", 9)); self.txt.pack(fill="x")

    def run(self):
        try:
            x = self.app.values_clean()
            alpha = float(self.app.alpha_var.get().replace(",", "."))
            if not 0 < alpha < 1:
                raise ValueError("α має бути в інтервалі (0; 1).")
            gamma = 1 - alpha
            k = self.app.get_k()
            (a_lo, a, a_hi), (e_lo, e, e_hi), shape = dist.shape_table(x, gamma)
        except ValueError as ex:
            return messagebox.showerror("Помилка", str(ex))
        V = dist.variation_coeff(x)

        self.tshape.delete(*self.tshape.get_children())
        for law, tA, okA, tE, okE in shape:
            self.tshape.insert("", "end",
                               values=(law, tA, "так" if okA else "ні", tE, "так" if okE else "ні"))
        self.shape_lab.config(text=f"A ∈ [{a_lo:.3f}; {a_hi:.3f}] (A = {a:.3f}),  "
                                   f"E ∈ [{e_lo:.3f}; {e_hi:.3f}] "
                                   f"(E = {e:.3f}),  V = S/x̄ = {V:.3f},  γ = {gamma:.2f}")
        self.test.delete(*self.test.get_children())
        self.est, D, notes = {}, {}, []
        for law in dist.LAWS:
            try:
                self.est[law] = dist.estimates_dict(law, x)
                D[law] = dist.distance_D(law, x, self.est[law])
                for r in dist.intervals(law, x, gamma):
                    self.test.insert("", "end",
                                     values=(r[0], r[1], _fmt(r[2]), _fmt(r[3]), _fmt(r[4]), _fmt(r[5]),
                                                        f"{D[law]:.4f}"))
            except ValueError as ex:
                notes.append(f"{law}: {ex}")
        if not D:
            return messagebox.showerror("Помилка", "; ".join(notes))
        self.best = min(D, key=D.get)
        self.lab.config(text=f"Найменше D: {self.best} (D = {D[self.best]:.4f})")
        edges = ms.edges_by_k(x, k)
        self.fig.clear()
        viz.plot_laws(*self.fig.subplots(1, 2), x, edges, self.est)
        self.fig.tight_layout(); self.cv.draw()

        fit_AE = [law for law, _, okA, _, okE in shape if okA and okE]
        s = ["D = max|Fₙ(x) − F(x)|: " + ", ".join(f"{l}: {d:.4f}" for l, d in D.items()),
             f"Обраний закон (мін. D): {self.best}; θ̂ = " + ", ".join(f"{k_} = {v:.4g}"
                                                                       for k_, v in self.est[self.best].items()),
             "Узгоджуються з A та E одночасно: " + (", ".join(fit_AE) if fit_AE else "жоден закон повністю"),
             f"V = {V:.3f}: для експоненціального закону V має бути ≈ 1"]
        s += notes
        self.txt.delete("1.0", "end"); self.txt.insert("1.0", "\n".join(s))
        self.app.sim_tab.on_law_ready()

class SimTab(ttk.Frame):
    """Моделювання методом зворотної функції, T-тест, R повторень, δ(N)"""
    def __init__(self, master, app):
        super().__init__(master)
        self.base_seed_var = tk.StringVar(value="12345")
        self.app = app
        self.rows = self.tvals = self.delta = self.req = None
        self.header = ""
        self.law_var = tk.StringVar(value=dist.LAWS[0])
        self.p1, self.p2 = tk.StringVar(), tk.StringVar()
        self.R_var, self.N_var = tk.StringVar(value="300"), tk.StringVar(value="50, 400, 1000")
        self.seed_var, self.n1_var = tk.StringVar(value="12345"), tk.StringVar(value="1000")
        self.param_var = tk.StringVar()
        self.run_no = 0
        self.run_no = 0
        left = ttk.LabelFrame(self, text="Параметри експерименту"); left.pack(side="left", fill="y", padx=4, pady=3)
        ttk.Label(left, text="Закон").pack(anchor="w")
        cb = ttk.Combobox(left, textvariable=self.law_var, values=dist.LAWS, state="readonly", width=20)
        cb.pack(); cb.bind("<<ComboboxSelected>>", lambda e: self._law_changed())
        self.l1, self.l2 = ttk.Label(left, text="m"), ttk.Label(left, text="σ")
        self.e1, self.e2 = ttk.Entry(left, textvariable=self.p1, width=12), ttk.Entry(left, textvariable=self.p2, width=12)
        self.l1.pack(anchor="w"); self.e1.pack(); self.l2.pack(anchor="w"); self.e2.pack()
        ttk.Button(left, text="Взяти оцінки з даних", command=self.take_estimates).pack(fill="x", pady=3)
        for lab, var in (("R (повторень, 200–500)", self.R_var), ("α (рівень значущості)", self.app.alpha_var),
                         ("Набір обсягів N", self.N_var), ("seed", self.seed_var), ("N для однієї вибірки", self.n1_var)):
            ttk.Label(left, text=lab).pack(anchor="w"); ttk.Entry(left, textvariable=var, width=18).pack()
        ttk.Button(left, text="Одна вибірка", command=self.one_sample).pack(fill="x", pady=(8, 2))
        ttk.Button(left, text="Запустити експеримент", command=self.run).pack(fill="x")
        ttk.Button(left, text="Експорт таблиць", command=self.export).pack(fill="x", pady=2)
        self.pb = ttk.Progressbar(left, length=160); self.pb.pack(pady=3)
        self.status = ttk.Label(left, text="", wraplength=170); self.status.pack()

        right = ttk.Frame(self); right.pack(side="left", fill="both", expand=True)
        self.inb = ttk.Notebook(right); self.inb.pack(fill="both", expand=True)
        self.figs = {}
        for key, name in (("s", "Вибірка"), ("t", "Гістограми t"), ("d", "δ(N)")):
            fr = ttk.Frame(self.inb); self.inb.add(fr, text=name)
            fig = Figure(figsize=(8, 3.6), dpi=100)
            cv = FigureCanvasTkAgg(fig, master=fr); cv.get_tk_widget().pack(fill="both", expand=True)
            self.figs[key] = (fig, cv)

        bar = ttk.Frame(right); bar.pack(fill="x")
        ttk.Label(bar, text="Параметр:").pack(side="left")
        self.pcb = ttk.Combobox(bar, textvariable=self.param_var, state="readonly", width=6)
        self.pcb.pack(side="left"); self.pcb.bind("<<ComboboxSelected>>", lambda e: self._show_param())
        self.head_lab = ttk.Label(bar, text="", font=("Arial", 8)); self.head_lab.pack(side="left", padx=8)
        self.tsum = _table(right, [("N", "N"), ("nu", "ν"), ("tc", "tкр"), ("mt", "середня θ̂"), ("sd", "СКВ θ̂"),
                                   ("sg", "σ{θ̂} теор."), ("mean_t", "середнє t"), ("sd_t", "СКВ t"),
                                   ("rej", "відхилень H₀, %")], (60, 60, 70, 90, 80, 90, 80, 70, 110), 4)
        self.tsum.pack(fill="x")
        self.treq = _table(right, [("p", "Параметр"), ("a", "δ ≤ 10 %"), ("b", "δ ≤ 5 %"), ("c", "δ ≤ 2 %"),
                                   ("d", "δ ≤ 1 %")], (80, 100, 100, 100, 100), 2)
        ttk.Label(right, text="Необхідний обсяг N (півширина довірчого інтервалу ≤ δ):", font=("Arial", 8)).pack(anchor="w")
        self.treq.pack(fill="x")
        self._law_changed()


    # допоміжне
    def _law_changed(self):
        names = dist.PARAMS[self.law_var.get()]
        self.l1.config(text=names[0]); self.e1.config(state="normal")
        if len(names) == 2:
            self.l2.config(text=names[1]); self.e2.config(state="normal")
        else:
            self.l2.config(text="—"); self.p2.set(""); self.e2.config(state="disabled")
        self.pcb.config(values=names); self.param_var.set(names[0])

    def on_law_ready(self):
        """Після визначення закону підставляємо найкращий закон і його оцінки."""
        if self.app.law_tab.best:
            self.law_var.set(self.app.law_tab.best); self._law_changed(); self.take_estimates()

    def _theta0(self):
        law = self.law_var.get(); names = dist.PARAMS[law]
        try:
            vals = [float(v.get().replace(",", ".")) for v in (self.p1, self.p2)[:len(names)]]
        except ValueError:
            raise ValueError("Параметри закону мають бути числами.")
        return dict(zip(names, vals))

    def take_estimates(self):
        law = self.law_var.get()
        try:
            est = self.app.law_tab.est.get(law) or dist.estimates_dict(law, self.app.values_clean())
        except ValueError as ex:
            return messagebox.showerror("Помилка", str(ex))
        names = dist.PARAMS[law]
        self.p1.set(f"{est[names[0]]:.6g}")
        if len(names) == 2:
            self.p2.set(f"{est[names[1]]:.6g}")

    def _progress(self, done, total, text):
        self.pb["maximum"] = total; self.pb["value"] = done
        self.status.config(text=text); self.update_idletasks()

    def _read_common(self):
        law = self.law_var.get();
        th0 = self._theta0()
        try:
            R = int(self.R_var.get());
            alpha = float(self.app.alpha_var.get().replace(",", "."))
            base = int(self.base_seed_var.get())
        except ValueError:
            raise ValueError("R і seed мають бути цілими числами, α - числом ")
        seed = base + self.run_no  # новий запуск - seed на 1 більший за попередній
        self.run_no += 1
        Ns = sim.parse_n_set(self.N_var.get())
        sim.check_experiment(law, th0, R, alpha, Ns)
        return law, th0, R, alpha, seed, Ns

    # дії
    def one_sample(self):
        try:
            law = self.law_var.get(); th0 = self._theta0(); dist.check_params(law, th0)
            N = int(self.n1_var.get())
            seed = int(self.base_seed_var.get()) + self.run_no
            self.run_no += 1
            if N < 5:
                raise ValueError("N має бути ≥ 5.")
        except ValueError as ex:
            return messagebox.showerror("Помилка", str(ex))
        rng = np.random.default_rng(seed)
        x = sim.simulate(law, th0, N, rng)
        fig, cv = self.figs["s"]; fig.clear(); viz.plot_sim_sample(fig.subplots(), x, law, th0)
        fig.tight_layout(); cv.draw(); self.inb.select(0)
        rows = sim.sample_vs_theory(law, th0, x)
        self.status.config(text="Вибірка vs теорія:\n" + "\n".join(f"{n}: теор. {t:.4g} / вибірк. {v:.4g}" for n, t, v in rows))

    def demo_samples(self, K=5):
        """K окремих вибірок: кожна має свою гістограму, θ̂, t, tкр і висновок"""
        try:
            law = self.law_var.get(); th0 = self._theta0(); dist.check_params(law, th0)
            N = int(self.n1_var.get()); seed = int(self.seed_var.get())
            alpha = float(self.app.alpha_var.get().replace(",", "."))
            if N < 5:
                raise ValueError("N має бути ≥ 5.")
            param = self.param_var.get() or dist.PARAMS[law][0]
        except ValueError as ex:
            return messagebox.showerror("Помилка", str(ex))
        win = tk.Toplevel(self); win.title(f"{K} окремих вибірок, N = {N}, параметр {param}")
        fig = Figure(figsize=(9, 2.6 * K), dpi=90)
        cv = FigureCanvasTkAgg(fig, master=win); cv.get_tk_widget().pack(fill="both", expand=True)
        for k in range(K):
            d = sim.one_sample_decision(law, th0, N, alpha, seed + k, param)
            ax = fig.add_subplot(K, 1, k + 1)
            viz.plot_sim_sample(ax, d["x"], law, th0)
            ax.set_title(f"Вибірка {k + 1} (seed={seed + k}): θ̂={d['θ̂']:.2f}, σ{{θ̂}}={d['σ{θ̂}']:.2f}, "
                         f"t={d['t']:.3f}, tкр={d['tкр']:.3f} → {d['висновок']}", fontsize=8)
        fig.tight_layout(); cv.draw()

    def run(self):
        try:
            law, th0, R, alpha, seed, Ns = self._read_common()
            self.rows, self.tvals = sim.run_experiment(law, th0, Ns, R, alpha, seed, self._progress)
            self.delta = sim.delta_curve(law, th0, R, alpha, seed, progress=self._progress)
            self.req = sim.required_n(law, th0, alpha)
        except ValueError as ex:
            return messagebox.showerror("Помилка", str(ex))
        self.header = (f"{law}; θ₀ = " + ", ".join(f"{k} = {v:.5g}" for k, v in th0.items()) +
                       f"; R = {R}; α = {alpha}; seed = {seed}")
        self.seed_var.set(str(seed))
        self.head_lab.config(text=self.header)
        self.pcb.config(values=list(self.rows)); self.param_var.set(list(self.rows)[0])
        self.treq.delete(*self.treq.get_children())
        for name, row in self.req.items():
            self.treq.insert("", "end", values=(name, row[10], row[5], row[2], row[1]))
        fig, cv = self.figs["d"]; fig.clear(); viz.plot_delta(fig.subplots(), self.delta); fig.tight_layout(); cv.draw()
        self.status.config(text="Готово"); self._show_param()

    def _show_param(self):
        if not self.rows:
            return
        p = self.param_var.get()
        self.tsum.delete(*self.tsum.get_children())
        for r in self.rows[p]:
            self.tsum.insert("", "end", values=(r["N"], r["ν"], _fmt(r["tкр"]), _fmt(r["середня θ̂"]), _fmt(r["СКВ θ̂"]),
                                                _fmt(r["σ{θ̂} теор."]), _fmt(r["середнє t"]), _fmt(r["СКВ t"]),
                                                f"{r['відхилень H₀, %']:.2f}"))
        Ns = [r["N"] for r in self.rows[p]]
        pick = [n for n in (50, 1000) if n in Ns] or [Ns[0], Ns[-1]]
        fig, cv = self.figs["t"]; fig.clear()
        axes = fig.subplots(1, len(pick), squeeze=False)[0]
        for ax, N in zip(axes, pick):
            tcrit = next(r["tкр"] for r in self.rows[p] if r["N"] == N)
            viz.plot_t_hist(ax, self.tvals[(p, N)], tcrit, N, p)
        fig.tight_layout(); cv.draw()

    def export(self):
        if not self.rows:
            return messagebox.showinfo("Експорт", "Спочатку запустіть експеримент")
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
        if not path:
            return
        frames = {p: pd.DataFrame(r) for p, r in self.rows.items()}
        req = pd.DataFrame([{"Параметр": p, **{f"δ ≤ {d} %": n for d, n in row.items()}} for p, row in self.req.items()])
        try:
            if path.lower().endswith(".csv"):
                allp = pd.concat([f.assign(Параметр=p) for p, f in frames.items()])
                with open(path, "w", encoding="utf-8-sig", newline="") as fh:
                    fh.write("# " + self.header + "\n")
                    allp.to_csv(fh, index=False)
            else:
                with pd.ExcelWriter(path) as w:
                    pd.DataFrame({"Експеримент": [self.header]}).to_excel(w, sheet_name="Опис", index=False)
                    for p, f in frames.items():
                        f.to_excel(w, sheet_name=f"T-тест {p}"[:31], index=False)
                    dec_rows = []
                    for (p, N), tv in self.tvals.items():
                        tc = next(r["tкр"] for r in self.rows[p] if r["N"] == N)
                        for i, v in enumerate(tv):
                            dec_rows.append({"параметр": p, "N": N, "№ вибірки": i + 1, "t": float(v), "tкр": tc,
                                             "висновок": "відхиляємо H₀" if abs(v) > tc else "не відхиляємо H₀"})
                    dec = pd.DataFrame(dec_rows)
                    dec.to_excel(w, sheet_name="Рішення по вибірках", index=False)
        except Exception as ex:
            messagebox.showerror("Помилка експорту", str(ex))
