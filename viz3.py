"""viz3.py - графіки ЛР3"""
import numpy as np
import twodim as td
def plot_scatter(ax, x, y, mx, my, sx, sy, r):
    """Кореляційне поле + лінії рівня відтвореної щільності (еліпси розсіювання)"""
    ax.clear()
    ax.scatter(x, y, s=10, alpha=.6, label="точки вибірки (xᵢ, yᵢ)")
    gx = np.linspace(mx - 3.5 * sx, mx + 3.5 * sx, 120)
    gy = np.linspace(my - 3.5 * sy, my + 3.5 * sy, 120)
    GX, GY = np.meshgrid(gx, gy)
    Z = td.density(x, y, mx, my, sx, sy, r)(GX, GY)
    ax.contour(GX, GY, Z, levels=5, colors="red", linewidths=1)
    ax.plot([], [], color="red", label="лінії рівня f(x,y)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Кореляційне поле, r̂ = {td.estimates(x, y)['rxy']:.3f}")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

def plot_heat(ax, n, p, ex, ey):
    """Вид зверху: колір і число в клітинці — відносна частота p_ij = n_ij / N"""
    ax.clear()
    ax.imshow(p, origin="lower", aspect="auto", cmap="viridis",
              extent=[ex[0], ex[-1], ey[0], ey[-1]])
    for j in range(n.shape[0]):
        for i in range(n.shape[1]):
            ax.text((ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2, f"{p[j, i]:.3f}",
                    ha="center", va="center", color="white", fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Двовимірна гістограма: відносні частоти p_ij = n_ij / N")

def plot_bar3d(fig, n, p, ex, ey):
    """3D-стовпчики висотою відносної частоти p_ij = n_ij / N"""
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    hx, hy = ex[1] - ex[0], ey[1] - ey[0]
    xs, ys = np.meshgrid(ex[:-1], ey[:-1])
    dz = p.ravel()
    ax.bar3d(xs.ravel(), ys.ravel(), np.zeros_like(dz), hx * 0.9, hy * 0.9, dz, shade=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("p_ij")
    ax.set_title("Двовимірна гістограма (3D): відносні частоти")
def plot_chi_compare(fig, n, Np, cx, cy):
    """Емпіричні та очікувані частоти поруч (для наочності критерію χ²)"""
    fig.clear()
    a1, a2 = fig.subplots(1, 2)
    a1.imshow(n, origin="lower", aspect="auto", cmap="Blues")
    a1.set_title("емпіричні n_ij")
    a2.imshow(Np, origin="lower", aspect="auto", cmap="Oranges")
    a2.set_title("очікувані N·p_ij")
    for a in (a1, a2):
        a.set_xlabel("клас x"); a.set_ylabel("клас y")