!pip install ripser persim

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# ---------- تنظیمات ژورنالی ----------
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "figure.dpi": 300,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "lines.linewidth": 1.8
})

# ---------- 1. مدل ----------
W = np.array([[0.7, -0.2, 0.1],
              [0.3,  0.6, -0.4],
              [-0.2, 0.5, 0.8]])

def sigma(x):
    return np.tanh(x)

# ورودی چندفرکانسی
A1, omega1 = 1.0, 0.5
A2, omega2 = 0.7, 1.2

def I_multi(t):
    s = A1*np.sin(omega1*t) + A2*np.sin(omega2*t)
    return np.array([s, s, s])

def ode_multi(t, x):
    return -x + W @ sigma(x) + I_multi(t)

# ---------- 2. شبیه‌سازی ----------
t_eval = np.linspace(0, 300, 8000)
x0 = np.array([0.2, -0.1, 0.3])

sol = solve_ivp(ode_multi, (0, 300), x0, t_eval=t_eval)
X = sol.y.T   # point cloud

# فقط بخش پایدار
stable_idx = int(0.7 * len(X))
X_stable = X[stable_idx:]

# ---------- 3. محاسبهٔ Persistent Homology ----------
diagrams = ripser(X_stable, maxdim=1)['dgms']

# ---------- 4. رسم شکل ----------
fig = plt.figure(figsize=(15, 4.5))

# Panel A — point cloud
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(X_stable[:,0], X_stable[:,1], X_stable[:,2], '.', markersize=1, color='navy')
ax1.set_title("A) Stable Geometry (Point Cloud)")
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$x_3$")

# Panel B — persistence diagram
ax2 = fig.add_subplot(132)
plot_diagrams(diagrams, ax=ax2)
ax2.set_title("B) Persistence Diagram")

# Panel C — barcode
ax3 = fig.add_subplot(133)
for dim, diag in enumerate(diagrams):
    for birth, death in diag:
        ax3.plot([birth, death], [dim, dim], color='crimson')
ax3.set_title("C) Persistence Barcode")
ax3.set_xlabel("Scale")
ax3.set_yticks([0,1])
ax3.set_yticklabels(["H0","H1"])

plt.tight_layout()
plt.show()
