import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline

plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "figure.dpi": 300,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "lines.linewidth": 2.0
})

# -----------------------------
#  Network Model
# -----------------------------
W = np.array([[0.7, -0.2, 0.1],
              [0.3,  0.6, -0.4],
              [-0.2, 0.5, 0.8]])

def sigma(x):
    return np.tanh(x)

# -----------------------------
#  Multi-frequency Input
# -----------------------------
A1, w1 = 1.0, 0.5     # فرکانس اول
A2, w2 = 0.7, 1.2     # فرکانس دوم
A3, w3 = 0.5, 2.0     # فرکانس سوم

def I_multi(t):
    return np.array([
        A1*np.sin(w1*t) + A2*np.sin(w2*t) + A3*np.sin(w3*t),
        A1*np.sin(w1*t + 0.3) + A2*np.sin(w2*t + 0.5) + A3*np.sin(w3*t + 1.0),
        A1*np.sin(w1*t + 0.7) + A2*np.sin(w2*t + 1.1) + A3*np.sin(w3*t + 1.5)
    ])

def ode_multi(t, x):
    return -x + W @ sigma(x) + I_multi(t)

# -----------------------------
#  Simulation
# -----------------------------
t_eval = np.linspace(0, 300, 6000)

initial_conditions = [
    [0.1, -0.2, 0.3],
    [1.0, -0.5, 0.2],
    [-0.7, 0.4, -0.3],
    [0.5, 0.5, 0.5]
]

solutions = [
    solve_ivp(ode_multi, (0, 300), x0, t_eval=t_eval)
    for x0 in initial_conditions
]

# -----------------------------
#  Plot
# -----------------------------
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['navy', 'crimson', 'forestgreen', 'purple']

for sol, c in zip(solutions, colors):
    ax.plot(sol.y[0], sol.y[1], sol.y[2], color=c, alpha=0.85)

ax.set_title("Figure 2 — Multi-frequency CRG")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")

plt.tight_layout()
plt.show()
