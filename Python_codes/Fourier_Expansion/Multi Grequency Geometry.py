A1, omega1 = 1.0, 0.5
A2, omega2 = 0.7, 1.2

def I_multi(t):
    s = A1*np.sin(omega1*t) + A2*np.sin(omega2*t)
    return np.array([s, s, s])

def ode_multi(t, x): return -x + W @ sigma(x) + I_multi(t)

solutions = [solve_ivp(ode_multi, (0, 300), x0, t_eval=t_eval) 
             for x0 in initial_conditions]

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

for sol, c in zip(solutions, colors):
    ax.plot(sol.y[0], sol.y[1], sol.y[2], color=c, alpha=0.85)

ax.set_title("Figure 2 — Multi-frequency Geometry")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.tight_layout()
plt.show()
