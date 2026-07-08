# Single-frequency
sol_single = solve_ivp(ode_single, (0, 300), [0.2, -0.1, 0.3], t_eval=t_eval)

# Multi-frequency
sol_multi = solve_ivp(ode_multi, (0, 300), [0.2, -0.1, 0.3], t_eval=t_eval)

# Stable part
stable_idx = int(0.7 * len(t_eval))
x_stable = sol_multi.y[:, stable_idx:]

fig = plt.figure(figsize=(15, 5))

# Panel A
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(sol_single.y[0], sol_single.y[1], sol_single.y[2], color='navy')
ax1.set_title("A) Single-frequency CRG")

# Panel B
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(sol_multi.y[0], sol_multi.y[1], sol_multi.y[2], color='crimson')
ax2.set_title("B) Multi-frequency Geometry")

# Panel C
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(x_stable[0], x_stable[1], x_stable[2], color='forestgreen')
ax3.set_title("C) Fourier–Geometric Expansion")

plt.tight_layout()
plt.show()
