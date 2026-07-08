import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "figure.dpi": 300,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "lines.linewidth": 1.8
})

# Panel A — frequency inputs
t = np.linspace(0, 2*np.pi, 1000)
s1 = np.sin(1*t)
s2 = 0.8*np.sin(2*t)
s3 = 0.6*np.sin(3*t)

# Panel B — base CRGs
theta = np.linspace(0, 2*np.pi, 400)
x1, y1 = np.cos(theta), np.sin(theta)
x2, y2 = 0.8*np.cos(theta), 0.8*np.sin(theta)
x3, y3 = 0.6*np.cos(theta), 0.6*np.sin(theta)

# Panel C — linked 3D geometry
phi = np.linspace(0, 2*np.pi, 600)
xL1, yL1, zL1 = np.cos(phi), np.sin(phi), 0.3*np.sin(2*phi)
xL2, yL2, zL2 = 0.8*np.cos(phi), 0.8*np.sin(phi), 0.3*np.cos(2*phi)
xL3, yL3, zL3 = 0.6*np.cos(phi), 0.6*np.sin(phi), 0.3*np.sin(3*phi)

fig = plt.figure(figsize=(15, 4.5))

# Panel A
ax1 = fig.add_subplot(131)
ax1.plot(t, s1, color='navy')
ax1.plot(t, s2, color='crimson')
ax1.plot(t, s3, color='forestgreen')
ax1.set_title("A) Frequency Inputs")
ax1.grid(True, alpha=0.3)

# Panel B
ax2 = fig.add_subplot(132)
ax2.plot(x1, y1, color='navy')
ax2.plot(x2, y2, color='crimson')
ax2.plot(x3, y3, color='forestgreen')
ax2.set_aspect('equal')
ax2.set_title("B) Base CRGs")
ax2.grid(True, alpha=0.3)

# Panel C
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(xL1, yL1, zL1, color='navy')
ax3.plot(xL2, yL2, zL2, color='crimson')
ax3.plot(xL3, yL3, zL3, color='forestgreen')
ax3.set_title("C) Fourier–Geometric Expansion")

plt.tight_layout()
plt.show()
