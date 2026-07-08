import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import hilbert
from networkx.algorithms import cycle_basis
from numpy.fft import fft, fftfreq

# -------------------------------
# Network parameters
# -------------------------------
N = 5000
T = 5.0
dt = 0.001
steps = int(T / dt)

# Weight matrix
W = np.random.randn(N, N) * 0.1
np.fill_diagonal(W, 0)

# Initial state
x = np.random.randn(N)

# Input signal (sinusoidal)
freq = 8.0
amp = 1.0
t = np.linspace(0, T, steps)
input_signal = amp * np.sin(2 * np.pi * freq * t)

# Storage for neural activity
activity = np.zeros((steps, N))

# -------------------------------
# Simulation loop
# -------------------------------
for i in range(steps):
    x = np.tanh(W @ x + input_signal[i])
    activity[i] = x

# ============================================================
# Figure S1 — Neural Activity (first 5 neurons)
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(t, activity[:, :5])
plt.title("Neural Activity (first 5 neurons)")
plt.xlabel("Time (s)")
plt.ylabel("Activity")
plt.grid(True)
plt.figtext(0.5, -0.10,
            "Figure S1 — Time‑series activity of the first five neurons under periodic stimulation.",
            ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# Figure S2 — Input Signal
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(t, input_signal)
plt.title("Input Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.figtext(0.5, -0.10,
            "Figure S2 — External periodic input applied uniformly to all neurons.",
            ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# Figure S3 — Network Connectivity
# ============================================================
G = nx.from_numpy_array(W)
plt.figure(figsize=(8, 8))
nx.draw(G, node_size=30, width=0.2)
plt.title("Network Connectivity")
plt.figtext(0.5, -0.05,
            "Figure S3 — Circular network visualization of the recurrent connectivity matrix.",
            ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# Figure S4 — Hilbert Phase (Neuron 1)
# ============================================================
analytic = hilbert(activity[:, 0])
phase = np.unwrap(np.angle(analytic))
plt.figure(figsize=(10, 4))
plt.plot(t, phase)
plt.title("Hilbert Phase (Neuron 1)")
plt.xlabel("Time (s)")
plt.ylabel("Phase")
plt.grid(True)
plt.figtext(0.5, -0.10,
            "Figure S4 — Instantaneous phase of Neuron 1 computed using the Hilbert transform.",
            ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# Figure S5 — Mean Activity (all neurons)
# ============================================================
mean_activity = np.mean(activity, axis=1)
plt.figure(figsize=(10, 4))
plt.plot(t, mean_activity)
plt.title("Mean Activity (all neurons)")
plt.xlabel("Time (s)")
plt.ylabel("Mean Activity")
plt.grid(True)
plt.figtext(0.5, -0.10,
            "Figure S5 — Mean activity across all neurons over time.",
            ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# Figure S6 — FFT Spectrum (Neuron 1)
# ============================================================
yf = fft(activity[:, 0])
xf = fftfreq(steps, dt)[:steps//2]
plt.figure(figsize=(10, 4))
plt.plot(xf, 2.0/steps * np.abs(yf[0:steps//2]))
plt.title("FFT Spectrum (Neuron 1)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.figtext(0.5, -0.10,
            "Figure S6 — Amplitude spectrum of Neuron 1 obtained via Fast Fourier Transform.",
            ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# Figure S7 — Correlation Matrix of Neurons
# ============================================================
corr_matrix = np.corrcoef(activity.T)
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label="Correlation")
plt.title("Correlation Matrix of Neurons")
plt.figtext(0.5, -0.08,
            "Figure S7 — Correlation matrix showing pairwise correlations across all neurons.",
            ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# Figure S8 — Closed Cycles in Network Graph
# ============================================================
cycles = cycle_basis(G)
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=30, width=0.2)
for cycle in cycles[:5]:
    nx.draw_networkx_nodes(G, pos, nodelist=cycle, node_color='r')
plt.title("Closed Cycles in Network Graph")
plt.figtext(0.5, -0.05,
            "Figure S8 — Visualization of closed loops (cycle basis) in the connectivity graph.",
            ha='center', fontsize=12)
plt.tight_layout()
plt.show()
