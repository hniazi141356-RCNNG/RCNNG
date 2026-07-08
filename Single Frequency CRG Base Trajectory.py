import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline

def single_frequency_crg(n_points=2000, freq=1.0, radius=1.0, phase=0.0):
    t = np.linspace(0, 2*np.pi, n_points)
    x1 = radius * np.cos(freq*t + phase)
    x2 = radius * np.sin(freq*t + phase)
    x3 = 0.3 * np.sin(freq*t + phase)
    return x1, x2, x3

def plot_crg(x1, x2, x3, title="CRG"):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x1, x2, x3, color='blue')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title(title)
    plt.show()

# ---- اجرای مستقیم ----
x1, x2, x3 = single_frequency_crg()
plot_crg(x1, x2, x3, "Single-frequency CRG")

