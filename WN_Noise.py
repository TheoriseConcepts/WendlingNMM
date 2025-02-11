import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
fs_sim = 1000  # Convolution sampling frequency
dt = 1 / fs_sim  # Seconds per sample
T = 10  # Total simulation time in seconds (you can adjust this)
N = int(T * fs_sim)  # Time measured in points
t = np.linspace(0, T, N)

# Parameters
params1 = [3.25, 100, 22.0, 50.0, 20.0, 500.0, 135.0, 220.0, 0.56, 6.0, 2.5, 1] + [10 * np.random.rand() for _ in range(11)]

X = np.array(params1[13:]).reshape(-1, 1)  # Initial state

# Begin Model Simulation
dW = np.sqrt(dt) * np.random.randn(1, N)

# Extract parameters
A, a, B, b, G, g, Cvec, P, r, v0, e0, ss = params1[:12]

params = {
    'A': A, 'a': a, 'B': B, 'b': b, 'G': G, 'g': g, 'Cvec': Cvec,
    'P': P, 'r': r, 'v0': v0, 'e0': e0, 'ss': ss
}

C1 = params['Cvec']
C2 = 0.8 * params['Cvec']
C3 = 0.25 * params['Cvec']
C4 = 0.25 * params['Cvec']
C5 = 0.3 * params['Cvec']
C6 = 0.1 * params['Cvec']
C7 = 0.8 * params['Cvec']

a2 = params['a'] ** 2
Aa = params['a'] * params['A']
Bb = params['b'] * params['B']
b2 = params['b'] ** 2
g2 = params['g'] ** 2
Gg = params['G'] * params['g']

Xem1 = np.full((1, len(t)), np.nan)

for j in range(len(t)):

    X[0] = X[0] + X[5] * dt
    X[1] = X[1] + X[6] * dt
    X[2] = X[2] + X[7] * dt
    X[3] = X[3] + X[8] * dt
    X[4] = X[4] + X[9] * dt

    X[5] = X[5] + (Aa * (2 * params['e0'] / (1 + np.exp(params['r'] * (params['v0'] - (X[1] - X[2] - X[3]))))) - 2 * params['a'] * X[5] - X[0] * a2) * dt
    X[6] = X[6] + (Aa * (params['P'] + C2 * 2 * params['e0'] / (1 + np.exp(params['r'] * (params['v0'] - (C1 * X[0]))))) - 2 * params['a'] * X[6] - X[1] * a2) * dt + Aa * params['ss'] * dW[:,j]
    ev = 2 * params['e0'] / (1 + np.exp(params['r'] * (params['v0'] - (C3 * X[0]))))
    X[7] = X[7] + (Bb * (C4 * ev) - 2 * params['b'] * X[7] - X[2] * b2) * dt
    X[8] = X[8] + (Gg * C7 * (2 * params['e0'] / (1 + np.exp(params['r'] * (params['v0'] - (C5 * X[0] - X[4]))))) - 2 * params['g'] * X[8] - X[3] * g2) * dt
    X[9] = X[9] + (Bb * (C6 * ev) - 2 * params['b'] * X[9] - X[4] * b2) * dt

    Xem1[0, j] = X[1] - X[2] - X[3]
    # Remove transient
    Xemburn = Xem1[:, int(5 * fs_sim):N - 1]

# Plot the result
plt.plot(Xemburn.T)
plt.xlabel('Time (samples)')
plt.ylabel('Xemburn')
plt.title('Simulation Result')
plt.show()
