import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
fs_sim = 1000  # Convolution sampling frequency
dt = 1 / fs_sim  # Seconds per sample
T = 10  # Total simulation time in seconds (you can adjust this)
N = int(T * fs_sim)  # Time measured in points
t = np.linspace(0, T, N)

# Parameters
params1 = [3.25, 100, 22.0, 50.0, 20.0, 500.0, 135.0, 220.0, 0.56, 6.0, 2.5, 1, 1000, 10**(-0.5), 100] + [10 * np.random.rand() for _ in range(11)]
params2 = [3.25, 100, 22.0, 50.0, 20.0, 500.0, 135.0, 220.0, 0.56, 6.0, 2.5, 1, 1000, 10**(-0.5), 100] + [10 * np.random.rand() for _ in range(11)]

X = np.array([params1[15:],params2[15:]]).reshape(-1, 1)  # Initial state
OU1 = params1[14]
OU2 = params2[14]

# Begin Model Simulation
dW1 = np.sqrt(dt) * np.random.randn(1, N)
dW2 = np.sqrt(dt) * np.random.randn(1, N)

# Extract parameters
params = {
    'A': [params1[0], params2[0]],
    'a': [params1[1], params2[1]],
    'B': [params1[2], params2[2]],
    'b': [params1[3], params2[3]],
    'G': [params1[4], params2[4]],
    'g': [params1[5], params2[5]],
    'Cvec': [params1[6], params2[6]],
    'P': [params1[7], params2[7]],
    'r': [params1[8], params2[8]],
    'v0': [params1[9], params2[9]],
    'e0': [params1[10], params2[10]],
    'ss': [params1[11], params2[11]],
    'K': [params1[12], params2[12]],
    'tau': [params1[13], params2[13]],
    'D': [params1[14], params2[14]]
}

C1 = [params['Cvec'][0], params['Cvec'][1]]
C2 = [0.8 * params['Cvec'][0], 0.8 * params['Cvec'][1]]
C3 = [0.25 * params['Cvec'][0], 0.25 * params['Cvec'][1]]
C4 = [0.25 * params['Cvec'][0], 0.25 * params['Cvec'][1]]
C5 = [0.3 * params['Cvec'][0], 0.3 * params['Cvec'][1]]
C6 = [0.1 * params['Cvec'][0], 0.1 * params['Cvec'][1]]
C7 = [0.8 * params['Cvec'][0], 0.8 * params['Cvec'][1]]

a2 = [params['a'][0] ** 2, params['a'][1] ** 2]
Aa = [params['a'][0] * params['A'][0], params['a'][1] * params['A'][1]]
Bb = [params['b'][0] * params['B'][0], params['b'][1] * params['B'][1]]
b2 = [params['b'][0] ** 2, params['b'][1] ** 2]
g2 = [params['g'][0] ** 2, params['g'][1] ** 2]
Gg = [params['G'][0] * params['g'][0], params['G'][1] * params['g'][1]]

Xem1 = np.full((1, len(t)), np.nan)
Xem2 = np.full((1, len(t)), np.nan)

for j in range(len(t)):
    
    Xem1[0,j] = X[2] - X[3] - X[4] # Membrane potential output
    Xem2[0,j] = X[12] - X[13] - X[14] # Membrane potential output
    
    OU1 = OU1 - (OU1 / params['tau'][0]) * dt + (np.sqrt(2 * params['D'][0]) / params['tau'][0]) * dW1[0, j]
    OU2 = OU2 - (OU2 / params['tau'][1]) * dt + (np.sqrt(2 * params['D'][1]) / params['tau'][1]) * dW2[0, j]
    
    # Column 1

    X[0] = X[0] + X[5] * dt
    X[1] = X[1] + X[6] * dt
    X[2] = X[2] + X[7] * dt
    X[3] = X[3] + X[8] * dt
    X[4] = X[4] + X[9] * dt

    X[5] = X[5] + (Aa[0] * (2 * params['e0'][0] / (1 + np.exp(params['r'][0] * (params['v0'][0] - (X[1] - X[2] - X[3]))))) - 2 * params['a'][0] * X[5] - X[0] * a2[0]) * dt
    X[6] = X[6] + (Aa[0] * (params['P'][0] + C2[0] * 2 * params['e0'][0] / (1 + np.exp(params['r'][0] * (params['v0'][0] - (C1[0] * X[0])))) + params['K'][1]*Xem2[0,j]) - 2 * params['a'][0] * X[6] - X[1] * a2[0]) * dt + Aa[0] * params['ss'][0] * OU1
    ev = 2 * params['e0'][0] / (1 + np.exp(params['r'][0] * (params['v0'][0] - (C3[0] * X[0]))))
    X[7] = X[7] + (Bb[0] * (C4[0] * ev) - 2 * params['b'][0] * X[7] - X[2] * b2[0]) * dt
    X[8] = X[8] + (Gg[0] * C7[0] * (2 * params['e0'][0] / (1 + np.exp(params['r'][0] * (params['v0'][0] - (C5[0] * X[0] - X[4]))))) - 2 * params['g'][0] * X[8] - X[3] * g2[0]) * dt
    X[9] = X[9] + (Bb[0] * (C6[0] * ev) - 2 * params['b'][0] * X[9] - X[4] * b2[0]) * dt
    
    # Column 2
    
    X[10] = X[10] + X[15] * dt
    X[11] = X[11] + X[16] * dt
    X[12] = X[12] + X[17] * dt
    X[13] = X[13] + X[18] * dt
    X[14] = X[14] + X[19] * dt

    X[15] = X[15] + (Aa[1] * (2 * params['e0'][1] / (1 + np.exp(params['r'][1] * (params['v0'][1] - (X[11] - X[12] - X[13]))))) - 2 * params['a'][1] * X[15] - X[10] * a2[1]) * dt
    X[16] = X[16] + (Aa[1] * (params['P'][1] + C2[1] * 2 * params['e0'][1] / (1 + np.exp(params['r'][1] * (params['v0'][1] - (C1[1] * X[10])))) + params['K'][0]*Xem1[0,j]) - 2 * params['a'][1] * X[16] - X[11] * a2[1]) * dt + Aa[1] * params['ss'][1] * OU2
    ev = 2 * params['e0'][1] / (1 + np.exp(params['r'][1] * (params['v0'][1] - (C3[1] * X[10]))))
    X[17] = X[17] + (Bb[1] * (C4[1] * ev) - 2 * params['b'][1] * X[17] - X[12] * b2[1]) * dt
    X[18] = X[18] + (Gg[1] * C7[1] * (2 * params['e0'][1] / (1 + np.exp(params['r'][1] * (params['v0'][1] - (C5[1] * X[10] - X[14]))))) - 2 * params['g'][1] * X[18] - X[13] * g2[1]) * dt
    X[19] = X[19] + (Bb[1] * (C6[1] * ev) - 2 * params['b'][1] * X[19] - X[14] * b2[1]) * dt

    Xem1[0,j] = X[1] - X[2] - X[3]
    Xem2[0,j] = X[12] - X[13] - X[14]

    Xemburn = Xem1[:, int(5 * fs_sim):N - 1]

# Plot the result
plt.plot(Xemburn.T)
plt.xlabel('Time (samples)')
plt.ylabel('Xemburn')
plt.title('Simulation Result')
plt.show()
