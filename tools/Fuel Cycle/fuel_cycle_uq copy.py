import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# --- Parameters ---
Ntin = 0.00009 * 3600
t0 = 0.0
tN = 2400.0
h = 0.1
steps = int(round((tN - t0) / h)) + 1
samples = 1000

# Randomization flags (0 or 1)
a = 0  # Ntburn
b = 1  # TBR
c = 0  # Istartup
d = 0  # tIFC
e = 0  # tOFC

# --- Time array ---
time = np.linspace(t0, tN, steps)

# --- Output arrays (steps x samples) ---
IOFC      = np.zeros((steps, samples))
IIFC      = np.zeros((steps, samples))
Istorage  = np.zeros((steps, samples))
NIOFC     = np.zeros((steps, samples))
NIIFC     = np.zeros((steps, samples))
NIstorage = np.zeros((steps, samples))

# --- Simulation ---
for i in range(samples):
    # Sample parameters (constant over the simulation for each sample)
    ntburn   = 0.01 * Ntin + a * 0.001 * Ntin * np.random.randn()
    tbr      = 1.08 + b * 0.03  * np.random.randn()
    istartup = 1.5  + c * 0.15  * np.random.randn()
    tifc     = 4.0  + d * 0.4   * np.random.randn()
    tofc     = 24.0 + e * 2.4   * np.random.randn()
    TBE      = ntburn / Ntin

    # Exact (analytical) solutions — vectorized over all time steps
    IOFC[:, i] = ntburn * tofc * tbr * (1 - np.exp(-time / tofc))

    IIFC[:, i] = (
        ntburn * tifc * ((1 - TBE) / TBE) * (1 - np.exp(-time / tifc))
        + ntburn * tbr * tifc * (1 - np.exp(-time / tifc))
        + ntburn * tbr * (tifc * tofc / (tofc - tifc))
        * (np.exp(-time / tifc) - np.exp(-time / tofc))
    )

    Istorage[:, i] = (
        istartup
        + ntburn * (tbr - 1) * time
        + ntburn * tbr * (tifc**2 / (tofc - tifc)) * (1 - np.exp(-time / tifc))
        + ntburn * tbr * (tofc**2 / (tofc - tifc)) * (1 - np.exp(-time / tofc))
        - ntburn * tifc * ((1 - TBE) / TBE) * (1 - np.exp(-time / tifc))
    )

    # Euler (numerical) solutions — vectorized via closed-form recurrence solutions
    # NIOFC: scalar linear recurrence NIOFC[j+1] = alpha*NIOFC[j] + h*ntburn*tbr
    alpha = 1.0 - h / tofc
    NIOFC[:, i] = ntburn * tbr * tofc * (1.0 - alpha ** np.arange(steps))

    # NIIFC: driven linear recurrence NIIFC[j+1] = gamma*NIIFC[j] + forcing[j]
    # solved with lfilter (causal convolution with geometric kernel)
    gamma   = 1.0 - h / tifc
    forcing = h * ((1 - TBE) / TBE * ntburn + NIOFC[:, i] / tofc)
    NIIFC[1:, i] = lfilter([1.0], [1.0, -gamma], forcing)[:-1]

    # NIstorage: cumulative sum of increments
    delta = h * (NIIFC[:, i] / tifc - ntburn / TBE)
    NIstorage[0, i] = istartup
    NIstorage[1:, i] = istartup + np.cumsum(delta[:-1])


# --- Helper: Gaussian PDF ---
def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# --- Plotting ---
if a == 1 and b == 0 and c == 0 and d == 0 and e == 0:
    final = Istorage[-1, :]
    fig, ax = plt.subplots(num=7)
    ax.hist(final, density=True)
    x4 = np.linspace(final.min(), final.max(), 1001)
    ax.plot(x4, normal_pdf(x4, final.mean(), final.std()))

elif a == 0 and b == 1 and c == 0 and d == 0 and e == 0:
    final = Istorage[-1, :]

    fig1, ax1 = plt.subplots(num=1)
    ax1.hist(final, density=True)
    x4 = np.linspace(final.min(), final.max(), 1001)
    ax1.plot(x4, normal_pdf(x4, final.mean(), final.std()))
    ax1.set_xlabel('Tritium in storage (kg)')
    ax1.set_ylabel('Probability density')

    fig2, ax2 = plt.subplots(num=2)
    for i in range(samples):
        ax2.plot(time, Istorage[:, i],  label='Exact' if i == 0 else '_')
        ax2.plot(time, NIstorage[:, i], '--', label='Euler' if i == 0 else '_')
    ax2.set_xlabel('time (h)')
    ax2.set_ylabel('Tritium in storage (kg)')
    ax2.legend()

elif a == 0 and b == 0 and c == 1 and d == 0 and e == 0:
    data = Istorage.ravel()
    fig, ax = plt.subplots(num=7)
    ax.hist(data, density=True)
    x4 = np.linspace(data.min(), data.max(), 1001)
    ax.plot(x4, normal_pdf(x4, data.mean(), data.std()))
    ax.set_xlabel('Tritium in storage (kg) after one cycle with only initial tritium randomised')
    ax.set_ylabel('Probability density')

elif a == 0 and b == 0 and c == 0 and d == 1 and e == 0:
    data = Istorage.ravel()
    fig, ax = plt.subplots(num=7)
    ax.hist(data, density=True)
    x4 = np.linspace(data.min(), data.max(), 1001)
    ax.plot(x4, normal_pdf(x4, data.mean(), data.std()))

elif a == 0 and b == 0 and c == 0 and d == 0 and e == 1:
    data = Istorage.ravel()
    fig, ax = plt.subplots(num=7)
    ax.hist(data, density=True)
    x4 = np.linspace(data.min(), data.max(), 1001)
    ax.plot(x4, normal_pdf(x4, data.mean(), data.std()))
    samplemean = np.sum(data) / samples
    samplesd   = np.sqrt(np.sum(data ** 2) / samples - samplemean ** 2)
    ax.plot(x4, normal_pdf(x4, samplemean, samplesd))

elif a == 1 and b == 1 and c == 1 and d == 1 and e == 1:
    data = Istorage.ravel()
    fig, ax = plt.subplots(num=7)
    ax.hist(data, density=True)
    x4 = np.linspace(data.min(), data.max(), 1001)
    ax.plot(x4, normal_pdf(x4, data.mean(), data.std()))
    ax.set_xlabel('Tritium in storage (kg) after one cycle with all parameters randomised')
    ax.set_ylabel('Probability density')

plt.show()