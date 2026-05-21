import numpy as np

# Physical constants
k = 1.38e-23
mu_zero = 1.2566e-6
Av = 6.02e23

# Plasma pressure
m = 1
Mm = 2.0141
moles = m / Mm
atoms = moles * Av
volume = np.pi * (0.5**2) * 10
n_i = atoms / volume
n_e = atoms / volume
T = 1000
Pp = (n_i + n_e) * k * T

# Prior and likelihood parameters
prior_mean = 125000**2
prior_sd = 3000**2
single_likelihood_sd = 30
r_0 = 0.5
M = 0.25
omega = 2
Observations = 1000

# Bayesian update over observations
Pm = np.zeros(Observations)
S_1 = 0.0
S_2 = 0.0
for i in range(1, Observations + 1):
    Pm[i - 1] = Pp + single_likelihood_sd * np.random.randn()
    coil_term = (8 * np.pi**2 * (r_0 + M * np.sin(omega * i))**2) / mu_zero
    S_1 += 1 / (coil_term**2 * single_likelihood_sd**2)
    S_2 += Pm[i - 1] / (coil_term * single_likelihood_sd**2)

posterior_variance = 1 / (1 / prior_sd**2 + S_1)
posterior_mean = posterior_variance * (prior_mean / prior_sd**2 + S_2)
posterior_sd = np.sqrt(posterior_variance)

# Monte Carlo estimate of mean current
Montes = 1000
I_square = np.zeros(Montes)
S_3 = 0.0
for j in range(Montes):
    I_square[j] = posterior_mean + posterior_sd * np.random.randn()
    S_3 += np.sqrt(I_square[j])

mean_current = S_3 / Montes

print(f"Plasma pressure (Pp): {Pp:.4f} Pa")
print(f"Posterior mean: {posterior_mean:.4f}")
print(f"Posterior SD:   {posterior_sd:.4f}")
print(f"Mean current:   {mean_current:.4f} A")
