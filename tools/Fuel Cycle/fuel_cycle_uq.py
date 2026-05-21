import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Tritium fuel cycle simulation with Monte Carlo uncertainty quantification'
)
parser.add_argument('--mean_tbr',       type=str,                    help='Path to file containing mean tritium breeding ratio')
parser.add_argument('--std_tbr',        type=str,                    help='Path to file containing standard deviation of tritium breeding ratio')
parser.add_argument('--fuelling_rate',  type=float, default=0.00009, help='Fuelling rate (kg/s)')
parser.add_argument('--t0',             type=float, default=0.0,     help='Start time (h)')
parser.add_argument('--tN',             type=float, default=2400.0,  help='End time (h)')
parser.add_argument('--h',              type=float, default=0.1,     help='Time step size (h)')
parser.add_argument('--samples',        type=int,   default=1000,    help='Number of Monte Carlo samples')
parser.add_argument('--output_histogram',  type=str, default='histogram.png',  help='Output histogram PNG path')
parser.add_argument('--output_timeseries', type=str, default='timeseries.png', help='Output time-series PNG path')
parser.add_argument('--output_data',       type=str, default='output.tabular', help='Output tabular data path')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
with open(args.mean_tbr) as _f:
    mean_tbr = float(_f.read().strip())
with open(args.std_tbr) as _f:
    tbr_sigma = float(_f.read().strip())

Ntin    = args.fuelling_rate * 3600   # convert kg/s → kg/h
t0      = args.t0
tN      = args.tN
h       = args.h
samples = args.samples

steps = int(round((tN - t0) / h)) + 1

istartup     = 1.5    # initial tritium inventory (kg)
tifc         = 4.0    # in-facility cycle time (h)
tofc         = 24.0   # out-of-facility cycle time (h)

# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------
time      = np.linspace(t0, tN, steps)

IOFC      = np.zeros((steps, samples))
IIFC      = np.zeros((steps, samples))
Istorage  = np.zeros((steps, samples))
NIOFC     = np.zeros((steps, samples))
NIIFC     = np.zeros((steps, samples))
NIstorage = np.zeros((steps, samples))

# ---------------------------------------------------------------------------
# Simulation  (TBR randomised; all other parameters at their mean values)
# ---------------------------------------------------------------------------
ntburn = 0.01 * Ntin                        # burn rate (constant across samples)
TBE    = ntburn / Ntin                      # tritium burn efficiency

for i in range(samples):
    tbr = mean_tbr + tbr_sigma * np.random.randn()

    # --- Exact (analytical) solutions — vectorised over time ---
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

    # --- Euler (numerical) solutions — vectorised via linear recurrences ---
    alpha = 1.0 - h / tofc
    NIOFC[:, i] = ntburn * tbr * tofc * (1.0 - alpha ** np.arange(steps))

    gamma   = 1.0 - h / tifc
    forcing = h * ((1 - TBE) / TBE * ntburn + NIOFC[:, i] / tofc)
    NIIFC[1:, i] = lfilter([1.0], [1.0, -gamma], forcing)[:-1]

    delta = h * (NIIFC[:, i] / tifc - ntburn / TBE)
    NIstorage[0, i] = istartup
    NIstorage[1:, i] = istartup + np.cumsum(delta[:-1])

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# ---------------------------------------------------------------------------
# Plot 1 — Histogram of final storage across samples
# ---------------------------------------------------------------------------
final = Istorage[-1, :]
fig1, ax1 = plt.subplots()
ax1.hist(final, density=True, bins='auto', alpha=0.7, label='MC samples')
x4 = np.linspace(final.min(), final.max(), 1001)
ax1.plot(x4, normal_pdf(x4, final.mean(), final.std()), label='Normal fit')
ax1.set_xlabel('Tritium in storage at t=tN (kg)')
ax1.set_ylabel('Probability density')
ax1.set_title(f'Storage distribution — mean TBR={mean_tbr}, N={samples}')
ax1.legend()
fig1.tight_layout()
fig1.savefig(args.output_histogram, dpi=150, format='png')
plt.close(fig1)

# ---------------------------------------------------------------------------
# Plot 2 — Mean ± 1σ storage vs time
# ---------------------------------------------------------------------------
mean_stor  = Istorage.mean(axis=1)
std_stor   = Istorage.std(axis=1)
mean_nstor = NIstorage.mean(axis=1)

fig2, ax2 = plt.subplots()
ax2.plot(time, mean_stor,  label='Exact (mean)')
ax2.fill_between(time, mean_stor - std_stor, mean_stor + std_stor, alpha=0.3, label='±1σ')
ax2.plot(time, mean_nstor, '--', label='Euler (mean)')
ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Tritium in storage (kg)')
ax2.set_title(f'Storage vs time — mean TBR={mean_tbr}, N={samples}')
ax2.legend()
fig2.tight_layout()
fig2.savefig(args.output_timeseries, dpi=150, format='png')
plt.close(fig2)

# ---------------------------------------------------------------------------
# CSV output — time series summary statistics
# ---------------------------------------------------------------------------
header = 'time_h,exact_mean_kg,exact_std_kg,euler_mean_kg,euler_std_kg'
rows = np.column_stack([
    time,
    Istorage.mean(axis=1),  Istorage.std(axis=1),
    NIstorage.mean(axis=1), NIstorage.std(axis=1),
])
np.savetxt(args.output_data, rows, delimiter=',', header=header, comments='', fmt='%.6g')
