"""
Cancellation Anomaly Score (CAS) demo on SYNTHETIC data.

No real data is used. The service-request counts below are randomly generated
to show how the score, heatmap, and power-law fit work.

CAS for owner i and request type j:
    S_ij = [c_ij / (c_ij + x_ij)] * [c_ij / sum_j c_ij] * [c_ij / sum_i c_ij] * 100
where c_ij = cancelled requests and x_ij = completed requests.
Reading it left to right: how often this owner cancels this type, how much of the
owner's cancellations are this type, and how much of this type's cancellations
are this owner's.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

rng = np.random.default_rng(42)
N_OWNERS, N_TYPES = 60, 8
owners = [f"Owner {i+1:02d}" for i in range(N_OWNERS)]
types = [f"Type {chr(65+j)}" for j in range(N_TYPES)]

# --- 1. Synthetic requests: volume per cell, baseline cancel rate ~8% ---
volume = rng.poisson(lam=rng.uniform(10, 120, size=(N_OWNERS, N_TYPES)))
p_cancel = np.full((N_OWNERS, N_TYPES), 0.08)
# plant 4 outlier cells (higher cancel rate) so the demo has something to find
planted = [(5, 2), (17, 2), (31, 6), (44, 2)]
for i, j in planted:
    p_cancel[i, j] = 0.45
    volume[i, j] = max(volume[i, j], 90)
c = rng.binomial(volume, p_cancel)          # cancelled
x = volume - c                              # completed
C = pd.DataFrame(c, index=owners, columns=types)
X = pd.DataFrame(x, index=owners, columns=types)

# --- 2. Cancellation Anomaly Score ---
rate = C / (C + X).replace(0, np.nan)
owner_share = C.div(C.sum(axis=1), axis=0)
type_share = C.div(C.sum(axis=0), axis=1)
S = (rate * owner_share * type_share * 100).fillna(0)

# --- 3. Rank cells and fit a power law  y = C0 / rank^a  (log-log least squares) ---
scores = S.stack().sort_values(ascending=False)
scores = scores[scores > 0]
rank = np.arange(1, len(scores) + 1)
slope, intercept = np.polyfit(np.log(rank), np.log(scores.values), 1)
alpha = -slope
fit = np.exp(intercept) * rank ** slope
# Flag cells whose log-score is a robust outlier (median/MAD z-score > 2.5)
lg = np.log(scores.values)
mad = np.median(np.abs(lg - np.median(lg))) * 1.4826
flag = (lg - np.median(lg)) / mad > 2.5
outliers = scores[flag]

print(f"Cells scored: {len(scores)}")
print(f"Power-law exponent a = {alpha:.2f}")
print("\nFlagged outlier cells (robust z-score of log CAS > 2.5):")
print(outliers.round(3).to_string())
print("\nPlanted outliers:", [(owners[i], types[j]) for i, j in planted])

# --- 4. Plots ---
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(S.values + 1e-6, aspect="auto", cmap="viridis",
               norm=LogNorm(vmin=max(S.values[S.values > 0].min(), 1e-4), vmax=S.values.max()))
ax.set_xticks(range(N_TYPES)); ax.set_xticklabels(types, rotation=45, ha="right")
ax.set_yticks(range(0, N_OWNERS, 5)); ax.set_yticklabels(owners[::5])
ax.set_title("Synthetic data: CAS heatmap (log color scale)")
fig.colorbar(im, ax=ax, label="CAS")
plt.tight_layout(); plt.savefig("cas_heatmap.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(rank, scores.values, "o", ms=3, alpha=0.6, label="Cell score")
ax.loglog(rank, fit, "r-", label=f"Power-law fit (a = {alpha:.2f})")
ax.loglog(rank[flag], scores.values[flag], "ko", ms=6, mfc="none", label="Flagged outliers")
ax.set_xlabel("Rank"); ax.set_ylabel("CAS"); ax.legend()
ax.set_title("Synthetic data: ranked CAS with power-law fit")
plt.tight_layout(); plt.savefig("cas_powerlaw.png", dpi=150); plt.close()
