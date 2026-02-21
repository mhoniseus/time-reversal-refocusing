"""
Generate the convergence study figure for the split-step Strang scheme.

For a fixed random medium realization, propagate the Gaussian beam from z=0 to z=L
using M sub-steps per slice (h = z_c / M) for M in {1, 2, 4, 8, 16, 32, 64}.
Reference solution: M_ref = 256.
Plot relative L2 error vs h with O(h^2) reference line.

Output: report/figures/convergence_split_step.pdf
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from src.paraxial_solver import split_step_propagate
from src.random_medium import generate_medium_slices

# Parameters (same as in the report)
r0 = 2.0
N = 1024
x_max = 60.0
k = 1.0
L = 10.0
z_c = 1.0
x_c = 4.0
sigma = 1.0

x = np.linspace(-x_max / 2, x_max / 2, N, endpoint=False)
phi0 = np.exp(-(x**2) / r0**2)

# Generate one fixed random medium (10 slices at z_c = 1)
rng = np.random.default_rng(42)
mu_slices_base = generate_medium_slices(x, sigma, x_c, z_c, L, rng)
n_slices = mu_slices_base.shape[0]  # 10

# Convergence study
M_values = [1, 2, 4, 8, 16, 32, 64]
M_ref = 256


def propagate_with_substeps(phi0, x, k, mu_slices_base, z_c, M):
    """Propagate through random medium with M sub-steps per slice."""
    h = z_c / M
    mu_fine = np.repeat(mu_slices_base, M, axis=0)
    n_steps = mu_fine.shape[0]
    return split_step_propagate(phi0, x, k, h, n_steps, mu_fine)


# Reference solution
phi_ref = propagate_with_substeps(phi0, x, k, mu_slices_base, z_c, M_ref)
norm_ref = np.sqrt(np.sum(np.abs(phi_ref) ** 2))

errors = []
h_values = []
for M in M_values:
    h = z_c / M
    phi_h = propagate_with_substeps(phi0, x, k, mu_slices_base, z_c, M)
    err = np.sqrt(np.sum(np.abs(phi_h - phi_ref) ** 2)) / norm_ref
    errors.append(err)
    h_values.append(h)
    print(f"  M = {M:2d},  h = {h:.4f},  relative L2 error = {err:.6e}")

h_values = np.array(h_values)
errors = np.array(errors)

# Print convergence ratios in the asymptotic regime
print("\nConvergence ratios (asymptotic regime):")
for i in range(1, len(M_values)):
    if errors[i] > 0 and errors[i - 1] > 0:
        ratio = errors[i - 1] / errors[i]
        print(f"  h={h_values[i-1]:.4f} -> h={h_values[i]:.4f}:  ratio = {ratio:.2f}  (expected 4.0 for O(h^2))")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

ax.loglog(
    h_values, errors, "o-", color="C0", linewidth=1.5, markersize=7,
    label="Erreur $L^2$ relative", zorder=3,
)

# O(h^2) reference line fitted through the asymptotic points (M >= 8)
mask_asymptotic = np.array(h_values) <= 0.13
if mask_asymptotic.sum() >= 2:
    # Fit log(err) = 2*log(h) + log(C) on asymptotic points
    log_h = np.log(h_values[mask_asymptotic])
    log_e = np.log(errors[mask_asymptotic])
    slope, intercept = np.polyfit(log_h, log_e, 1)
    C_fit = np.exp(intercept)
    print(f"\nFitted slope in asymptotic regime: {slope:.2f} (expected 2.0)")
    h_line = np.logspace(np.log10(h_values[-1] * 0.5), np.log10(h_values[0] * 1.5), 100)
    ax.loglog(h_line, C_fit * h_line**2, "--", color="gray", linewidth=1,
              label="$O(h^2)$ (pente: %.2f)" % slope)

# Mark the nominal step h = z_c = 1
ax.axvline(x=1.0, color="C3", linestyle=":", linewidth=1, alpha=0.7)
ax.annotate(
    "$h = z_c$ (nominal)", xy=(1.0, errors[0]),
    xytext=(0.25, errors[0] * 1.5), fontsize=9, color="C3",
    arrowprops=dict(arrowstyle="->", color="C3", lw=0.8),
)

ax.set_xlabel("Taille du pas $h = z_c / M$", fontsize=11)
ax.set_ylabel("Erreur $L^2$ relative $\\varepsilon(h)$", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(__file__), "..", "report", "figures", "convergence_split_step.pdf"
)
fig.savefig(out_path, bbox_inches="tight")
print(f"\nFigure saved to {out_path}")
plt.close()
