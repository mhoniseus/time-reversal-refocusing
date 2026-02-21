"""
Regenerate Part 5 figures: broadband time-reversal refocusing.

Demonstrates statistical stability of broadband time-reversal:
  - Left panel:  Broadband same-medium TR  (stable, low CV)
  - Right panel: Narrowband, different-medium (independent fwd/bwd) TR (unstable)

The broadband same-medium case benefits from both phase compensation
(same medium) and frequency averaging. The narrowband different-medium
case has large fluctuations because the backward medium doesn't undo
the forward scattering.

Run from repo root:
    python scripts/regenerate_part5_figures.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.paraxial_solver import split_step_propagate
from src.random_medium import generate_medium_slices
from src.mirrors import compact_mirror
from src.time_reversal import time_reverse_field
from src.theory import mean_refocused_same_medium

# Parameters
r0 = 2.0
c0 = 1.0
L = 10.0

z_c = 1.0
x_c = 1.5
sigma = 2.0
h = 0.5
n_steps = int(L / h)
steps_per_slice = int(z_c / h)
gamma0 = sigma**2 * z_c
gamma2 = 2 * sigma**2 * z_c / x_c**2

omega_0 = 3.0
B = 2.0
N_omega = 30
r_M = 8.0
omegas = np.linspace(omega_0 - B, omega_0 + B, N_omega)

k_nb = omega_0 / c0

N_grid = 2**11
x_max = 150.0
dx = x_max / N_grid
x = np.arange(-N_grid // 2, N_grid // 2) * dx
phi0 = np.exp(-x**2 / r0**2)
center_idx = N_grid // 2

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

dw_corr = 2.0 / np.sqrt(gamma0 * L)
N_eff_est = 2 * B / dw_corr

print(f"Grid: N={N_grid}, dx={dx:.4f}, x in [{x[0]:.1f}, {x[-1]:.1f}]")
print(f"omega_0={omega_0}, B={B}, h={h}, sigma={sigma}")
print(f"Frequencies: {omegas[0]:.2f} to {omegas[-1]:.2f}, N_omega={N_omega}")
print(f"r_M = {r_M}")

plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3,
                     'lines.linewidth': 1.8, 'figure.dpi': 110})


# Helper functions
def expand_slices(mu_slices):
    """Repeat each z_c-wide slice for steps_per_slice sub-steps."""
    return np.repeat(mu_slices, steps_per_slice, axis=0)


def broadband_refocus_same(x, r0, r_M, omegas, h, n_steps, mu_slices):
    """Broadband same-medium TR: coherent sum of fields (pulse peak at t=0).

    Returns |u_BB(x)|^2 where u_BB = (1/N) sum_j phi_r_j.
    At x=0 the fields are approximately real (phase compensation),
    so the peak is stable. At x!=0, random phases across frequencies
    cancel the sidelobes -- this IS the self-averaging effect.
    """
    chi = compact_mirror(x, r_M)
    u_bb = np.zeros(len(x), dtype=complex)
    for omega_j in omegas:
        k_j = omega_j / c0
        phi0_j = np.exp(-x**2 / r0**2)
        phi_t = split_step_propagate(phi0_j, x, k_j, h, n_steps, mu_slices)
        phi_tr = time_reverse_field(phi_t, chi)
        phi_r = split_step_propagate(phi_tr, x, k_j, h, n_steps,
                                     mu_slices[::-1])
        u_bb += phi_r
    return np.abs(u_bb / len(omegas))**2


def narrowband_refocus_same(x, r0, r_M, k, h, n_steps, mu_slices):
    """Narrowband same-medium TR."""
    chi = compact_mirror(x, r_M)
    phi0_j = np.exp(-x**2 / r0**2)
    phi_t = split_step_propagate(phi0_j, x, k, h, n_steps, mu_slices)
    phi_tr = time_reverse_field(phi_t, chi)
    phi_r = split_step_propagate(phi_tr, x, k, h, n_steps, mu_slices[::-1])
    return phi_r


def narrowband_refocus_diff(x, r0, r_M, k, h, n_steps, mu_fwd, mu_bwd):
    """Narrowband different-medium TR: forward and backward through independent media."""
    chi = compact_mirror(x, r_M)
    phi0_j = np.exp(-x**2 / r0**2)
    phi_t = split_step_propagate(phi0_j, x, k, h, n_steps, mu_fwd)
    phi_tr = time_reverse_field(phi_t, chi)
    phi_r = split_step_propagate(phi_tr, x, k, h, n_steps, mu_bwd[::-1])
    return phi_r


def broadband_theory_same(x, r0, r_M, omegas, L, gamma2):
    """Theoretical mean broadband field (coherent sum), then |.|^2."""
    u = np.zeros(len(x), dtype=complex)
    for omega_j in omegas:
        k_j = omega_j / c0
        u += mean_refocused_same_medium(
            x, r0, r_M, k_j, L, omega_j, gamma2)
    return np.abs(u / len(omegas))**2


# Monte Carlo runs
print("=" * 60)
print("Running Monte Carlo simulations")
print("=" * 60)

n_mc = 150
n_show = 10


def recenter(profile, x, center_idx):
    """Shift profile so its peak aligns with x=0 (removes beam wander)."""
    peak_idx = np.argmax(profile)
    return np.roll(profile, center_idx - peak_idx)


# Broadband same-medium MC
all_bb_intens = []
rng_bb = np.random.default_rng(999)

for i in tqdm(range(n_mc), desc='Broadband same-medium'):
    mu_i = expand_slices(generate_medium_slices(x, sigma, x_c, z_c, L, rng_bb))
    E_i = broadband_refocus_same(x, r0, r_M, omegas, h, n_steps, mu_i)
    all_bb_intens.append(recenter(E_i, x, center_idx))

all_bb_intens = np.array(all_bb_intens)
mean_intens_bb = np.mean(all_bb_intens, axis=0)

# Narrowband same-medium MC (same seed)
all_nb_same = []
rng_nb_s = np.random.default_rng(999)

for i in tqdm(range(n_mc), desc='Narrowband same-medium'):
    mu_i = expand_slices(generate_medium_slices(x, sigma, x_c, z_c, L, rng_nb_s))
    phi_r = narrowband_refocus_same(x, r0, r_M, k_nb, h, n_steps, mu_i)
    I_i = np.abs(phi_r)**2
    all_nb_same.append(recenter(I_i, x, center_idx))

all_nb_same = np.array(all_nb_same)
mean_nb_same = np.mean(all_nb_same, axis=0)

# Narrowband different-medium MC
all_nb_diff = []
rng_nb_d = np.random.default_rng(999)

for i in tqdm(range(n_mc), desc='Narrowband diff-medium'):
    mu_fwd = expand_slices(generate_medium_slices(x, sigma, x_c, z_c, L, rng_nb_d))
    mu_bwd = expand_slices(generate_medium_slices(x, sigma, x_c, z_c, L, rng_nb_d))
    phi_r = narrowband_refocus_diff(x, r0, r_M, k_nb, h, n_steps, mu_fwd, mu_bwd)
    all_nb_diff.append(np.abs(phi_r)**2)

all_nb_diff = np.array(all_nb_diff)
mean_nb_diff = np.mean(all_nb_diff, axis=0)

# Theory (no wander, so it matches re-centered MC)
theory_bb = broadband_theory_same(x, r0, r_M, omegas, L, gamma2)
theory_nb_same = mean_refocused_same_medium(x, r0, r_M, k_nb, L, omega_0, gamma2)

# CV: measure as profile-averaged normalized std (captures sidelobe self-averaging)
mask_cv = (x > -6) & (x < 6)
cv_bb = np.mean(np.std(all_bb_intens[:, mask_cv], axis=0)) / np.mean(mean_intens_bb[mask_cv])
cv_nb_same = np.mean(np.std(all_nb_same[:, mask_cv], axis=0)) / np.mean(mean_nb_same[mask_cv])
cv_nb_diff = np.mean(np.std(all_nb_diff[:, mask_cv], axis=0)) / np.mean(mean_nb_diff[mask_cv])
print(f"\nCV broadband same-medium  (N_omega={N_omega}): {cv_bb:.4f}")
print(f"CV narrowband same-medium (single omega):    {cv_nb_same:.4f}")
print(f"CV narrowband diff-medium (single omega):    {cv_nb_diff:.4f}\n")


# Figure 1: Broadband mean profile vs theory
print("Generating part5_broadband_mean.pdf ...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, mean_intens_bb, 'k-', linewidth=2, label='Moyenne MC (200 real.)')
ax.plot(x, theory_bb, 'r--', linewidth=2, label='Theorie')
ax.set_xlim(-8, 8)
ax.set_xlabel('$x$')
ax.set_ylabel('$|\\Phi_r^{\\mathrm{tr,bb}}|^2$')
ax.set_title(f'Partie 5 -- Profil large bande moyen '
             f'($\\omega_0={omega_0}$, $B={B}$, $N_\\omega={N_omega}$)')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'part5_broadband_mean.pdf'),
            bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved.")


# Figure 2: Individual broadband realizations
print("Generating part5_broadband_single.pdf ...")
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(min(10, n_mc)):
    ax.plot(x, all_bb_intens[i], alpha=0.4, linewidth=0.8, color='steelblue')
ax.plot(x, mean_intens_bb, 'k-', linewidth=2, label='Moyenne MC')
ax.plot(x, theory_bb, 'r--', linewidth=2, label='Theorie')
ax.set_xlim(-8, 8)
ax.set_xlabel('$x$')
ax.set_ylabel('$|\\Phi_r^{\\mathrm{tr,bb}}|^2$')
ax.set_title(f'Partie 5 -- Realisations individuelles large bande')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'part5_broadband_single.pdf'),
            bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved.")


# Figure 3: Statistical stability
print("Generating part5_statistical_stability.pdf ...")

# Percentile bands on raw (un-normalized) profiles
# Left panel: NB different-medium (unstable, CV~1.5)
# Right panel: BB same-medium (stable, CV~0.75)
p10_diff = np.percentile(all_nb_diff, 10, axis=0)
p90_diff = np.percentile(all_nb_diff, 90, axis=0)

p10_bb = np.percentile(all_bb_intens, 10, axis=0)
p90_bb = np.percentile(all_bb_intens, 90, axis=0)

# Common y-axis scale
mask_x = (x > -8) & (x < 8)
y_max = max(np.max(p90_diff[mask_x]), np.max(p90_bb[mask_x])) * 1.1

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: narrowband different-medium (very unstable -- no phase compensation)
axes[0].fill_between(x, p10_diff, p90_diff, alpha=0.25, color='coral',
                     label='10$^e$--90$^e$ perc.')
for i in range(min(n_show, n_mc)):
    axes[0].plot(x, all_nb_diff[i], alpha=0.2, linewidth=0.5, color='coral')
axes[0].plot(x, mean_nb_diff, 'k-', linewidth=2, label='Moyenne MC')
axes[0].set_xlim(-8, 8)
axes[0].set_ylim(0, y_max)
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$|\\varphi_r|^2$')
axes[0].set_title(f'NB milieu diff. ($\\omega = {omega_0}$, '
                  f'CV$={cv_nb_diff:.2f}$)')
axes[0].legend()

# Right: broadband same-medium (stable -- phase compensation + self-averaging)
axes[1].fill_between(x, p10_bb, p90_bb, alpha=0.25, color='forestgreen',
                     label='10$^e$--90$^e$ perc.')
for i in range(min(n_show, n_mc)):
    axes[1].plot(x, all_bb_intens[i], alpha=0.2, linewidth=0.5,
                 color='forestgreen')
axes[1].plot(x, mean_intens_bb, 'k-', linewidth=2, label='Moyenne MC')
axes[1].set_xlim(-8, 8)
axes[1].set_ylim(0, y_max)
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('$|\\Phi_r^{\\mathrm{tr,bb}}|^2$')
axes[1].set_title(f'BB meme milieu ($N_\\omega = {N_omega}$, '
                  f'CV$={cv_bb:.2f}$)')
axes[1].legend()

fig.suptitle(f'Partie 5 -- Stabilite statistique / auto-moyennage '
             f'($\\omega_0={omega_0}$, $B={B}$, $r_M={r_M}$)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'part5_statistical_stability.pdf'),
            bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved.")


# Figure 4: CV comparison bar chart
print("Generating part5_bandwidth_sweep.pdf ...")

fig, ax = plt.subplots(figsize=(8, 5))
labels = ['NB\nmeme milieu', f'BB (N={N_omega})\nmeme milieu', 'NB\nmilieu diff.']
cvs = [cv_nb_same, cv_bb, cv_nb_diff]
colors = ['steelblue', 'forestgreen', 'coral']
bars = ax.bar(labels, cvs, color=colors, width=0.5, edgecolor='black')
for bar, cv in zip(bars, cvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{cv:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_ylabel('Coefficient de variation en $x = 0$')
ax.set_title(f'Partie 5 -- Comparaison de stabilite '
             f'($\\omega_0={omega_0}$, $B={B}$, $r_M={r_M}$)')
ax.set_ylim(0, max(cvs) * 1.3)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'part5_bandwidth_sweep.pdf'),
            bbox_inches='tight', dpi=150)
plt.close(fig)
print("  Saved.")

print("\nDone.")
