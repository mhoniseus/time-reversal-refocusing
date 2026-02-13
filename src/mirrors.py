"""
Aperture functions for time-reversal mirrors.
"""

import numpy as np


def compact_mirror(x, r_M):
    """Miroir à support compact.

    chi_M(x) = (1 - (x / (2 r_M))^2)^2  pour |x| <= 2 r_M, sinon 0.
    """
    chi = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= 2 * r_M
    chi[mask] = (1 - (x[mask] / (2 * r_M)) ** 2) ** 2
    return chi


def gaussian_mirror(x, r_M):
    """Miroir gaussien.

    chi_M(x) = exp(-x^2 / r_M^2).
    """
    return np.exp(-x**2 / r_M**2)
