"""
Protocoles de refocalisation par retournement temporel pour l'équation paraxiale.

Auteur : Mouhssine Rifaki
"""

import numpy as np
from .paraxial_solver import fourier_propagate, split_step_propagate


def time_reverse_field(phi_t, chi_M):
    """Applique le retournement temporel au plan du miroir z = L.

    phi^tr(z=L, x) = conj(phi_t(x)) * chi_M(x)
    """
    return np.conj(phi_t) * chi_M


def refocus_homogeneous(phi0, x, k, L, chi_M):
    """Expérience complete de retournement temporel dans un milieu homogène.

    Propagation avant z : 0 -> L, retournement temporel en L, rétro-propagation z : L -> 2L.

    Retourne (phi_t, phi_r).
    """
    phi_t = fourier_propagate(phi0, x, k, L)
    phi_tr = time_reverse_field(phi_t, chi_M)
    phi_r = fourier_propagate(phi_tr, x, k, L)
    return phi_t, phi_r


def refocus_random_same_medium(phi0, x, k, L, h, chi_M, mu_slices):
    """Propagation avant à travers le milieu aléatoire, retour à travers le *même* milieu.

    La rétro-propagation utilise les tranches en ordre inverse afin que
    l'onde retraverse les mêmes inhomogénéités.

    Retourne (phi_t, phi_r).
    """
    n_steps = int(L / h)
    phi_t = split_step_propagate(phi0, x, k, h, n_steps, mu_slices)
    phi_tr = time_reverse_field(phi_t, chi_M)
    phi_r = split_step_propagate(phi_tr, x, k, h, n_steps, mu_slices[::-1])
    return phi_t, phi_r


def refocus_random_homogeneous_back(phi0, x, k, L, h, chi_M, mu_slices):
    """Propagation avant à travers le milieu aléatoire, retour à travers un milieu *homogène*.

    Retourne (phi_t, phi_r).
    """
    n_steps = int(L / h)
    phi_t = split_step_propagate(phi0, x, k, h, n_steps, mu_slices)
    phi_tr = time_reverse_field(phi_t, chi_M)
    phi_r = fourier_propagate(phi_tr, x, k, L)
    return phi_t, phi_r
