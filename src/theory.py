"""
Formules analytiques en forme fermée du sujet pour vérification.

Auteur : Mouhssine Rifaki
"""

import numpy as np


def phi_t_homogeneous(x, r0, k, L):
    """Champ transmis dans un milieu homogène.

    phi_t(x) = (r0 / r_t) exp(-x^2 / r_t^2)
    r_t = r0 (1 + 2i L / (k r0^2))^{1/2}
    """
    r_t = r0 * np.sqrt(1 + 2j * L / (k * r0**2))
    return (r0 / r_t) * np.exp(-x**2 / r_t**2)


def intensity_t_homogeneous(x, r0, k, L):
    """|phi_t(x)|^2 dans un milieu homogène.

    |phi_t|^2 = (r0 / R_t) exp(-2 x^2 / R_t^2)
    R_t = r0 (1 + 4 L^2 / (k^2 r0^4))^{1/2}
    """
    R_t = r0 * np.sqrt(1 + 4 * L**2 / (k**2 * r0**4))
    return (r0 / R_t) * np.exp(-2 * x**2 / R_t**2)


def refocused_gaussian_mirror_homogeneous(x, r0, r_M, k, L):
    """Profil refocalisé analytique avec miroir gaussien en milieu homogène.

    phi_r^tr(x) = (1 / a_tr) exp(-x^2 / r_tr^2)

    r_tr^2 = (1/r_M^2 + 1/(r0^2 - 2iL/k))^{-1} + 2i L / k
    a_tr   = (1 + 4L^2/(k^2 r0^2 r_M^2) + 2iL/(k r_M^2))^{1/2}
    """
    r_tr_sq = 1.0 / (1.0 / r_M**2 + 1.0 / (r0**2 - 2j * L / k)) + 2j * L / k
    a_tr = np.sqrt(1 + 4 * L**2 / (k**2 * r0**2 * r_M**2)
                   + 2j * L / (k * r_M**2))
    return (1.0 / a_tr) * np.exp(-x**2 / r_tr_sq)


def mean_phi_t_random(x, r0, k, L, omega, gamma0):
    """Champ transmis moyen E[phi_t(x)] dans un milieu aléatoire.

    E[phi_t] = (r0/r_t) exp(-x^2/r_t^2) exp(-gamma0 omega^2 L / 8)
    gamma0 = sigma^2 z_c
    """
    phi_homo = phi_t_homogeneous(x, r0, k, L)
    return phi_homo * np.exp(-gamma0 * omega**2 * L / 8)


def _refocus_params(r0, r_M, k, L):
    """Paramètres r_tr^2 et a_tr partagés pour les formules de refocalisation."""
    r_tr_sq = 1.0 / (1.0 / r_M**2 + 1.0 / (r0**2 - 2j * L / k)) + 2j * L / k
    a_tr = np.sqrt(1 + 4 * L**2 / (k**2 * r0**2 * r_M**2)
                   + 2j * L / (k * r_M**2))
    return r_tr_sq, a_tr


def mean_refocused_same_medium(x, r0, r_M, k, L, omega, gamma2):
    """E[phi_r^tr(x)] lors de la rétro-propagation à travers le même milieu aléatoire.

    = (1/a_tr) exp(-x^2/r_tr^2) exp(-x^2/r_a^2)
    r_a^{-2} = gamma2 omega^2 L / 48
    gamma2 = 2 sigma^2 z_c / x_c^2
    """
    r_tr_sq, a_tr = _refocus_params(r0, r_M, k, L)
    r_a_sq = 48.0 / (gamma2 * omega**2 * L)
    return (1.0 / a_tr) * np.exp(-x**2 / r_tr_sq) * np.exp(-x**2 / r_a_sq)


def mean_refocused_homogeneous_back(x, r0, r_M, k, L, omega, gamma0):
    """E[phi_r^tr(x)] lors de la rétro-propagation à travers un milieu homogène.

    = (1/a_tr) exp(-x^2/r_tr^2) exp(-gamma0 omega^2 L / 8)
    """
    r_tr_sq, a_tr = _refocus_params(r0, r_M, k, L)
    return (1.0 / a_tr) * np.exp(-x**2 / r_tr_sq) * np.exp(-gamma0 * omega**2 * L / 8)
