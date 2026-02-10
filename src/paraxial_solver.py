"""
Solveurs de Fourier et de Fourier à pas fractionnés pour l'équation paraxiale (Schrödinger).

Résout :
  Homogene :  dz phi = (i/2k) dxx phi
  Aléatoire : dz phi = (i/2k) dxx phi + (ik/2) mu(z,x) phi

Auteur : Mouhssine Rifaki
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq


def fourier_propagate(phi, x, k, dz):
    """Propage phi sur une distance dz dans un milieu homogène.

    Dans l'espace de Fourier, l'EDO est diagonale :
        phi_hat(z+dz, kappa) = phi_hat(z, kappa) * exp(-i kappa^2 dz / (2k))

    Paramètres
    phi : ndarray, forme (N,)
        Champ à la position z courante.
    x : ndarray, forme (N,)
        Grille spatiale (espacement uniforme suppose).
    k : float
        Nombre d'onde k = omega / c_0.
    dz : float
        Distance de propagation.

    Retourne
    ndarray, forme (N,)
        Champ a z + dz.
    """
    dx = x[1] - x[0]
    N = len(x)
    kappa = 2 * np.pi * fftfreq(N, d=dx)
    phi_hat = fft(phi)
    phi_hat *= np.exp(-1j * kappa**2 * dz / (2 * k))
    return ifft(phi_hat)


def split_step_propagate(phi, x, k, h, n_steps, mu_slices=None):
    """Propage phi par la méthode de Fourier à pas fractionnés symétrique (Strang).

    Pour chaque pas de taille h :
        1. Demi-pas de potentiel :    phi <- phi * exp(i k mu h / 4)
        2. Pas complet de diffraction dans l'espace de Fourier
        3. Demi-pas de potentiel :    phi <- phi * exp(i k mu h / 4)

    Si mu_slices est None, les pas de potentiel sont ignorés (milieu homogène).

    Paramètres
    phi : ndarray, forme (N,)
        Champ initial.
    x : ndarray, forme (N,)
        Grille spatiale.
    k : float
        Nombre d'onde.
    h : float
        Taille du pas longitudinal.
    n_steps : int
        Nombre de pas.
    mu_slices : ndarray, forme (n_steps, N), optionnel
        Potentiel aléatoire pour chaque pas.

    Retourne
    ndarray, forme (N,)
        Champ après propagation sur n_steps * h.
    """
    dx = x[1] - x[0]
    N = len(x)
    kappa = 2 * np.pi * fftfreq(N, d=dx)
    diffraction = np.exp(-1j * kappa**2 * h / (2 * k))

    phi = np.array(phi, dtype=complex)

    for n in range(n_steps):
        if mu_slices is not None:
            phi *= np.exp(1j * k * mu_slices[n] * h / 4)

        phi = ifft(fft(phi) * diffraction)

        if mu_slices is not None:
            phi *= np.exp(1j * k * mu_slices[n] * h / 4)

    return phi
