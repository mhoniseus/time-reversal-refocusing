"""
Génération de potentiel aléatoire pour le modèle paraxial hétérogène.

Génère des champs aléatoires gaussiens avec covariance
    E[mu(x) mu(x')] = sigma^2 exp(-(x-x')^2 / x_c^2)
en utilisant la méthode d'enchâssement circulant / méthode spectrale.

Auteur : Mouhssine Rifaki
"""

import numpy as np
from numpy.fft import fft, ifft


def generate_random_potential(x, sigma, x_c, rng=None):
    """Génère une réalisation d'un champ aléatoire gaussien à moyenne nulle.

    Paramètres
    x : ndarray, forme (N,)
        Grille spatiale (espacement uniforme).
    sigma : float
        Amplitude des fluctuations.
    x_c : float
        Longueur de corrélation.
    rng : numpy.random.Generator, optionnel

    Retourne
    mu : ndarray, forme (N,)
        Realisation à valeurs réelles.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(x)
    dx = x[1] - x[0]

    # Vecteur de covariance circulante (distances périodiques)
    j = np.arange(N)
    dist = np.minimum(j, N - j) * dx
    cov = sigma**2 * np.exp(-dist**2 / x_c**2)

    # Valeurs propres de la matrice de covariance circulante
    lam = np.real(fft(cov))
    lam = np.maximum(lam, 0.0)

    # Synthèse spectrale
    w = rng.standard_normal(N)
    mu = np.real(ifft(np.sqrt(lam) * fft(w)))

    return mu


def generate_medium_slices(x, sigma, x_c, z_c, L, rng=None):
    """Génère toutes les tranches de potentiel aléatoire indépendantes pour z dans [0, L].

    Le potentiel est constant par morceaux :
        mu(z, x) = mu_n(x)  pour z dans [n z_c, (n+1) z_c).

    Paramètres
    x : ndarray, forme (N,)
    sigma, x_c : float
        Paramètres de covariance.
    z_c : float
        Épaisseur de tranche (longueur de corrélation en z).
    L : float
        Longueur totale de propagation.
    rng : numpy.random.Generator, optionnel

    Retourne
    mu_slices : ndarray, forme (n_slices, N)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_slices = int(L / z_c)
    mu_slices = np.empty((n_slices, len(x)))
    for n in range(n_slices):
        mu_slices[n] = generate_random_potential(x, sigma, x_c, rng)

    return mu_slices
