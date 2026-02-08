"""
Fixtures partagées pour la suite de tests de refocalisation par retournement temporel paraxial.

Tous les paramètres suivent la spécification du sujet :
    r0 = 2, k = 1, omega = 1, L = 10
    x_max = 60, N = 1024
    Milieu aléatoire : h = 1, z_c = 1, x_c = 4, sigma = 1
"""

import numpy as np
import pytest


@pytest.fixture
def N():
    """Nombre de points de grille."""
    return 1024


@pytest.fixture
def x_max():
    """Demi-largeur du domaine spatial."""
    return 60.0


@pytest.fixture
def x(N, x_max):
    """Grille spatiale uniforme sur [-x_max, x_max)."""
    return np.linspace(-x_max, x_max, N, endpoint=False)


@pytest.fixture
def dx(x):
    """Pas de grille."""
    return x[1] - x[0]


@pytest.fixture
def k():
    """Nombre d'onde."""
    return 1.0


@pytest.fixture
def r0():
    """Largeur initiale du faisceau gaussien."""
    return 2.0


@pytest.fixture
def omega():
    """Fréquence angulaire."""
    return 1.0


@pytest.fixture
def L():
    """Distance de propagation."""
    return 10.0


@pytest.fixture
def h():
    """Taille du pas pour la méthode à pas fractionnés."""
    return 1.0


@pytest.fixture
def sigma():
    """Amplitude des fluctuations du potentiel aléatoire."""
    return 1.0


@pytest.fixture
def x_c():
    """Longueur de corrélation transversale."""
    return 4.0


@pytest.fixture
def z_c():
    """Longueur de corrélation longitudinale (épaisseur de tranche)."""
    return 1.0


@pytest.fixture
def phi0(x, r0):
    """Faisceau gaussien initial  phi_0(x) = exp(-x^2 / r0^2)."""
    return np.exp(-x**2 / r0**2)


@pytest.fixture
def r_M():
    """Demi-largeur du miroir."""
    return 20.0


@pytest.fixture
def rng():
    """Générateur de nombres aléatoires avec graine pour la reproductibilité."""
    return np.random.default_rng(42)
