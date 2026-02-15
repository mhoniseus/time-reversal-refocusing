"""
Tests pour les fonctions d'ouverture du miroir à retournement temporel.

Vérifie :
    - Support et valeur au pic du miroir compact.
    - Symetrie et pic du miroir gaussien.
    - Les deux miroirs sont non négatifs.
"""

import numpy as np
import numpy.testing as npt

from src.mirrors import compact_mirror, gaussian_mirror


class TestCompactMirror:
    """Tests pour compact_mirror  chi(x) = (1 - (x/(2 r_M))^2)^2  sur |x| <= 2 r_M."""

    def test_peak_value(self, x, r_M):
        """chi_M(0) devrait être égal à 1."""
        chi = compact_mirror(x, r_M)
        idx_center = np.argmin(np.abs(x))
        npt.assert_allclose(
            chi[idx_center],
            1.0,
            atol=1e-14,
            err_msg="Le pic du miroir compact n'est pas 1",
        )

    def test_support(self, x, r_M):
        """chi_M devrait être nul en dehors de |x| > 2 r_M."""
        chi = compact_mirror(x, r_M)
        outside = np.abs(x) > 2 * r_M
        npt.assert_allclose(
            chi[outside],
            0.0,
            atol=1e-14,
            err_msg="Le miroir compact est non nul en dehors de son support",
        )

    def test_non_negative(self, x, r_M):
        """Le miroir compact doit être non négatif partout."""
        chi = compact_mirror(x, r_M)
        assert np.all(chi >= -1e-15), "Le miroir compact à des valeurs négatives"

    def test_symmetry(self, x, r_M):
        """chi_M(x) devrait être une fonction paire."""
        chi = compact_mirror(x, r_M)
        chi_flip = compact_mirror(-x, r_M)
        npt.assert_allclose(
            chi,
            chi_flip,
            atol=1e-14,
            err_msg="Le miroir compact n'est pas symétrique",
        )

    def test_boundary_value(self, r_M):
        """chi_M(2 r_M) devrait être exactement 0."""
        x_bnd = np.array([2 * r_M, -2 * r_M])
        chi = compact_mirror(x_bnd, r_M)
        npt.assert_allclose(
            chi,
            0.0,
            atol=1e-14,
            err_msg="Le miroir compact est non nul a la frontière |x|=2 r_M",
        )

    def test_known_value(self, r_M):
        """Vérification de chi_M en x = r_M.  (1 - 1/4)^2 = (3/4)^2 = 9/16."""
        x_pt = np.array([r_M])
        chi = compact_mirror(x_pt, r_M)
        npt.assert_allclose(
            chi,
            (3.0 / 4.0) ** 2,
            atol=1e-14,
            err_msg="La valeur du miroir compact en x=r_M est incorrecte",
        )


class TestGaussianMirror:
    """Tests pour gaussian_mirror  chi(x) = exp(-x^2 / r_M^2)."""

    def test_peak_value(self, x, r_M):
        """chi_M(0) devrait être égal à 1."""
        chi = gaussian_mirror(x, r_M)
        idx_center = np.argmin(np.abs(x))
        npt.assert_allclose(
            chi[idx_center],
            1.0,
            atol=1e-14,
            err_msg="Le pic du miroir gaussien n'est pas 1",
        )

    def test_symmetry(self, x, r_M):
        """Le miroir gaussien devrait être une fonction paire."""
        chi = gaussian_mirror(x, r_M)
        chi_flip = gaussian_mirror(-x, r_M)
        npt.assert_allclose(
            chi,
            chi_flip,
            atol=1e-14,
            err_msg="Le miroir gaussien n'est pas symétrique",
        )

    def test_non_negative(self, x, r_M):
        """Le miroir gaussien doit être strictement positif."""
        chi = gaussian_mirror(x, r_M)
        assert np.all(chi > 0), "Le miroir gaussien à des valeurs non positives"

    def test_monotone_decrease(self, r_M):
        """Pour x >= 0, le miroir gaussien devrait être monotonement décroissant."""
        x_pos = np.linspace(0, 5 * r_M, 500)
        chi = gaussian_mirror(x_pos, r_M)
        assert np.all(np.diff(chi) <= 0), (
            "Le miroir gaussien n'est pas monotonement décroissant pour x >= 0"
        )

    def test_known_value(self, r_M):
        """chi_M(r_M) = exp(-1)."""
        x_pt = np.array([r_M])
        chi = gaussian_mirror(x_pt, r_M)
        npt.assert_allclose(
            chi,
            np.exp(-1.0),
            atol=1e-14,
            err_msg="La valeur du miroir gaussien en x=r_M est incorrecte",
        )

    def test_tails_decay(self, x, r_M):
        """Loin de l'origine, le miroir devrait être négligeablement petit."""
        chi = gaussian_mirror(x, r_M)
        far = np.abs(x) > 3 * r_M
        assert np.all(chi[far] < 1e-3), (
            "Les queues du miroir gaussien ne décroissent pas assez vite"
        )
