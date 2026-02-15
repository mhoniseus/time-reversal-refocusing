"""
Tests pour les solveurs paraxiaux de Fourier et à pas fractionnés de Fourier.

Vérifie :
    - La propagation de Fourier correspond a la solution analytique du faisceau gaussien.
    - Le pas fractionné sans potentiel correspond a la propagation de Fourier.
    - La propagation préserve la norme L2 (conservation de l'energie).
    - Une distance de propagation nulle renvoie l'entrée inchangée.
"""

import numpy as np
import numpy.testing as npt

from src.paraxial_solver import fourier_propagate, split_step_propagate
from src.theory import phi_t_homogeneous


class TestFourierPropagate:
    """Tests du propagateur de Fourier contre les résultats analytiques connus."""

    def test_gaussian_beam_matches_analytical(self, x, k, r0, L):
        """La propagation de Fourier d'une gaussienne devrait correspondre à la forme fermée
        phi_t(x) = (r0/r_t) exp(-x^2/r_t^2) du sujet."""
        phi0 = np.exp(-x**2 / r0**2)
        phi_num = fourier_propagate(phi0, x, k, L)

        phi_exact = phi_t_homogeneous(x, r0, k, L)

        # Comparer uniquement dans la region centrale ou le faisceau a un support,
        # en evitant les artefacts de repliement près de la frontière du domaine.
        central = np.abs(x) < 30.0
        npt.assert_allclose(
            np.abs(phi_num[central]),
            np.abs(phi_exact[central]),
            atol=1e-6,
            err_msg="L'amplitude de la propagation de Fourier ne correspond pas au faisceau gaussien analytique",
        )
        # Vérifier aussi les phases (à une constante globale près) via le produit scalaire normalise.
        overlap = np.sum(phi_num[central] * np.conj(phi_exact[central]))
        overlap /= np.abs(overlap)
        npt.assert_allclose(
            np.abs(overlap),
            1.0,
            atol=1e-6,
            err_msg="Le motif de phase de la propagation de Fourier s'écarte de l'analytique",
        )

    def test_short_distance_matches_analytical(self, x, k, r0):
        """Vérification pour une très courte distance de propagation ou la diffraction est faible."""
        dz = 0.1
        phi0 = np.exp(-x**2 / r0**2)
        phi_num = fourier_propagate(phi0, x, k, dz)
        phi_exact = phi_t_homogeneous(x, r0, k, dz)

        npt.assert_allclose(
            phi_num,
            phi_exact,
            atol=1e-8,
            err_msg="La propagation de Fourier a courte distance s'écarte de l'analytique",
        )

    def test_zero_distance_returns_input(self, x, k, phi0):
        """Propager sur une distance nulle devrait renvoyer le champ original."""
        result = fourier_propagate(phi0, x, k, dz=0.0)
        npt.assert_allclose(
            result,
            phi0,
            atol=1e-14,
            err_msg="La propagation a distance nulle a modifie le champ",
        )

    def test_preserves_l2_norm(self, x, k, L, dx, phi0):
        """La propagation de Fourier est unitaire : la norme L2 doit être conservée."""
        norm_before = np.sum(np.abs(phi0)**2) * dx
        phi_out = fourier_propagate(phi0, x, k, L)
        norm_after = np.sum(np.abs(phi_out)**2) * dx
        npt.assert_allclose(
            norm_after,
            norm_before,
            rtol=1e-12,
            err_msg="La propagation de Fourier ne préserve pas la norme L2",
        )


class TestSplitStepPropagate:
    """Tests du propagateur à pas fractionnés."""

    def test_no_potential_matches_fourier(self, x, k, L, h, phi0):
        """Le pas fractionné avec mu_slices=None devrait donner le même résultat qu'une
        seule propagation de Fourier sur la distance L."""
        n_steps = int(L / h)
        phi_split = split_step_propagate(phi0, x, k, h, n_steps, mu_slices=None)
        phi_fourier = fourier_propagate(phi0, x, k, L)

        npt.assert_allclose(
            phi_split,
            phi_fourier,
            atol=1e-10,
            err_msg="Le pas fractionné (sans potentiel) s'écarte de la propagation de Fourier",
        )

    def test_no_potential_preserves_l2_norm(self, x, k, h, dx, phi0):
        """La norme L2 devrait être conservée même avec de nombreux pas fractionnés."""
        n_steps = 20
        norm_before = np.sum(np.abs(phi0)**2) * dx
        phi_out = split_step_propagate(phi0, x, k, h, n_steps, mu_slices=None)
        norm_after = np.sum(np.abs(phi_out)**2) * dx
        npt.assert_allclose(
            norm_after,
            norm_before,
            rtol=1e-10,
            err_msg="Le pas fractionné (sans potentiel) ne préserve pas la norme L2",
        )

    def test_with_potential_preserves_l2_norm(self, x, k, h, dx, phi0, rng):
        """Avec un potentiel à valeurs réelles, le pas fractionné devrait toujours preserver
        la norme L2 (chaque facteur exponentiel a un module unité)."""
        n_steps = 10
        N = len(x)
        mu_slices = rng.standard_normal((n_steps, N))

        norm_before = np.sum(np.abs(phi0)**2) * dx
        phi_out = split_step_propagate(phi0, x, k, h, n_steps, mu_slices=mu_slices)
        norm_after = np.sum(np.abs(phi_out)**2) * dx
        npt.assert_allclose(
            norm_after,
            norm_before,
            rtol=1e-10,
            err_msg="Le pas fractionné (avec potentiel) ne préserve pas la norme L2",
        )

    def test_zero_steps_returns_input(self, x, k, h, phi0):
        """Zero pas devrait renvoyer le champ initial inchangé."""
        result = split_step_propagate(phi0, x, k, h, n_steps=0, mu_slices=None)
        npt.assert_allclose(
            result,
            phi0,
            atol=1e-14,
            err_msg="Le pas fractionné a zero pas a modifie le champ",
        )

    def test_single_step_matches_fourier(self, x, k, phi0):
        """Un seul pas fractionné sans potentiel devrait être égal à fourier_propagate
        avec dz = h."""
        h_val = 0.5
        phi_split = split_step_propagate(phi0, x, k, h_val, n_steps=1, mu_slices=None)
        phi_fourier = fourier_propagate(phi0, x, k, dz=h_val)
        npt.assert_allclose(
            phi_split,
            phi_fourier,
            atol=1e-13,
            err_msg="Un seul pas fractionné s'écarte d'un seul pas de Fourier",
        )
