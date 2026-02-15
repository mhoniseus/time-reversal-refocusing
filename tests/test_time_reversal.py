"""
Tests pour le module de refocalisation par retournement temporel.

Vérifie :
    - time_reverse_field produit conj(phi_t) * chi_M.
    - Les formes de sortie de refocus_homogeneous sont correctes.
    - La refocalisation avec miroir gaussien dans un milieu homogène correspond
      a la formule analytique du sujet.
    - Le champ refocalisé a un pic près de l'origine.
"""

import numpy as np
import numpy.testing as npt

from src.mirrors import gaussian_mirror, compact_mirror
from src.paraxial_solver import fourier_propagate
from src.time_reversal import (
    time_reverse_field,
    refocus_homogeneous,
    refocus_random_same_medium,
    refocus_random_homogeneous_back,
)
from src.theory import (
    phi_t_homogeneous,
    refocused_gaussian_mirror_homogeneous,
)
from src.random_medium import generate_medium_slices


class TestTimeReverseField:
    """Tests pour l'opération de base du retournement temporel."""

    def test_conjugate_times_mirror(self, x, r_M):
        """time_reverse_field devrait renvoyer conj(phi_t) * chi_M exactement."""
        phi_t = (1 + 2j) * np.exp(-x**2 / 9.0) + 0.5j * np.sin(x)
        chi_M = gaussian_mirror(x, r_M)

        result = time_reverse_field(phi_t, chi_M)
        expected = np.conj(phi_t) * chi_M

        npt.assert_allclose(
            result,
            expected,
            atol=1e-15,
            err_msg="time_reverse_field n'est pas conj(phi_t) * chi_M",
        )

    def test_real_input_unchanged_phase(self, x, r_M):
        """Pour une entrée à valeurs réelles, la conjugaison est un no-op, donc
        time_reverse_field(phi, chi) == phi * chi."""
        phi_real = np.exp(-x**2 / 4.0)
        chi_M = compact_mirror(x, r_M)

        result = time_reverse_field(phi_real, chi_M)
        expected = phi_real * chi_M

        npt.assert_allclose(
            result,
            expected,
            atol=1e-15,
            err_msg="time_reverse_field d'une entrée réelle diffère de phi * chi",
        )

    def test_output_shape(self, x, r_M):
        """La forme de sortie devrait correspondre à l'entrée."""
        phi_t = np.ones(len(x), dtype=complex)
        chi_M = gaussian_mirror(x, r_M)
        result = time_reverse_field(phi_t, chi_M)
        assert result.shape == x.shape


class TestRefocusHomogeneous:
    """Tests pour la refocalisation dans un milieu homogène."""

    def test_output_shapes(self, phi0, x, k, L, r_M):
        """phi_t et phi_r devraient avoir la même forme que l'entrée."""
        chi_M = gaussian_mirror(x, r_M)
        phi_t, phi_r = refocus_homogeneous(phi0, x, k, L, chi_M)
        assert phi_t.shape == x.shape, "Forme de phi_t incorrecte"
        assert phi_r.shape == x.shape, "Forme de phi_r incorrecte"

    def test_phi_t_matches_analytical(self, phi0, x, k, r0, L, r_M):
        """Le champ transmis devrait correspondre au faisceau gaussien analytique."""
        chi_M = gaussian_mirror(x, r_M)
        phi_t, _ = refocus_homogeneous(phi0, x, k, L, chi_M)
        phi_t_exact = phi_t_homogeneous(x, r0, k, L)

        central = np.abs(x) < 30.0
        npt.assert_allclose(
            np.abs(phi_t[central]),
            np.abs(phi_t_exact[central]),
            atol=1e-6,
            err_msg="Le champ transmis de refocus_homogeneous s'écarte de l'analytique",
        )

    def test_gaussian_mirror_refocusing_matches_formula(self, phi0, x, k, r0, L, r_M):
        """Avec un miroir gaussien, le profil refocalisé devrait correspondre à
        la formule en forme fermée du sujet."""
        chi_M = gaussian_mirror(x, r_M)
        _, phi_r = refocus_homogeneous(phi0, x, k, L, chi_M)
        phi_r_exact = refocused_gaussian_mirror_homogeneous(x, r0, r_M, k, L)

        # Comparer dans la region centrale ou le faisceau refocalisé a un support.
        central = np.abs(x) < 20.0
        npt.assert_allclose(
            np.abs(phi_r[central]),
            np.abs(phi_r_exact[central]),
            atol=1e-4,
            err_msg="L'intensite refocalisée avec miroir gaussien ne correspond pas a la formule analytique",
        )
        # Vérifier aussi la structure de phase via le recouvrement normalise.
        inner = np.sum(phi_r[central] * np.conj(phi_r_exact[central]))
        inner /= np.abs(inner)
        npt.assert_allclose(
            np.abs(inner),
            1.0,
            atol=1e-4,
            err_msg="La phase refocalisée avec miroir gaussien ne correspond pas a la formule analytique",
        )

    def test_refocused_field_peaks_at_origin(self, phi0, x, k, L, r_M):
        """L'intensite refocalisée devrait avoir un pic près de x = 0."""
        chi_M = gaussian_mirror(x, r_M)
        _, phi_r = refocus_homogeneous(phi0, x, k, L, chi_M)
        intensity = np.abs(phi_r)**2
        peak_idx = np.argmax(intensity)
        assert np.abs(x[peak_idx]) < 1.0, (
            f"Pic refocalisé en x = {x[peak_idx]:.2f}, attendu près de 0"
        )

    def test_compact_mirror_peaks_at_origin(self, phi0, x, k, L, r_M):
        """La refocalisation avec un miroir compact devrait aussi avoir un pic a l'origine."""
        chi_M = compact_mirror(x, r_M)
        _, phi_r = refocus_homogeneous(phi0, x, k, L, chi_M)
        intensity = np.abs(phi_r)**2
        peak_idx = np.argmax(intensity)
        assert np.abs(x[peak_idx]) < 1.0, (
            f"Pic refocalisé avec miroir compact en x = {x[peak_idx]:.2f}, attendu près de 0"
        )


class TestRefocusRandomSameMedium:
    """Tests pour la refocalisation à travers le même milieu aléatoire."""

    def test_output_shapes(self, phi0, x, k, L, h, r_M, sigma, x_c, z_c, rng):
        """phi_t et phi_r devraient avoir la forme correcte."""
        chi_M = gaussian_mirror(x, r_M)
        mu_slices = generate_medium_slices(x, sigma, x_c, z_c, L, rng)
        phi_t, phi_r = refocus_random_same_medium(phi0, x, k, L, h, chi_M, mu_slices)
        assert phi_t.shape == x.shape, "Forme de phi_t incorrecte"
        assert phi_r.shape == x.shape, "Forme de phi_r incorrecte"

    def test_refocused_field_peaks_near_origin(self, phi0, x, k, L, h, r_M, sigma, x_c, z_c, rng):
        """Lors de la rétro-propagation à travers le même milieu, le champ refocalisé
        devrait toujours avoir un pic près de x = 0 (auto-moyennage statistique)."""
        chi_M = gaussian_mirror(x, r_M)
        mu_slices = generate_medium_slices(x, sigma, x_c, z_c, L, rng)
        _, phi_r = refocus_random_same_medium(phi0, x, k, L, h, chi_M, mu_slices)
        intensity = np.abs(phi_r)**2
        peak_idx = np.argmax(intensity)
        assert np.abs(x[peak_idx]) < 2.0, (
            f"Pic refocalisé (même milieu) en x = {x[peak_idx]:.2f}, attendu près de 0"
        )


class TestRefocusRandomHomogeneousBack:
    """Tests pour la refocalisation : propagation avant aléatoire / retour homogène."""

    def test_output_shapes(self, phi0, x, k, L, h, r_M, sigma, x_c, z_c, rng):
        """phi_t et phi_r devraient avoir la forme correcte."""
        chi_M = gaussian_mirror(x, r_M)
        mu_slices = generate_medium_slices(x, sigma, x_c, z_c, L, rng)
        phi_t, phi_r = refocus_random_homogeneous_back(
            phi0, x, k, L, h, chi_M, mu_slices
        )
        assert phi_t.shape == x.shape, "Forme de phi_t incorrecte"
        assert phi_r.shape == x.shape, "Forme de phi_r incorrecte"

    def test_homogeneous_limit(self, phi0, x, k, L, h, r_M):
        """Avec un potentiel nul, propagation avant aléatoire + retour homogène devrait
        être égal à la refocalisation entierement homogène."""
        chi_M = gaussian_mirror(x, r_M)
        n_steps = int(L / h)
        mu_zero = np.zeros((n_steps, len(x)))
        _, phi_r_rand = refocus_random_homogeneous_back(
            phi0, x, k, L, h, chi_M, mu_zero
        )
        _, phi_r_homo = refocus_homogeneous(phi0, x, k, L, chi_M)

        npt.assert_allclose(
            phi_r_rand,
            phi_r_homo,
            atol=1e-9,
            err_msg=(
                "Le retour aléatoire-homogène avec potentiel nul "
                "s'écarte de la refocalisation purement homogène"
            ),
        )
