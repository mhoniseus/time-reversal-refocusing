"""
Tests pour le générateur de potentiel aléatoire.

Vérifie :
    - La forme de la sortie est correcte.
    - Les réalisations sont de moyenne nulle (approximativement).
    - La structure de covariance correspond au noyau gaussien prescrit (a un
      facteur de normalisation discrete N près de l'enchâssement circulant).
    - Les tranches sont indépendantes selon z.

Note sur la normalisation
La synthèse spectrale utilise  mu = real(ifft(sqrt(lam) * fft(w)))
ce qui donne une covariance ponctuelle de

    E[mu_i mu_j] = sigma^2 * exp(-(x_i - x_j)^2 / x_c^2).

Le facteur 1/N de ifft annule le facteur N de l'enchâssement circulant.
Les tests ci-dessous en tiennent compte.
"""

import numpy as np
import numpy.testing as npt

from src.random_medium import generate_random_potential, generate_medium_slices


class TestGenerateRandomPotential:
    """Tests pour generate_random_potential."""

    def test_output_shape(self, x, sigma, x_c, rng):
        """La sortie devrait avoir la même longueur que la grille spatiale."""
        mu = generate_random_potential(x, sigma, x_c, rng)
        assert mu.shape == x.shape, (
            f"Forme attendue {x.shape}, obtenue {mu.shape}"
        )

    def test_output_is_real(self, x, sigma, x_c, rng):
        """Le potentiel aléatoire doit être à valeurs réelles."""
        mu = generate_random_potential(x, sigma, x_c, rng)
        npt.assert_allclose(
            np.imag(mu),
            0.0,
            atol=1e-14,
            err_msg="Le potentiel aléatoire a une partie imaginaire non nulle",
        )

    def test_zero_mean_ensemble(self, x, sigma, x_c, N):
        """La moyenne d'ensemble sur de nombreuses réalisations devrait être approximativement nulle.

        L'écart-type ponctuel est sigma, donc l'erreur standard
        de la moyenne sur M échantillons est sigma / sqrt(M).
        """
        n_samples = 2000
        rng = np.random.default_rng(123)
        mu_sum = np.zeros(len(x))
        for _ in range(n_samples):
            mu_sum += generate_random_potential(x, sigma, x_c, rng)
        mu_mean = mu_sum / n_samples

        # Erreur standard ~ sigma / sqrt(n_samples) ~ 1/44.7 ~ 0.022
        # Utiliser une tolerance généreuse de 4 erreurs standards.
        std_error = sigma / np.sqrt(n_samples)
        npt.assert_allclose(
            mu_mean,
            0.0,
            atol=4.0 * std_error,
            err_msg="La moyenne d'ensemble du potentiel aléatoire n'est pas approximativement nulle",
        )

    def test_approximate_covariance(self, x, sigma, x_c, N):
        """La covariance échantillonnée au décalage 0 et a quelques décalages non nuls devrait
        approximer sigma^2 * exp(-(décalage*dx)^2 / x_c^2).

        La synthèse spectrale utilise ifft (qui inclut 1/N), donc la covariance
        ponctuelle est sigma^2 * exp(...) sans facteur N supplémentaire.
        """
        n_samples = 5000
        rng = np.random.default_rng(456)
        dx = x[1] - x[0]

        # Accumuler la covariance échantillonnée à des décalages sélectionnés
        lags = [0, 1, 5, 10]
        cov_accum = {lag: 0.0 for lag in lags}

        for _ in range(n_samples):
            mu = generate_random_potential(x, sigma, x_c, rng)
            for lag in lags:
                cov_accum[lag] += np.mean(mu[: N - lag] * mu[lag:])

        for lag in lags:
            cov_sample = cov_accum[lag] / n_samples
            dist = lag * dx
            cov_target = sigma**2 * np.exp(-dist**2 / x_c**2)

            # Tolerance généreuse car c'est une estimation Monte Carlo.
            npt.assert_allclose(
                cov_sample,
                cov_target,
                rtol=0.15,
                err_msg=f"La covariance échantillonnée au décalage {lag} s'écarte de la cible",
            )

    def test_different_seeds_give_different_realisations(self, x, sigma, x_c):
        """Deux generateurs avec des graines différentes doivent produire des champs différents."""
        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(20)
        mu1 = generate_random_potential(x, sigma, x_c, rng1)
        mu2 = generate_random_potential(x, sigma, x_c, rng2)
        assert not np.allclose(mu1, mu2), (
            "Des graines différentes ont produit des potentiels aléatoires identiques"
        )


class TestGenerateMediumSlices:
    """Tests pour generate_medium_slices."""

    def test_output_shape(self, x, sigma, x_c, z_c, L, rng):
        """La forme devrait être (n_slices, N) avec n_slices = L / z_c."""
        mu_slices = generate_medium_slices(x, sigma, x_c, z_c, L, rng)
        expected_n_slices = int(L / z_c)
        assert mu_slices.shape == (expected_n_slices, len(x)), (
            f"Forme attendue ({expected_n_slices}, {len(x)}), obtenue {mu_slices.shape}"
        )

    def test_slices_are_independent(self, x, sigma, x_c, z_c, L, N):
        """La corrélation croisee entre des tranches distinctes devrait être faible
        (elles sont tirées indépendamment).

        Chaque tranche a une variance ponctuelle ~ sigma^2. Le produit croise
        de deux tranches indépendantes a une moyenne 0 et un écart-type ~ sigma^2 / sqrt(N_x)
        ou N_x = len(x). La moyenne sur n_trials reduit davantage cela.
        """
        rng = np.random.default_rng(789)
        n_trials = 500
        cross_accum = 0.0

        for _ in range(n_trials):
            mu_slices = generate_medium_slices(x, sigma, x_c, z_c, L, rng)
            # Corréler la première et la deuxième tranche
            cross_accum += np.mean(mu_slices[0] * mu_slices[1])

        cross_mean = cross_accum / n_trials

        # Sous l'independance, E[mean(mu0 * mu1)] = 0.
        # L'écart-type de l'estimateur varie comme sigma^2 / sqrt(N_x * n_trials).
        # Utiliser une tolerance généreuse.
        pointwise_var = sigma**2
        tolerance = 4.0 * pointwise_var / np.sqrt(N * n_trials)
        npt.assert_allclose(
            cross_mean,
            0.0,
            atol=tolerance,
            err_msg="Les tranches semblent corrélées (non indépendantes)",
        )

    def test_each_slice_is_real(self, x, sigma, x_c, z_c, L, rng):
        """Chaque tranche devrait être à valeurs réelles."""
        mu_slices = generate_medium_slices(x, sigma, x_c, z_c, L, rng)
        npt.assert_allclose(
            np.imag(mu_slices),
            0.0,
            atol=1e-14,
            err_msg="Les tranches du milieu ont des parties imaginaires non nulles",
        )
