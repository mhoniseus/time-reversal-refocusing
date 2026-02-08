"""Refocalisation par retournement temporel dans les milieux homogènes et aléatoires (modèle paraxial)."""

from .paraxial_solver import fourier_propagate, split_step_propagate
from .random_medium import generate_random_potential, generate_medium_slices
from .mirrors import compact_mirror, gaussian_mirror
from .time_reversal import (
    time_reverse_field,
    refocus_homogeneous,
    refocus_random_same_medium,
    refocus_random_homogeneous_back,
)
from .theory import (
    phi_t_homogeneous,
    intensity_t_homogeneous,
    refocused_gaussian_mirror_homogeneous,
    mean_phi_t_random,
    mean_refocused_same_medium,
    mean_refocused_homogeneous_back,
)
