"""
Fonctions utilitaires pour le trace de graphiques.

Auteur : Mouhssine Rifaki
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_intensity(x, phi_num, phi_theo=None, title="", ylabel=r"$|\phi|^2$"):
    """Trace |phi|^2 (numérique et optionnellement théorique)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, np.abs(phi_num) ** 2, "b-", lw=1.5, label="Numérique")
    if phi_theo is not None:
        ax.plot(x, np.abs(phi_theo) ** 2, "r--", lw=1.5, label="Théorique")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
