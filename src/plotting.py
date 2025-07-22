import numpy as np
import matplotlib.pyplot as plt


# * Affiche un ou plusieurs spectres d'échantillon
def plot_specs(
    *specs, wavegrid: np.ndarray, labels=None, title: str = "Spectre", xlim=None
):
    """
    Affiche un ou plusieurs spectres d'échantillon, avec option de zoom sur une plage de longueurs d'onde.

    Args:
        *specs: Un ou plusieurs spectres à afficher, sous forme de tableaux numpy.ndarray de forme (L,) où L est le nombre de points.
        wavegrid (np.ndarray): La grille de longueurs d'onde associée aux spectres, de forme (L,).
        labels (list, optional): Liste de labels pour chaque spectre. Si None, aucun label n'est affiché.
        title (str): Le titre du graphique.
        xlim (tuple, optional): Plage de longueurs d'onde à afficher,
            sous la forme (min_wave, max_wave). Si None, la plage complète est affichée.
    Returns:
        None: Affiche le graphique des spectres.
    """
    plt.figure(figsize=(10, 5))
    for i, spec in enumerate(specs):
        lbl = labels[i] if labels and i < len(labels) else None
        plt.plot(wavegrid, spec, label=lbl)
    plt.title(title)
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Intensité normalisée")
    plt.grid()
    if xlim:
        plt.xlim(xlim)
    if labels:
        plt.legend()
    plt.show()


# * Affiche les n_raies raies à plus forte activité
def plot_lines(
    spec: np.ndarray,
    wavegrid: np.ndarray,
    title: str = "Spectre",
    mask: np.ndarray = None,
    window_size: float = 5,
    n_raies: int = 3,
):
    """
    Affiche les n_raies raies à plus forte activité en chargeant un masque de raies
    sous la forme d'un tableau numpy.ndarray de forme (N_raies, 2) où
    - La première colonne représente les positions des raies en Angström.
    - La deuxième colonne représente le poids des raies.

    Args:
        spec (np.ndarray): Le spectre à afficher.
        wavegrid (np.ndarray): La grille de longueurs d'onde.
        title (str): Le titre du graphique.
        mask (np.ndarray, optional): Le masque des raies à afficher. Si None, aucune raie n'est affichée.
        window_size (float): La taille de la fenêtre autour de chaque raie pour l'affichage.
        n_raies (int): Le nombre de raies les plus fortes à afficher.
    Returns:
        None: Affiche le graphique des raies.
    """
    # Filtre les raies contenues dans wavegrid
    wavegrid_min = wavegrid.min()
    wavegrid_max = wavegrid.max()
    if mask is not None:
        mask = mask[(mask[:, 0] >= wavegrid_min) & (mask[:, 0] <= wavegrid_max)]
    strong_lines_indices = np.argsort(mask[:, 1])[
        -n_raies:
    ]  # Les n_raies raies les plus fortes
    # Utilisation de subplot pour diviser l'affichage
    fig, axs = plt.subplots(n_raies, 1, figsize=(10, 5 * n_raies))
    fig.suptitle(title)
    for i, idx in enumerate(strong_lines_indices):
        line_position = mask[idx, 0]
        line_weight = mask[idx, 1]
        # Définition de la fenêtre autour de la raie
        window_mask = (wavegrid >= line_position - window_size) & (
            wavegrid <= line_position + window_size
        )
        axs[i].plot(
            wavegrid[window_mask],
            spec[window_mask],
            label=f"Raie à {line_position:.2f} Å",
        )
        axs[i].axvline(
            line_position,
            color="r",
            linestyle="--",
            label=f"Position: {line_position:.2f} Å",
        )
        axs[i].axhline(
            line_weight,
            color="g",
            linestyle="--",
            label=f"Intensité: {line_weight:.2f}",
        )
        axs[i].set_title(f"Raie {i + 1}: {line_position:.2f} Å")
        axs[i].set_xlabel("Longueur d'onde (nm)")
        axs[i].set_ylabel("Intensité normalisée")
        axs[i].grid()
        axs[i].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajuste la mise en page pour le titre
    plt.show()


def plot_ccf(
    ccf: np.ndarray,
    v_grid: np.ndarray,
    title: str = "CCF",
):
    """
    Affiche les CCF pour chaque vitesse de la grille.

    Args:
        ccf (np.ndarray): Le tableau des CCF de forme (n_specs, n_v).
        v_grid (np.ndarray): La grille de vitesses associée aux CCF.
        title (str): Le titre du graphique.
        xlim (tuple, optional): Plage de vitesses à afficher, sous la forme (min_v, max_v).
            Si None, la plage complète est affichée.
    Returns:
        None: Affiche le graphique des CCF.
    """
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Vitesse (km/s)")
    plt.ylabel("CCF")
    plt.grid()
    plt.plot(v_grid, ccf.T)  # Transpose pour avoir les vitesses en abscisse
    plt.legend()
    plt.show()
