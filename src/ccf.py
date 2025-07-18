import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit
import time


# * Charger un masque de raies / poids associés à partir d'un fichier texte
def get_G2_mask(filepath: str) -> np.ndarray:
    """
    Lis le fichier G2.txt composé de deux colonnes de données :
    - La première colonne représente les positions des raies en Angström.
    - La deuxième colonne représente le poids des raies.
    Retourne un numpy.ndarray composé des deux colonnes.

    Args:
        filepath (str): Chemin vers le fichier texte contenant les données des raies.

    Returns:
        np.ndarray: Un tableau numpy de forme (N_raies, 2) où N_raies est le nombre de raies,
                    la première colonne contient les positions des raies et
                    la deuxième colonne contient les poids des raies.
    """

    data = np.loadtxt(filepath)
    return data


# * Charge un spectre d'échantillon et une grille de longueurs d'onde
def get_sample_spec():
    """
    Charge un spectre d'échantillon et une grille de longueurs d'onde à
    partir de fichiers .npy de taille (L,) avec L le nombre de points et de dtype float64.
    Le spectre est normalisé pour que sa valeur maximale soit 1.

    Args:
        None

    Returns:
        spec (np.ndarray): Le spectre d'échantillon normalisé, de forme (L,).
        wavegrid (np.ndarray): La grille de longueurs d'onde associée au spectre, de forme (L,).

    """

    spec = np.load("../data/soapgpu/paper_dataset/spec.npy")
    spec = spec / spec.max()
    wavegrid = np.load("../data/soapgpu/paper_dataset/wavegrid.npy")
    return spec, wavegrid


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


spec, wavegrid = get_sample_spec()

# Exemple d'utilisation de la fonction plot_specs
plot_specs(
    (spec),
    wavegrid=wavegrid,
    labels=("Spectre d'échantillon"),
    title="Spectre d'échantillon",
)
plot_specs(
    (spec),
    wavegrid=wavegrid,
    labels=("Spectre d'échantillon Zoom"),
    title="Spectre d'échantillon Zoom",
    xlim=(5000, 5012),
)

# Exemple d'utilisation de la fonction plot_lines
mask = get_G2_mask("../data/masks/G2_mask.txt")
plot_lines(
    spec,
    wavegrid=wavegrid,
    title="Spectre d'échantillon avec raies",
    mask=mask,
    window_size=1,
    n_raies=3,
)


# * Calcule le facteur de Lorentz pour une vitesse donnée
@njit(inline="always")
def _gamma(v):
    return np.float64(np.sqrt((1 + v / 299792458) / (1 - v / 299792458)))


@njit(inline="always")
def extend_wavegrid(wavegrid: np.ndarray, v_grid: np.ndarray):
    """
    Étend la grille de longueurs d'onde pour tenir compte des décalages Doppler.
    Cette fonction calcule les décalages maximaux pour étendre la grille de longues d'onde
    en fonction de la grille de vitesses v_grid, et crée une grille de longueurs d'onde étendue.
    Args:
        wavegrid (np.ndarray): Grille de longueurs d'onde DE PAS CONSTANT sur laquelle les spectres sont définis.
        v_grid (np.ndarray): Grille de vitesses pour laquelle la grille de longueurs d'onde est étendue.
    Returns:
        wavegrid_extended (np.ndarray): Grille de longueurs d'onde étendue pour
            tenir compte des décalages Doppler, de forme (L_extended,).
    """
    # ! Attention, wavegrid doit être triée par ordre croissant et de pas constant

    wavegrid_step = wavegrid[1] - wavegrid[0]

    velocity_grid_max = v_grid.max() + 100  # Sécurité
    velocity_grid_min = v_grid.min() - 100  # Sécurité

    # Calcul des décalages maximaux pour étendre la grille de longueurs d'onde
    max_shift_gamma = _gamma(velocity_grid_max)
    min_shift_gamma = _gamma(-velocity_grid_min)

    # Calcul des longueurs d'onde maximales et minimales possibles après shift maximal / minimal
    max_possible_wavelength = (wavegrid * max_shift_gamma).max()
    min_possible_wavelength = (wavegrid * min_shift_gamma).min()

    # Extension avant la grille
    wave_before = np.arange(min_possible_wavelength, wavegrid.min(), wavegrid_step)
    # Extension après la grille
    wave_after = np.arange(
        wavegrid.max() + wavegrid_step, max_possible_wavelength, wavegrid_step
    )

    # Concaténation compatible avec Numba
    total_length = len(wave_before) + len(wavegrid) + len(wave_after)
    wavegrid_extended = np.empty(total_length)

    # Copier les trois parties
    wavegrid_extended[: len(wave_before)] = wave_before
    wavegrid_extended[len(wave_before) : len(wave_before) + len(wavegrid)] = wavegrid
    wavegrid_extended[len(wave_before) + len(wavegrid) :] = wave_after

    return wavegrid_extended, wave_before, wave_after


# * Calcule le masque CCF avec gestion des pixels fractionnaires pour toutes les vitesses de la grille
@jit(nopython=True)
def build_CCF_masks(
    line_pos: np.ndarray,
    line_weights: np.ndarray,
    v_grid: np.ndarray,
    wavegrid: np.ndarray,
    window_size_velocity: float,
):
    """
    Calcule le masque pour toutes les vitesses de la grille.
    Cette fonction calcule le masque pour chaque vitesse de la grille en utilisant les positions et poids des raies,
    et étend la grille de longueurs d'onde pour tenir compte des décalages Doppler.

    Args:
        line_pos (np.ndarray): Positions des raies spectrales.
        line_weights (np.ndarray): Poids associés à chaque raie spectrale.
        v_grid (np.ndarray): Grille de vitesses pour laquelle le masque est calculé.
        wavegrid (np.ndarray): Grille de longueurs d'onde DE PAS CONSTANT sur laquelle les spectres sont définis.
        window_size_velocity (float): Taille de la fenêtre en espace de vitesse.

    Returns:
        np.ndarray: Masque CCF de forme (len(v_grid), len(wavegrid)),
                    où wavegrid est la grille de longueurs d'onde étendue pour tenir compte des décalages Doppler.
    """
    wavegrid_extended, wave_before, wave_after = extend_wavegrid(wavegrid, v_grid)

    wavegrid_step = wavegrid[1] - wavegrid[0]

    begin_wave = wavegrid_extended[0]

    CCF_masks = np.zeros((len(v_grid), len(wavegrid_extended)))

    # Pour chaque vitesse de la grille, on calcule le masque CCF associé
    for i, v_shift in enumerate(v_grid):
        # Facteur de Lorentz pour le décalage Doppler de la vitesse v_shift sur la grille v_grid
        gamma_shift = _gamma(v_shift)

        # Facteur de Lorentz pour la taille de la fenêtre en espace de vitesse, chaque raie a une taille de fenêtre différente
        # qui dépend de sa longueur d'onde
        gamma_window = _gamma(window_size_velocity)

        # Pour chaque raie on doit calculer la position de la raie dans la grille étendue, en effet, les positions
        # des raies ne tombent pas forcément sur les points de la grille étendue.
        for j in range(len(line_pos)):
            line_shifted = line_pos[j] * gamma_shift
            window_size = line_pos[j] * gamma_window - line_pos[j]

            window_start = line_shifted - window_size / 2
            window_end = line_shifted + window_size / 2

            start_idx = int(
                np.ceil((window_start - begin_wave) / wavegrid_step)
            )  # donne l'index du début de la fenêtre SUR LA GRILLE ETENDUE (celle-ci commence généralement avant le début de la grille)
            end_idx = int(
                np.ceil((window_end - begin_wave) / wavegrid_step) - 1
            )  # donne l'index de la fin de la fenêtre SUR LA GRILLE ETENDUE (celle-ci finit généralement après la fin de la grille)

            if start_idx < 1 or end_idx >= len(wavegrid_extended) - 1:
                continue

            if start_idx > end_idx:
                continue

            # On calcule les fractions de pixels des fenêtres autour de chaque raies qui sortent de la grille étendue
            fraction_before = (
                abs(wavegrid_extended[start_idx] - window_start) / wavegrid_step
            )
            fraction_after = (
                abs(wavegrid_extended[end_idx] - window_end) / wavegrid_step
            )

            CCF_masks[i, start_idx - 1] += line_weights[j] * fraction_before
            CCF_masks[i, end_idx] += line_weights[j] * fraction_after

            for k in range(start_idx, end_idx):
                CCF_masks[i, k] += line_weights[j]

    # On crop le mask pour ne garder que la partie utile (wave)
    start_crop = len(wave_before)
    end_crop = len(wave_before) + len(wavegrid)
    CCF_masks = CCF_masks[:, start_crop:end_crop]

    return CCF_masks


# Utilisation de la fonction build_CCF_masks
v_grid = np.linspace(-20000, 20000, 250)  # Exemple de grille de vitesses
window_size_velocity = 820  # Exemple de taille de fenêtre en espace de vitesse

start = time.time()
CCF_masks = build_CCF_masks(
    line_pos=mask[:, 0],
    line_weights=mask[:, 1],
    v_grid=v_grid,
    wavegrid=wavegrid,
    window_size_velocity=window_size_velocity,
)
end = time.time()
print(f"Temps de calcul du masque CCF : {end - start:.2f} secondes")
