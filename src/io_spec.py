import numpy as np
import h5py
import torch
from doppler import shift_spec, get_max_nv
from cuda import get_free_memory


# * Charger un masque de raies / poids associés à partir d'un fichier texte
def get_G2_mask(
    filepath: str = "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/masks/G2_mask.txt",
) -> np.ndarray:
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

    spec = np.load(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/spec.npy"
    )
    spec = spec / spec.max()
    wavegrid = np.load(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/wavegrid.npy"
    )
    return spec, wavegrid


# * Charge un dataset de spectres à partir d'un fichier HDF5
def get_specs_from_h5(
    filepath: str = "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/spec_cube_tot.h5",
    idx_spec_start: int = 0,
    idx_spec_end: int = 99,
    wavemin: float = 5000,
    wavemax: float = 5050,
) -> np.ndarray:
    """
    Charge un spectre d'échantillon à partir d'un fichier HDF5.
    Le fichier doit contenir un groupe 'spec' avec un tableau de données.

    Args:
        filepath (str): Chemin vers le fichier HDF5 contenant les données des spectres.

    Returns:
        np.ndarray: Un tableau numpy de forme (L,) où L est le nombre de points du spectre.
    """

    wavegrid = np.load(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/wavegrid.npy"
    )
    if wavemin is None:
        wavemin = wavegrid.min()
    if wavemax is None:
        wavemax = wavegrid.max()

    wavegrid_mask = (wavegrid >= wavemin) & (wavegrid <= wavemax)
    wavegrid = wavegrid[wavegrid_mask]

    with h5py.File(filepath, "r") as f:
        specs = f["spec_cube"][idx_spec_start : idx_spec_end + 1, wavegrid_mask]

    return specs, wavegrid


# * Génère un dataset de spectres à partir d'un modèle
def get_dataset_from_template(
    template: np.ndarray,
    wavegrid: np.ndarray,
    velocities: np.ndarray = None,
    noise_level: float = 0.01,
    wavemin: float = 5000,
    wavemax: float = 5050,
    dtype=torch.float32,
    verbose: bool = True,
) -> np.ndarray:
    """
    Génère un dataset de spectres à partir d'un modèle de template.
    Le modèle est un tableau numpy de forme (L,) où L est le nombre de points du spectre.
    Le dataset est généré en ajoutant du bruit gaussien au modèle.

    Args:
        template (np.ndarray): Le modèle de template à utiliser pour générer les spectres.
        n_specs (int): Le nombre de spectres à générer.
        noise_level (float): Le niveau de bruit à ajouter au modèle.

    Returns:
        np.ndarray: Un tableau numpy de forme (n_specs, L) où L est le nombre de points du spectre.
    """

    torch.cuda.empty_cache()  # Nettoyage de la mémoire GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Génération du dataset de spectres à partir du modèle... (sur {device})")

    if wavemin is None:
        wavemin = wavegrid.min()
    if wavemax is None:
        wavemax = wavegrid.max()

    wavegrid_mask = (wavegrid >= wavemin) & (wavegrid <= wavemax)
    wavegrid = wavegrid[wavegrid_mask]

    template_torch = torch.tensor(template[wavegrid_mask], dtype=dtype, device=device)

    if velocities is None:
        velocities = np.linspace(-100, 100, 100)  # Grille de vitesses en km/s

    nv_max = get_max_nv(
        L=wavegrid.shape[0],
        free_memory_bytes=get_free_memory()
        * 0.9,  # Utilisation de 90% de la mémoire GPU libre
        dtype=dtype,
    )  # Calcul du maximum de velocities pour le redshift

    # Découpage en batches
    batches = [velocities[i : i + nv_max] for i in range(0, len(velocities), nv_max)]

    # Calcul de la CCF
    dataset = []
    for batch in batches:
        shifted = shift_spec(template_torch, wavegrid, batch, dtype)
        if verbose:
            print(f"Batch traité, taille : {len(batch)}")

        dataset.append(shifted.cpu().numpy())

        del shifted

    dataset = np.concatenate(dataset, axis=0)

    dataset += np.random.normal(
        loc=0.0, scale=noise_level, size=dataset.shape
    )  # Ajout du bruit gaussien

    return dataset, wavegrid


if __name__ == "__main__":
    # Exemple d'utilisation
    template = np.load(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/spec.npy"
    )
    template = template / template.max()  # Normalisation du modèle

    wavegrid = np.load(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/wavegrid.npy"
    )

    Kp = 10
    P = 10
    N_specs = 1000
    t = np.arange(0, N_specs)
    velocities = Kp * np.sin(2 * np.pi * t / P)  # Génération des vitesses

    velocities = torch.tensor(velocities, dtype=torch.float32)

    dataset, wavegrid = get_dataset_from_template(
        template,
        wavegrid,
        velocities=velocities,
        noise_level=0,
        wavemax=None,
        wavemin=None,
        verbose=True,
    )
    print(f"Dataset généré de forme: {dataset.shape}")
    print(f"Premier spectre: {dataset[0]}")
    print(
        f"Grille de longueurs d'onde: {np.load('/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/wavegrid.npy')}"
    )
    print(f"Masque de raies: {get_G2_mask()}")
    print(f"Spectre d'échantillon: {get_sample_spec()[0]}")

    # Plot
    from plotting import plot_specs

    plot_specs(
        (dataset[0]),
        wavegrid=wavegrid,
        title="Dataset de spectres généré",
        xlim=(5000, 5012),
    )
