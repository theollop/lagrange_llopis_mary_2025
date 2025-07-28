import numpy as np
import h5py
import torch
from doppler import shift_spec, get_max_nv
from cuda import get_free_memory
import pandas as pd


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

    return dataset, wavegrid, template[wavegrid_mask]


def get_rvdatachallenge_dataset(
    n_specs: int = None,
    wavemin: float = 5000,
    wavemax: float = 5050,
    planetary_signal_amplitudes: list = None,
    planetary_signal_periods: list = None,
    verbose: bool = False,
    noise_level: float = 0.01,
):
    filespaths = [
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy",
    ]
    analyse_material = np.load(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
        allow_pickle=True,
    )

    wavegrid = analyse_material["wave"].to_numpy()
    template = analyse_material["stellar_template"].to_numpy()

    analyse_summary = pd.read_csv(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57000_E61000_planet-FallChallenge3/HARPN/STAR1134_HPN_Analyse_summary.csv"
    )

    jdb = analyse_summary["jdb"]
    if n_specs is None or n_specs > len(jdb):
        n_specs = len(jdb)
    if wavemin is None:
        wavemin = wavegrid.min()
    if wavemax is None:
        wavemax = wavegrid.max()
    wave_mask = (wavegrid >= wavemin) & (wavegrid <= wavemax)
    wavegrid = wavegrid[wave_mask]
    template = template[wave_mask]

    specs = []
    for filepath in filespaths:
        data = np.load(filepath)
        data = data[:n_specs, wave_mask]
        # data /= np.max(np.abs(data), axis=1, keepdims=True)
        specs.append(data)

    dataset = np.concatenate(specs, axis=0)
    n_specs = dataset.shape[0]

    if planetary_signal_amplitudes is not None and planetary_signal_periods is not None:
        assert len(planetary_signal_amplitudes) == len(planetary_signal_periods), (
            "Les listes d'amplitudes et de périodes doivent avoir la même longueur."
        )
        shifted_dataset = []
        for i in range(n_specs):
            if verbose:
                print(f"Traitement du spectre {i + 1}/{n_specs}")

            velocity = 0.0
            for amplitude, period in zip(
                planetary_signal_amplitudes, planetary_signal_periods
            ):
                velocity += amplitude * np.sin(2 * np.pi * (jdb[i] / period))

            shifted_spec = shift_spec(
                spec=dataset[i],
                wavegrid=wavegrid,
                velocities=np.array([velocity]),
                dtype=torch.float64,
            )

            shifted_spec = shifted_spec.squeeze(0).cpu().numpy()
            shifted_dataset.append(shifted_spec)

        dataset = np.array(shifted_dataset)

        dataset += np.random.normal(
            loc=0.0, scale=noise_level, size=dataset.shape
        )  # Ajout du bruit gaussien

    return dataset, wavegrid, template, jdb[:n_specs]


def get_mask(mask_type: str = "G2", custom_path: str = None) -> np.ndarray:
    """
    Charge différents types de masques CCF pour l'analyse des vitesses radiales.

    Args:
        mask_type (str): Type de masque à charger:
            - "G2": Masque G2 par défaut
            - "HARPN_Kitcat": Masque Kitcat optimisé pour HARPN
            - "ESPRESSO_F9": Masque ESPRESSO F9
            - "custom": Masque personnalisé (nécessite custom_path)
        custom_path (str): Chemin vers un masque personnalisé (format .txt, .fits, ou .p)

    Returns:
        np.ndarray: Tableau (N_raies, 2) avec positions [Å] et poids des raies
    """
    import os

    if mask_type == "G2":
        return get_G2_mask()

    elif mask_type == "HARPN_Kitcat":
        mask_path = "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/masks/HARPN_Kitcat_mask.txt"
        if not os.path.exists(mask_path):
            # Extraire depuis le fichier pickle si pas encore fait
            import pickle

            pickle_path = "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57000_E61000_planet-FallChallenge3/HARPN/STAR1134_HPN_Kitcat_mask.p"
            with open(pickle_path, "rb") as f:
                kitcat_data = pickle.load(f)
            catalogue = kitcat_data["catalogue"]
            line_positions = catalogue["wave"].values
            line_weights = catalogue["weight_rv"].values
            kitcat_mask = np.column_stack([line_positions, line_weights])
            np.savetxt(mask_path, kitcat_mask, fmt="%.6f %.6f")
            print(f"Masque Kitcat extrait et sauvegardé sous: {mask_path}")
        return np.loadtxt(mask_path)

    elif mask_type == "ESPRESSO_F9":
        mask_path = "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/masks/ESPRESSO_F9_mask.txt"
        if not os.path.exists(mask_path):
            # Extraire depuis le fichier FITS si pas encore fait
            from astropy.io import fits

            fits_path = "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/masks/ESPRESSO_F9.fits"
            with fits.open(fits_path) as hdul:
                data = hdul[1].data
                wavelengths = data["lambda"]
                weights = data["contrast"]
                espresso_mask = np.column_stack([wavelengths, weights])
                np.savetxt(mask_path, espresso_mask, fmt="%.6f %.6f")
                print(f"Masque ESPRESSO F9 extrait et sauvegardé sous: {mask_path}")
        return np.loadtxt(mask_path)

    elif mask_type == "custom":
        if custom_path is None:
            raise ValueError("custom_path doit être spécifié pour mask_type='custom'")

        if custom_path.endswith(".txt"):
            return np.loadtxt(custom_path)
        elif custom_path.endswith(".fits"):
            from astropy.io import fits

            with fits.open(custom_path) as hdul:
                data = hdul[1].data
                # Adapter selon la structure du fichier FITS
                wavelengths = data[data.columns.names[0]]
                weights = data[data.columns.names[1]]
                return np.column_stack([wavelengths, weights])
        elif custom_path.endswith(".p"):
            import pickle

            with open(custom_path, "rb") as f:
                data = pickle.load(f)
            # Adapter selon la structure du fichier pickle
            if isinstance(data, dict) and "catalogue" in data:
                catalogue = data["catalogue"]
                return np.column_stack(
                    [catalogue["wave"].values, catalogue["weight_rv"].values]
                )
            else:
                raise ValueError("Structure de fichier pickle non supportée")
        else:
            raise ValueError(
                "Format de fichier non supporté. Utilisez .txt, .fits ou .p"
            )

    else:
        raise ValueError(f"Type de masque non reconnu: {mask_type}")


if __name__ == "__main__":
    filepath = "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_flux_YVA.npy"
    data = np.load(filepath)
    data = data[:1000, :]

    analyse_material = np.load(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57001_E61001_planet-FallChallenge1/HARPN/STAR1136_HPN_Analyse_material.p",
        allow_pickle=True,
    )

    wavegrid = analyse_material["wave"].to_numpy()
    template = analyse_material["stellar_template"].to_numpy()

    analyse_summary = pd.read_csv(
        "/home/tliopis/Codes/lagrange_llopis_mary_2025/data/rv_datachallenge/Sun_B57000_E61000_planet-FallChallenge3/HARPN/STAR1134_HPN_Analyse_summary.csv"
    )

    # plot du spectre 364 comparé au premier
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(wavegrid, data[0], label="Spectre 0")
    plt.plot(wavegrid, data[335], label="Spectre 334")
    plt.xlabel("Longueur d'onde (Å)")
    plt.ylabel("Flux normalisé")
    plt.title("Comparaison des spectres 0 et 364")
    plt.legend()
    plt.show()
