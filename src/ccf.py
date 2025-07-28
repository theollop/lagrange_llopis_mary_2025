from tempfile import template
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.sparse import coo_matrix, csr_matrix
import time
from io_spec import get_mask, get_rvdatachallenge_dataset
from doppler import _gamma_numba, extend_wavegrid_numba
import torch
from cuda import get_free_memory
from doppler import shift_spec, get_max_nv


# TODO : revoir tout le code concernant les masques, il doit y avoir un problème de calcul car les masque retournés par les méthode sparses non gauss, ont des valeurs supérieurs à 1...


@njit(nopython=True, parallel=True)
def _count_per_velocity(
    line_pos, v_grid, wavegrid_ext, window_size_velocity, wavegrid_step, begin_wave
):
    """
    Compte le nombre d'entrées dans la matrice creuse pour chaque vitesse de la grille.

    Args:
        line_pos (np.ndarray): Positions des raies dans la grille de longueurs d'onde.
        v_grid (np.ndarray): Grille de vitesses pour laquelle on compte les entrées.
        wavegrid_ext (np.ndarray): Grille de longueurs d'onde étendue.
        window_size_velocity (float): Taille de la fenêtre en espace de vitesse.
        wavegrid_step (float): Pas de la grille de longueurs d'onde.
        begin_wave (float): Première valeur de la grille de longueurs d'onde étendue.

    Returns:
        counts (np.ndarray): Tableau de taille (n_v,) contenant le nombre d'entrées pour chaque vitesse.
    """
    n_v = v_grid.shape[0]
    counts = np.zeros(n_v, np.int64)

    for i in prange(n_v):
        γ_shift = _gamma_numba(v_grid[i])
        γ_window = _gamma_numba(window_size_velocity)
        c = 0
        for j in range(line_pos.shape[0]):
            λ0 = line_pos[j]
            ws = λ0 * γ_shift
            w = λ0 * γ_window - λ0

            start = ws - w / 2
            end = ws + w / 2

            i0 = int(np.ceil((start - begin_wave) / wavegrid_step))
            i1 = int(np.ceil((end - begin_wave) / wavegrid_step) - 1)
            if i0 < 1 or i1 >= wavegrid_ext.shape[0] - 1 or i0 > i1:
                continue

            # chaque raie produit 2 pixels partiels + (i1-i0) entiers
            c += 2 + (i1 - i0)
        counts[i] = c

    return counts


@njit(nopython=True, parallel=True)
def _fill_entries(
    line_pos,
    line_weights,
    v_grid,
    wavegrid_ext,
    window_size_velocity,
    wavegrid_step,
    begin_wave,
    offsets,
    rows,
    cols,
    vals,
):
    """
    Remplit les entrées de la matrice creuse avec les valeurs des masques CCF.
    Args:
        line_pos (np.ndarray): Positions des raies dans la grille de longueurs d'onde.
        line_weights (np.ndarray): Poids des raies.
        v_grid (np.ndarray): Grille de vitesses pour laquelle on remplit les entrées.
        wavegrid_ext (np.ndarray): Grille de longueurs d'onde étendue.
        window_size_velocity (float): Taille de la fenêtre en espace de vitesse.
        wavegrid_step (float): Pas de la grille de longueurs d'onde.
        begin_wave (float): Première valeur de la grille de longueurs d'onde étendue.
        offsets (np.ndarray): Offsets pour chaque vitesse dans la matrice creuse.
        rows (np.ndarray): Tableau pour les indices de lignes dans la matrice creuse.
        cols (np.ndarray): Tableau pour les indices de colonnes dans la matrice creuse.
        vals (np.ndarray): Tableau pour les valeurs dans la matrice creuse.
    Returns:
        None: Remplit les tableaux rows, cols et vals avec les indices et valeurs des masques CCF.
    """
    # ! Attention, wavegrid_ext doit être triée par ordre croissant et de pas constant
    n_v = v_grid.shape[0]
    for i in prange(n_v):
        γ_shift = _gamma_numba(v_grid[i])
        γ_window = _gamma_numba(window_size_velocity)
        idx_base = offsets[i]
        idx = idx_base

        for j in range(line_pos.shape[0]):
            λ0 = line_pos[j]
            ws = λ0 * γ_shift
            w = λ0 * γ_window - λ0

            start = ws - w / 2
            end = ws + w / 2

            i0 = int(np.ceil((start - begin_wave) / wavegrid_step))
            i1 = int(np.ceil((end - begin_wave) / wavegrid_step) - 1)
            if i0 < 1 or i1 >= wavegrid_ext.shape[0] - 1 or i0 > i1:
                continue

            frac0 = abs(wavegrid_ext[i0] - start) / wavegrid_step
            frac1 = abs(wavegrid_ext[i1] - end) / wavegrid_step

            # pixel fractionnaire gauche
            rows[idx] = i
            cols[idx] = i0 - 1
            vals[idx] = line_weights[j] * frac0
            idx += 1

            # pixel fractionnaire droit
            rows[idx] = i
            cols[idx] = i1
            vals[idx] = line_weights[j] * frac1
            idx += 1

            # pixels entiers
            for k in range(i0, i1):
                rows[idx] = i
                cols[idx] = k
                vals[idx] = line_weights[j]
                idx += 1
        # fin de la vitesse i


def build_CCF_masks_sparse(
    line_pos: np.ndarray,
    line_weights: np.ndarray,
    v_grid: np.ndarray,
    wavegrid: np.ndarray,
    window_size_velocity: float,
):
    """
    Construit les masques CCF sous forme de matrice creuse pour toutes les vitesses de la grille.
    Args:
        line_pos (np.ndarray): Positions des raies dans la grille de longueurs d'onde.
        line_weights (np.ndarray): Poids des raies.
        v_grid (np.ndarray): Grille de vitesses pour laquelle on construit les masques CCF.
        wavegrid (np.ndarray): Grille de longueurs d'onde DE PAS CONSTANT sur laquelle les spectres sont définis.
        window_size_velocity (float): Taille de la fenêtre en espace de vitesse.
    Returns:
        CCF (csr_matrix): Matrice creuse contenant les masques CCF pour chaque vitesse de la grille.
    """
    # 1) extension Dopplerk
    wavegrid_ext, wave_before, wave_after = extend_wavegrid_numba(wavegrid, v_grid)
    step = wavegrid[1] - wavegrid[0]
    begin = wavegrid_ext[0]
    n_v = len(v_grid)
    n_w_ext = len(wavegrid_ext)

    # 2) passage 1 : on compte par vitesse
    counts = _count_per_velocity(
        line_pos, v_grid, wavegrid_ext, window_size_velocity, step, begin
    )

    # 3) calcul des offsets
    offsets = np.empty(n_v, np.int64)
    total = 0
    for i in range(n_v):
        offsets[i] = total
        total += counts[i]

    # 4) allocation des tableaux de triplets
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    vals = np.empty(total, dtype=np.float64)

    # 5) passage 2 : remplissage
    _fill_entries(
        line_pos,
        line_weights,
        v_grid,
        wavegrid_ext,
        window_size_velocity,
        step,
        begin,
        offsets,
        rows,
        cols,
        vals,
    )

    # 6) assemblage sparse + recadrage
    coo = coo_matrix((vals, (rows, cols)), shape=(n_v, n_w_ext))
    CCF = coo.tocsr()
    s = len(wave_before)
    e = s + len(wavegrid)
    return CCF[:, s:e]


@njit(nopython=True, parallel=True)
def _count_per_velocity_gauss(
    line_pos,
    line_weights,
    v_grid,
    wavegrid_ext,
    window_size_velocity,
    wavegrid_step,
    begin_wave,
):
    """
    Compte le nombre d'entrées de la matrice creuse pour chaque vitesse en profil gaussien.
    On prend un support ±4σ autour de chaque raie.
    """
    c = 299792458.0
    n_v = v_grid.shape[0]
    counts = np.zeros(n_v, np.int64)
    for i in prange(n_v):
        shift = 1.0 + v_grid[i] / c  # approximation non-relativiste
        total = 0
        for j in range(line_pos.shape[0]):
            lam0 = line_pos[j]
            lam_c = lam0 * shift
            sigma = lam0 * (window_size_velocity / c)
            start = lam_c - 4.0 * sigma
            end = lam_c + 4.0 * sigma
            idx0 = int(np.searchsorted(wavegrid_ext, start))
            idx1 = int(np.searchsorted(wavegrid_ext, end))
            if idx0 < 0:
                idx0 = 0
            if idx1 >= wavegrid_ext.shape[0]:
                idx1 = wavegrid_ext.shape[0] - 1
            if idx1 >= idx0:
                total += idx1 - idx0 + 1
        counts[i] = total
    return counts


@njit(nopython=True, parallel=True)
def _fill_entries_gauss(
    line_pos,
    line_weights,
    v_grid,
    wavegrid_ext,
    window_size_velocity,
    wavegrid_step,
    begin_wave,
    offsets,
    rows,
    cols,
    vals,
):
    """
    Remplit les entrées de la matrice creuse en profil gaussien.
    Chaque raie contribue selon exp(-0.5*((lambda-lam_c)/sigma)^2).
    """
    c = 299792458.0
    n_v = v_grid.shape[0]
    for i in prange(n_v):
        shift = 1.0 + v_grid[i] / c
        idx_base = offsets[i]
        idx = idx_base
        for j in range(line_pos.shape[0]):
            lam0 = line_pos[j]
            weight = line_weights[j]
            lam_c = lam0 * shift
            sigma = lam0 * (window_size_velocity / c)
            start = lam_c - 4.0 * sigma
            end = lam_c + 4.0 * sigma
            idx0 = int(np.searchsorted(wavegrid_ext, start))
            idx1 = int(np.searchsorted(wavegrid_ext, end))
            if idx0 < 0:
                idx0 = 0
            if idx1 >= wavegrid_ext.shape[0]:
                idx1 = wavegrid_ext.shape[0] - 1
            for k in range(idx0, idx1 + 1):
                lam = wavegrid_ext[k]
                x = (lam - lam_c) / sigma
                val = weight * np.exp(-0.5 * x * x)
                rows[idx] = i
                cols[idx] = k
                vals[idx] = val
                idx += 1


def build_CCF_masks_sparse_gauss(
    line_pos: np.ndarray,
    line_weights: np.ndarray,
    v_grid: np.ndarray,
    wavegrid: np.ndarray,
    window_size_velocity: float,
) -> csr_matrix:
    """
    Construit une matrice creuse CSR de masques CCF gaussiens.
    Args:
      - line_pos, line_weights: positions et poids des raies
      - v_grid: grille RV (m/s)
      - wavegrid: grille de longueurs d'onde (constant step)
      - window_size_velocity: sigma du profil gaussien en m/s
    Returns:
      CSR matrix de dimension (len(v_grid), len(wavegrid))
    """
    # extension doppler
    wavegrid_ext, wave_before, wave_after = extend_wavegrid_numba(wavegrid, v_grid)
    step = wavegrid[1] - wavegrid[0]
    begin = wavegrid_ext[0]
    n_v = v_grid.size

    # compter les entrées
    counts = _count_per_velocity_gauss(
        line_pos, line_weights, v_grid, wavegrid_ext, window_size_velocity, step, begin
    )

    # offsets
    offsets = np.empty(n_v, np.int64)
    total = 0
    for i in range(n_v):
        offsets[i] = total
        total += counts[i]

    # allouer triplets
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    vals = np.empty(total, dtype=np.float64)

    # remplir
    _fill_entries_gauss(
        line_pos,
        line_weights,
        v_grid,
        wavegrid_ext,
        window_size_velocity,
        step,
        begin,
        offsets,
        rows,
        cols,
        vals,
    )

    # assembler et recadrer
    coo = coo_matrix((vals, (rows, cols)), shape=(n_v, wavegrid_ext.size))
    CCF = coo.tocsr()
    s = len(wave_before)
    e = s + len(wavegrid)
    return CCF[:, s:e]


# Exemple d'intégration dans compute_CCFs_mask


def compute_CCFs_mask_gauss(
    specs: np.ndarray,
    v_grid: np.ndarray,
    wavegrid: np.ndarray,
    window_size_velocity: float,
    mask_type: str = "gauss",
    custom_mask: np.ndarray = None,
    verbose: bool = False,
):
    # charger masque
    mask = custom_mask if custom_mask is not None else get_mask(mask_type)
    line_pos = mask[:, 0]
    line_weights = mask[:, 1]
    if verbose:
        print("Construction CCF gaussian sparse...")
    CCF_masks = build_CCF_masks_sparse_gauss(
        line_pos, line_weights, v_grid, wavegrid, window_size_velocity
    )
    # calcul CCFs
    CCFs = (CCF_masks.dot((specs - 1.0).T)).T
    CCFs -= np.min(CCFs, axis=1, keepdims=True)
    return CCFs


def compute_CCFs_mask(
    specs: np.ndarray,
    v_grid: np.ndarray,
    wavegrid: np.ndarray,
    window_size_velocity: float,
    mask_type: str = "G2",
    custom_mask_path: str = None,
    verbose: bool = False,
):
    """
    Fonction de haut niveau pour construire les masques CCF.
    Args:
        specs (np.ndarray): Spectres à analyser
        v_grid (np.ndarray): Grille de vitesses pour laquelle on construit les masques CCF.
        wavegrid (np.ndarray): Grille de longueurs d'onde DE PAS CONSTANT sur laquelle les spectres sont définis.
        window_size_velocity (float): Taille de la fenêtre en espace de vitesse.
        mask_type (str): Type de masque à utiliser ("G2", "HARPN_Kitcat", "ESPRESSO_F9", "custom")
        custom_mask_path (str): Chemin vers un masque personnalisé (si mask_type="custom")
        verbose (bool): Affichage détaillé
    Returns:
        CCFs (np.ndarray): CCFs calculées pour tous les spectres
    """
    if verbose:
        print(f"Chargement du masque de raies ({mask_type})...")

    mask = get_mask(mask_type=mask_type, custom_path=custom_mask_path)
    line_pos = mask[:, 0]
    line_weights = mask[:, 1]

    if verbose:
        print(
            f"Masque chargé: {len(line_pos)} raies, plage {line_pos.min():.1f}-{line_pos.max():.1f} Å"
        )

    if verbose:
        print("Construction des masques CCF...")
    start = time.time()
    CCF_masks = build_CCF_masks_sparse(
        line_pos=line_pos,
        line_weights=line_weights,
        v_grid=v_grid,
        wavegrid=wavegrid,
        window_size_velocity=window_size_velocity,
    )
    end = time.time()
    if verbose:
        print(
            f"Temps de calcul pour build_CCF_masks_sparse: {end - start:.2f} secondes"
        )

        print(f"Forme du tableau CCF: {CCF_masks.shape}")

        print("Calcul des CCFs sur la grille")

    start = time.time()
    tmp = CCF_masks.dot(specs.T)  # shape (n_v, n_specs)
    CCFs = tmp.T  # shape (n_specs, n_v)
    end = time.time()
    if verbose:
        print(f"Temps de calcul pour les CCFs: {end - start:.2f} secondes")
    return CCFs


def compute_CCFs_template(
    specs: np.ndarray,
    template: np.ndarray,
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge les spectres, calcule la mémoire GPU disponible, découpe la grille de vitesses en batches,
    calcule la fonction de corrélation croisée (CCF) et renvoie les vitesses et la CCF.

    Args:
        v_min: Vitesse minimale (m/s) pour la grille.
        v_max: Vitesse maximale (m/s) pour la grille.
        v_step: Pas de la grille de vitesses (m/s).
        idx_start: Index du premier spectre à charger.
        idx_end: Index du dernier spectre à charger.
        safety_factor: Fraction de la mémoire GPU à utiliser (0 < facteur ≤ 1).
        dtype: Type PyTorch pour les calculs (float32 ou float64).

    Returns:
        v_grid: Tableau des vitesses (m/s).
        ccf: Tableau de la fonction de corrélation croisée pour chaque vitesse.
    """

    # Libération de la mémoire GPU
    torch.cuda.empty_cache()

    CCFs = []

    for k, spec in enumerate(specs):
        if verbose:
            print(
                f"Calcul de la CCF pour le spectre {k + 1}/{len(specs)} avec {len(v_grid)} vitesses"
            )
        spec_for_ccf = torch.as_tensor(spec, dtype=dtype, device="cuda")

        # Calcul de la mémoire GPU utilisable
        safety_factor = 0.9
        free_mem = get_free_memory() * safety_factor
        L = wavegrid.shape[0]

        # Taille maximale de batch
        nv_max = get_max_nv(L, free_mem, dtype)
        if verbose:
            print(f"Taille max de vitesses par batch : {nv_max}")

        # Découpage en batches
        batches = [v_grid[i : i + nv_max] for i in range(0, len(v_grid), nv_max)]

        # Calcul de la CCF
        ccf_values = []
        t0 = time.time()
        for batch in batches:
            shifted = shift_spec(template, wavegrid, batch, dtype)
            ccf_batch = torch.sum(shifted * spec_for_ccf.unsqueeze(0), dim=1)
            ccf_values.append(ccf_batch.cpu().numpy())
            del shifted, ccf_batch

        # Concaténation et mesure du temps
        ccf = np.concatenate(ccf_values)
        dt = time.time() - t0
        if verbose:
            print(f"Temps total CCF : {dt:.2f} s pour {len(v_grid)} vitesses")

        CCFs.append(ccf)

    # Conversion en tableau numpy
    CCFs = np.array(CCFs)

    return CCFs


def compute_CCFs_template_optimized(
    specs: np.ndarray,
    template: np.ndarray,
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    batch_size_specs: int = None,
) -> np.ndarray:
    """
    Version optimisée : pré-calcule les templates décalés une seule fois,
    puis traite les spectres par batches avec calculs vectorisés.

    OPTIMISATIONS CLÉS :
    1. Pré-calcul des templates décalés (évite la recomputation)
    2. Traitement par batches de spectres
    3. Utilisation de torch.mm pour les produits matriciels
    4. Gestion optimale de la mémoire GPU
    """
    # Libération de la mémoire GPU
    torch.cuda.empty_cache()

    n_specs = specs.shape[0]
    L = wavegrid.shape[0]
    n_v = len(v_grid)

    # Calcul de la mémoire GPU utilisable
    safety_factor = 0.85
    free_mem = get_free_memory() * safety_factor

    if verbose:
        print(f"Mémoire GPU libre : {free_mem / 1e9:.2f} GB")
        print(f"Traitement de {n_specs} spectres avec {n_v} vitesses")

    # Détermination de la taille de batch pour les vitesses
    nv_max = get_max_nv(L, free_mem, dtype)
    if verbose:
        print(f"Taille max de vitesses par batch : {nv_max}")

    # Détermination de la taille de batch pour les spectres
    if batch_size_specs is None:
        # Estimation basée sur la mémoire disponible
        bytes_per_element = 4 if dtype == torch.float32 else 8
        memory_per_spec = L * bytes_per_element
        memory_per_vbatch = nv_max * L * bytes_per_element  # pour les templates décalés
        remaining_memory = free_mem - memory_per_vbatch
        batch_size_specs = max(
            1, int(remaining_memory // memory_per_spec // 2)
        )  # facteur 2 de sécurité

    batch_size_specs = min(batch_size_specs, n_specs)

    if verbose:
        print(f"Taille de batch pour les spectres : {batch_size_specs}")

    # Découpage en batches pour les vitesses
    v_batches = [v_grid[i : i + nv_max] for i in range(0, len(v_grid), nv_max)]

    # Découpage en batches pour les spectres
    spec_batches = [
        specs[i : i + batch_size_specs] for i in range(0, n_specs, batch_size_specs)
    ]

    CCFs = np.zeros(
        (n_specs, n_v), dtype=np.float32 if dtype == torch.float32 else np.float64
    )

    total_start = time.time()

    # Boucle sur les batches de vitesses
    for v_idx, v_batch in enumerate(v_batches):
        if verbose:
            print(
                f"Traitement du batch de vitesses {v_idx + 1}/{len(v_batches)} ({len(v_batch)} vitesses)"
            )

        # Pré-calcul des templates décalés pour ce batch de vitesses
        v_start_time = time.time()
        shifted_templates = shift_spec(
            template, wavegrid, v_batch, dtype
        )  # (nv_batch, L)
        template_time = time.time() - v_start_time

        if verbose:
            print(f"  Temps de calcul des templates décalés : {template_time:.3f}s")

        # Boucle sur les batches de spectres
        for s_idx, spec_batch in enumerate(spec_batches):
            if verbose and len(spec_batches) > 1:
                print(
                    f"    Batch de spectres {s_idx + 1}/{len(spec_batches)} ({len(spec_batch)} spectres)"
                )

            # Transfert des spectres vers GPU
            specs_gpu = torch.as_tensor(
                spec_batch, dtype=dtype, device="cuda"
            )  # (n_batch, L)

            # Calcul vectorisé de la CCF: (nv_batch, L) @ (n_batch, L).T = (nv_batch, n_batch)
            ccf_batch = torch.mm(shifted_templates, specs_gpu.T)  # (nv_batch, n_batch)

            # Stockage des résultats
            v_start = v_idx * nv_max
            v_end = v_start + len(v_batch)
            s_start = s_idx * batch_size_specs
            s_end = s_start + len(spec_batch)

            CCFs[s_start:s_end, v_start:v_end] = ccf_batch.T.cpu().numpy()

            # Nettoyage
            del specs_gpu, ccf_batch

        # Nettoyage des templates décalés
        del shifted_templates
        torch.cuda.empty_cache()

    total_time = time.time() - total_start
    if verbose:
        print(
            f"Temps total optimisé : {total_time:.2f}s pour {n_specs} spectres et {n_v} vitesses"
        )
        speedup_estimate = (n_specs * n_v * L) / (total_time * 1e9)  # GFLOPS estimate
        print(f"Performance estimée : {speedup_estimate:.1f} GFLOPS")

    return CCFs


def normalize_CCFs(CCFs: np.ndarray) -> np.ndarray:
    """
    Normalise les CCFs en divisant par le max / min de chaque CCF.

    Args:
        CCFs (np.ndarray): Tableau des CCFs à normaliser.

    Returns:
        np.ndarray: Tableau des CCFs normalisés.
    """

    return CCFs / np.max(np.abs(CCFs), axis=1, keepdims=True)


def compute_CCFs_masked_template(
    specs: np.ndarray,
    template: np.ndarray,
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
    mask_type: str = "G2",
    window_size_velocity: float = 410.0,
    custom_mask: np.ndarray = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Calcule les CCFs en utilisant un masque de raies qui masque un template, puis on shift le template sur la grille de vitesses afin de calculer les CCFs.

    Args:
        specs (np.ndarray): Spectres à analyser.
        template (np.ndarray): Template de référence.
        wavegrid (np.ndarray): Grille de longueurs d'onde.
        v_grid (np.ndarray): Grille de vitesses.
        mask_type (str): Type de masque à utiliser ("G2", "HARPN_Kitcat", "ESPRESSO_F9", "custom").
        window_size_velocity (float): Taille de la fenêtre en espace de vitesse.
        custom_mask (np.ndarray): Masque personnalisé (si mask_type="custom").
        verbose (bool): Affichage détaillé.

    Returns:
        np.ndarray: CCFs calculées pour tous les spectres.
    """
    # Première étape, calculer le masque pour sélectionner les raies du template
    mask = get_mask(mask_type=mask_type)
    line_pos = mask[:, 0]
    line_weights = mask[:, 1]

    template_mask = build_CCF_masks_sparse_gauss(
        line_pos=line_pos,
        line_weights=line_weights,
        v_grid=np.array([0]),  # On ne calcule pas pour v_grid ici
        wavegrid=wavegrid,
        window_size_velocity=window_size_velocity,
    )
    template_mask /= template_mask.max()  # Normalisation du masque

    masked_template = template_mask.multiply(template)

    masked_template = masked_template.toarray()[0]

    CCFs = compute_CCFs_template_optimized(
        specs=specs,
        template=masked_template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        verbose=verbose,
    )

    return CCFs

    # # Construction du template décalé
    # shifted_templates = build_shifted_templates(
    #     template=template,
    #     v_grid=v_grid,
    #     line_pos=line_pos,
    #     line_weights=line_weights,
    #     window_size_velocity=window_size_velocity,
    # )

    # # Calcul des CCFs
    # CCFs = compute_CCFs(
    #     specs=specs,
    #     shifted_templates=shifted_templates,
    #     wavegrid=wavegrid,
    #     v_grid=v_grid,
    #     verbose=verbose,
    # )

    # return CCFs


# Point d'entrée du script: appel de la fonction compute_ccf et tracé
if __name__ == "__main__":
    torch.cuda.empty_cache()  # Nettoyage de la mémoire GPU

    # Hyperparamètres
    verbose: bool = True
    wavemin: float = 4500
    wavemax: float = 6000
    n_specs = 1000
    planetary_signal_amplitudes: list = None
    planetary_signal_periods: list = None
    verbose: bool = False
    noise_level: float = 0
    v_grid_max: int = 20000
    v_grid_step: int = 100
    v_grid: np.ndarray = np.arange(-v_grid_max, v_grid_max, v_grid_step)
    window_size_velocity: int = 410  # Taille de la fenêtre en espace de vitesse en m/s
    fit_window_size: int = 10000  # Taille de la fenêtre pour le fit en m/s

    dataset, wavegrid, template, jdb = get_rvdatachallenge_dataset(
        n_specs=10,
        wavemin=wavemin,
        wavemax=wavemax,
        planetary_signal_amplitudes=None,
        planetary_signal_periods=None,
        verbose=verbose,
        noise_level=0,
    )

    template = template / template.max()  # Normalisation du template

    CCFs = compute_CCFs_masked_template(
        specs=dataset,
        template=template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        mask_type="G2",
        window_size_velocity=410,
        verbose=verbose,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(v_grid, CCFs[0], label="CCF for first spectrum")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("CCF Value")
    plt.title("Cross-Correlation Function (CCF)")
    plt.legend()
    plt.grid()
    plt.show()
