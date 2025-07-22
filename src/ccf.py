import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.sparse import coo_matrix
import time
from io_spec import get_G2_mask, get_specs_from_h5
from doppler import _gamma_numba, extend_wavegrid_numba
import torch
from cuda import get_free_memory
from doppler import shift_spec, get_max_nv

# * On va maintenant construire les masques CCF pour toutes les vitesses de la grille.
# * Ce tableau possède à la fin une taille de (len(v_grid), len(wavegrid)),
# * Sachant que plus de 90 % du tableau est rempli de zéros, il est plus judicieux de le stocker sous forme de matrice creuse
# * ce qui permet de gagner en mémoire mais complique l'écriture de la fonction.
# * En gros il nous faut une liste de triplets (i, j, w)
# *     i est l'index de la vitesse dans v_grid,
# *     j est l'index de la longueur d'onde dans wavegrid
# *     w est la valeur du masque CCF.
# * Le reste vaudra 0
# * Pour calculer chaque valeur du masque CCF : w, on va utiliser la méthode des pixels fractionnaires.

# * Voici comment on procède :

# * 1) On étend la grille de longueurs d'onde initiale pour tenir compte des décalages Doppler qui sortiraient les positions de raies standardisées
# *    de la grille de longueurs d'onde. On utilise la fonction extend_wavegrid_numba pour cela.

# * 2) On définit une largeur de fenêtre autour de chaque raie en espace de vitesse, non en longueur d'onde pourquoi ? :
# * Certaines raies sont plus shiftées que d'autres (bords du spectre) donc il nous faut une fenêtre plus large pour ces raies.
# * Définir une largeur de fenêtre en espace de vitesse permet de s'assurer que toutes les raies sont prises en compte,
# * même celles qui sont très décalées par rapport à la grille de longueurs d'onde.
# * La taille de la fenêtre en espace de vitesse est définie par l'utilisateur
# * (window_size_velocity) et est appliquée à chaque raie en fonction de sa position.
# * On utilise la fonction _gamma_numba pour calculer le facteur de Lorentz pour la vitesse de la grille et pour la taille de la fenêtre en espace de vitesse.

# * 3) Pour chaque vitesse de la grille ->

# *   Pour chaque raie dans le masque de raies ->

# *     On calcule la position de la raie dans la grille étendue, en effet, les positions
# *     des raies ne tombent pas forcément sur les points de la grille étendue.

# *     On calcule la position de début et de fin de la fenêtre pour cette raie.
# *     On calcule les indices de début et de fin de la fenêtre dans la grille étendue.

# *     On calcule les fractions de pixels des fenêtres autour de chaque raie qui sortent de la grille étendue.

# *     On obtient donc pour les pixels de bords (fractionnaire) un poids : (poids_de_raie) x (% du pixel de la fenêtre qui est dans la grille étendue)
# *     Pour les pixels entiers, on a donc un poids : (poids_de_raie) x (1)

# * 4) On stocke les valeurs dans une matrice creuse (sparse matrix) pour ne pas perdre de mémoire.


# * Ce calcul est assez coûteux en temps de calcul car il implique une double boucle, on a enfait n_raies * n_vitesses itérations.

# * On utilise donc Numba pour accélérer le calcul en compilant la fonction en code machine.
# * Le problème avec Numba c'est qu'il ne supporte pas les matrices creuses ni les listes dynamiques on doit préallouer la taille de la matrice creuse.
# * Donc on doit d'abord compter combien on aura de (i, j, w) à stocker dans la matrice creuse.
# * Pour cela on commence par coder la fonction _count_entries qui va compter le nombre d'entrées dans la matrice creuse.


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


def compute_CCF_mask(
    specs: np.ndarray,
    v_grid: np.ndarray,
    wavegrid: np.ndarray,
    window_size_velocity: float,
):
    """
    Fonction de haut niveau pour construire les masques CCF.
    Args:
        line_pos (np.ndarray): Positions des raies dans la grille de longueurs d'onde.
        line_weights (np.ndarray): Poids des raies.
        v_grid (np.ndarray): Grille de vitesses pour laquelle on construit les masques CCF.
        wavegrid (np.ndarray): Grille de longueurs d'onde DE PAS CONSTANT sur laquelle les spectres sont définis.
        window_size_velocity (float): Taille de la fenêtre en espace de vitesse.
    Returns:
        CCF_masks (csr_matrix): Matrice creuse contenant les masques CCF pour chaque vitesse de la grille.
    """

    print("Chargement du masque de raies...")
    mask = get_G2_mask()
    line_pos = mask[:, 0]
    line_weights = mask[:, 1]

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
    print(f"Temps de calcul pour build_CCF_masks_sparse: {end - start:.2f} secondes")

    print(f"Forme du tableau CCF: {CCF_masks.shape}")

    print("Calcul des CCFs sur la grille")
    start = time.time()
    tmp = CCF_masks.dot(specs.T)  # shape (n_v, n_specs)
    CCFs = tmp.T  # shape (n_specs, n_v)
    end = time.time()
    print(f"Temps de calcul pour les CCFs: {end - start:.2f} secondes")
    return CCFs


# -----------------------------------------------------------------------------
# Exécution principale
# -----------------------------------------------------------------------------
def compute_ccf_template(
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
            if verbose:
                print(f"Batch traité, taille : {len(batch)}")
            del shifted, ccf_batch

        # Concaténation et mesure du temps
        ccf = np.concatenate(ccf_values)
        # Normalisation de toutes les CCFs
        ccf /= np.linalg.norm(ccf)
        dt = time.time() - t0
        print(f"Temps total CCF : {dt:.2f} s pour {len(v_grid)} vitesses")

        CCFs.append(ccf)

    # Conversion en tableau numpy
    CCFs = np.array(CCFs)

    return CCFs


def extract_rv_template(
    specs: np.ndarray,
    template: np.ndarray,
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
    fit_model: str = "gaussian",
    window_size: float = 2000,
    poly_order: int = 2,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    **fit_kwargs,
) -> dict:
    """
    Calcule les CCFs par méthode template puis extrait les vitesses radiales par ajustement.

    Args:
        specs: Spectres à analyser (n_specs, n_wave)
        template: Spectre template de référence (n_wave,)
        wavegrid: Grille de longueurs d'onde (n_wave,)
        v_grid: Grille de vitesses pour le calcul CCF (m/s)
        fit_model: Modèle d'ajustement ('gaussian', 'lorentzian', 'voigt', 'polynomial')
        window_size: Taille de la fenêtre d'ajustement (m/s)
        poly_order: Ordre du polynôme (si fit_model='polynomial')
        dtype: Type PyTorch pour les calculs
        verbose: Affichage détaillé
        **fit_kwargs: Arguments supplémentaires pour l'ajustement

    Returns:
        dict: Dictionnaire contenant les résultats pour chaque spectre
            - 'rv': vitesses radiales (m/s)
            - 'rv_errors': erreurs sur les VR (m/s)
            - 'fit_results': résultats détaillés des ajustements
            - 'ccfs': CCFs calculées
            - 'v_grid': grille de vitesses utilisée
            - 'success': booléens indiquant le succès des ajustements
    """
    print("=== EXTRACTION VR PAR MÉTHODE TEMPLATE ===")
    print(f"Nombre de spectres: {len(specs)}")
    print(f"Modèle d'ajustement: {fit_model}")
    print(f"Fenêtre d'ajustement: {window_size} m/s")

    # Import du fitter
    try:
        from fitter import CCFFitter, extract_rv_from_fit
    except ImportError:
        raise ImportError(
            "Module 'fitter' non trouvé. Assurez-vous que fitter.py est dans le même répertoire."
        )

    # 1. Calcul des CCFs avec la méthode template
    print("\n1. Calcul des CCFs...")
    start_ccf = time.time()
    CCFs = compute_ccf_template(
        specs=specs,
        template=template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        dtype=dtype,
        verbose=verbose,
    )
    end_ccf = time.time()
    print(f"Temps total CCF template: {end_ccf - start_ccf:.2f} s")

    # 2. Initialisation du fitter
    fitter = CCFFitter()

    # 3. Ajustements sur chaque CCF
    print(f"\n2. Ajustements {fit_model} sur {len(CCFs)} CCFs...")
    start_fit = time.time()

    results = {
        "rv": [],
        "rv_errors": [],
        "fit_results": [],
        "ccfs": CCFs,
        "v_grid": v_grid,
        "success": [],
        "r_squared": [],
        "method": "template",
        "fit_model": fit_model,
    }

    for i, ccf in enumerate(CCFs):
        if verbose:
            print(f"  Ajustement spectre {i + 1}/{len(CCFs)}")

        try:
            # Ajustement
            fit_result = fitter.fit_ccf(
                ccf=ccf,
                v_grid=v_grid,
                model=fit_model,
                window_size=window_size,
                poly_order=poly_order,
                **fit_kwargs,
            )

            if fit_result["success"]:
                rv, rv_error = extract_rv_from_fit(fit_result)
                results["rv"].append(rv)
                results["rv_errors"].append(rv_error)
                results["r_squared"].append(fit_result["r_squared"])
                results["success"].append(True)

                if verbose:
                    print(
                        f"    VR = {rv:.2f} ± {rv_error:.2f} m/s (R² = {fit_result['r_squared']:.4f})"
                    )
            else:
                results["rv"].append(np.nan)
                results["rv_errors"].append(np.nan)
                results["r_squared"].append(np.nan)
                results["success"].append(False)

                if verbose:
                    print(
                        f"    Échec ajustement: {fit_result.get('error_message', 'Erreur inconnue')}"
                    )

            results["fit_results"].append(fit_result)

        except Exception as e:
            print(f"    Erreur spectre {i + 1}: {e}")
            results["rv"].append(np.nan)
            results["rv_errors"].append(np.nan)
            results["r_squared"].append(np.nan)
            results["success"].append(False)
            results["fit_results"].append({"success": False, "error_message": str(e)})

    # Conversion en numpy arrays
    results["rv"] = np.array(results["rv"])
    results["rv_errors"] = np.array(results["rv_errors"])
    results["r_squared"] = np.array(results["r_squared"])
    results["success"] = np.array(results["success"])

    end_fit = time.time()
    print(f"Temps total ajustements: {end_fit - start_fit:.2f} s")

    # Statistiques
    n_success = np.sum(results["success"])
    print("\nRésultats:")
    print(f"  Ajustements réussis: {n_success}/{len(specs)}")

    if n_success > 0:
        valid_rvs = results["rv"][results["success"]]
        valid_errors = results["rv_errors"][results["success"]]
        valid_r2 = results["r_squared"][results["success"]]

        print(f"  VR moyenne: {np.mean(valid_rvs):.1f} ± {np.std(valid_rvs):.1f} m/s")
        print(f"  Erreur moyenne: {np.mean(valid_errors):.1f} m/s")
        print(f"  R² moyen: {np.mean(valid_r2):.4f}")

    return results


def extract_rv_masks(
    specs: np.ndarray,
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
    window_size_velocity: float = 820,
    fit_model: str = "gaussian",
    window_size_fit: float = 2000,
    poly_order: int = 2,
    verbose: bool = False,
    **fit_kwargs,
) -> dict:
    """
    Calcule les CCFs par méthode masques de raies puis extrait les vitesses radiales par ajustement.

    Args:
        specs: Spectres à analyser (n_specs, n_wave)
        wavegrid: Grille de longueurs d'onde DE PAS CONSTANT (n_wave,)
        v_grid: Grille de vitesses pour le calcul CCF (m/s)
        window_size_velocity: Taille fenêtre en espace vitesse pour construction masques (m/s)
        fit_model: Modèle d'ajustement ('gaussian', 'lorentzian', 'voigt', 'polynomial')
        window_size_fit: Taille de la fenêtre d'ajustement (m/s)
        poly_order: Ordre du polynôme (si fit_model='polynomial')
        verbose: Affichage détaillé
        **fit_kwargs: Arguments supplémentaires pour l'ajustement

    Returns:
        dict: Dictionnaire contenant les résultats pour chaque spectre
            - 'rv': vitesses radiales (m/s)
            - 'rv_errors': erreurs sur les VR (m/s)
            - 'fit_results': résultats détaillés des ajustements
            - 'ccfs': CCFs calculées
            - 'v_grid': grille de vitesses utilisée
            - 'success': booléens indiquant le succès des ajustements
    """
    print("=== EXTRACTION VR PAR MÉTHODE MASQUES ===")
    print(f"Nombre de spectres: {len(specs)}")
    print(f"Fenêtre vitesse (masques): {window_size_velocity} m/s")
    print(f"Modèle d'ajustement: {fit_model}")
    print(f"Fenêtre d'ajustement: {window_size_fit} m/s")

    # Import du fitter
    try:
        from fitter import CCFFitter, extract_rv_from_fit
    except ImportError:
        raise ImportError(
            "Module 'fitter' non trouvé. Assurez-vous que fitter.py est dans le même répertoire."
        )

    # 1. Calcul des CCFs avec la méthode masques
    print("\n1. Calcul des CCFs avec masques de raies...")
    start_ccf = time.time()
    CCFs = compute_CCF_mask(
        specs=specs,
        v_grid=v_grid,
        wavegrid=wavegrid,
        window_size_velocity=window_size_velocity,
    )
    end_ccf = time.time()
    print(f"Temps total CCF masques: {end_ccf - start_ccf:.2f} s")

    # 2. Initialisation du fitter
    fitter = CCFFitter()

    # 3. Ajustements sur chaque CCF
    print(f"\n2. Ajustements {fit_model} sur {len(CCFs)} CCFs...")
    start_fit = time.time()

    results = {
        "rv": [],
        "rv_errors": [],
        "fit_results": [],
        "ccfs": CCFs,
        "v_grid": v_grid,
        "success": [],
        "r_squared": [],
        "method": "masks",
        "fit_model": fit_model,
        "window_size_velocity": window_size_velocity,
    }

    for i, ccf in enumerate(CCFs):
        if verbose:
            print(f"  Ajustement spectre {i + 1}/{len(CCFs)}")

        try:
            # Ajustement
            fit_result = fitter.fit_ccf(
                ccf=ccf,
                v_grid=v_grid,
                model=fit_model,
                window_size=window_size_fit,
                poly_order=poly_order,
                **fit_kwargs,
            )

            if fit_result["success"]:
                rv, rv_error = extract_rv_from_fit(fit_result)
                results["rv"].append(rv)
                results["rv_errors"].append(rv_error)
                results["r_squared"].append(fit_result["r_squared"])
                results["success"].append(True)

                if verbose:
                    print(
                        f"    VR = {rv:.2f} ± {rv_error:.2f} m/s (R² = {fit_result['r_squared']:.4f})"
                    )
            else:
                results["rv"].append(np.nan)
                results["rv_errors"].append(np.nan)
                results["r_squared"].append(np.nan)
                results["success"].append(False)

                if verbose:
                    print(
                        f"    Échec ajustement: {fit_result.get('error_message', 'Erreur inconnue')}"
                    )

            results["fit_results"].append(fit_result)

        except Exception as e:
            print(f"    Erreur spectre {i + 1}: {e}")
            results["rv"].append(np.nan)
            results["rv_errors"].append(np.nan)
            results["r_squared"].append(np.nan)
            results["success"].append(False)
            results["fit_results"].append({"success": False, "error_message": str(e)})

    # Conversion en numpy arrays
    results["rv"] = np.array(results["rv"])
    results["rv_errors"] = np.array(results["rv_errors"])
    results["r_squared"] = np.array(results["r_squared"])
    results["success"] = np.array(results["success"])

    end_fit = time.time()
    print(f"Temps total ajustements: {end_fit - start_fit:.2f} s")

    # Statistiques
    n_success = np.sum(results["success"])
    print("\nRésultats:")
    print(f"  Ajustements réussis: {n_success}/{len(specs)}")

    if n_success > 0:
        valid_rvs = results["rv"][results["success"]]
        valid_errors = results["rv_errors"][results["success"]]
        valid_r2 = results["r_squared"][results["success"]]

        print(f"  VR moyenne: {np.mean(valid_rvs):.1f} ± {np.std(valid_rvs):.1f} m/s")
        print(f"  Erreur moyenne: {np.mean(valid_errors):.1f} m/s")
        print(f"  R² moyen: {np.mean(valid_r2):.4f}")

    return results


# Point d'entrée du script: appel de la fonction compute_ccf et tracé
if __name__ == "__main__":
    # Exemple d'utilisation basique
    print("=== EXEMPLE D'UTILISATION DES FONCTIONS CCF ===")

    # Chargement des données
    specs, wavegrid = get_specs_from_h5(
        filepath="/home/tliopis/Codes/lagrange_llopis_mary_2025/data/soapgpu/paper_dataset/spec_cube_tot.h5",
        idx_spec_start=0,
        idx_spec_end=5,  # Seulement quelques spectres pour l'exemple
        wavemin=None,
        wavemax=None,
    )

    template = specs[0]
    specs_to_analyze = specs[1:]

    v_grid = np.arange(-20000, 20000, 100)  # Grille de vitesses

    print("\n1. Test de la fonction extract_rv_template:")
    try:
        results_template = extract_rv_template(
            specs=specs_to_analyze,
            template=template,
            wavegrid=wavegrid,
            v_grid=v_grid,
            fit_model="gaussian",
            window_size=2000,
            verbose=True,
        )

        print("\nRésultats méthode template:")
        for i, (rv, err, success) in enumerate(
            zip(
                results_template["rv"],
                results_template["rv_errors"],
                results_template["success"],
            )
        ):
            if success:
                print(f"  Spectre {i + 1}: VR = {rv:.1f} ± {err:.1f} m/s")
            else:
                print(f"  Spectre {i + 1}: ÉCHEC")

    except Exception as e:
        print(f"Erreur méthode template: {e}")

    print("\n2. Test de la fonction extract_rv_masks:")
    try:
        results_masks = extract_rv_masks(
            specs=specs_to_analyze,
            wavegrid=wavegrid,
            v_grid=v_grid,
            window_size_velocity=820,
            fit_model="gaussian",
            window_size_fit=2000,
            verbose=True,
        )

        print("\nRésultats méthode masques:")
        for i, (rv, err, success) in enumerate(
            zip(
                results_masks["rv"],
                results_masks["rv_errors"],
                results_masks["success"],
            )
        ):
            if success:
                print(f"  Spectre {i + 1}: VR = {rv:.1f} ± {err:.1f} m/s")
            else:
                print(f"  Spectre {i + 1}: ÉCHEC")

    except Exception as e:
        print(f"Erreur méthode masques: {e}")

    # Comparaison des méthodes si les deux ont fonctionné
    if "results_template" in locals() and "results_masks" in locals():
        print("\n3. Comparaison des méthodes:")
        print("Spectre | Template (m/s) | Masques (m/s) | Différence")
        print("-" * 55)

        for i in range(len(specs_to_analyze)):
            rv_temp = (
                results_template["rv"][i] if results_template["success"][i] else np.nan
            )
            rv_mask = results_masks["rv"][i] if results_masks["success"][i] else np.nan

            if not (np.isnan(rv_temp) or np.isnan(rv_mask)):
                diff = rv_temp - rv_mask
                print(f"{i + 1:7d} | {rv_temp:12.1f} | {rv_mask:11.1f} | {diff:8.1f}")
            else:
                print(f"{i + 1:7d} | {'N/A':>12} | {'N/A':>11} | {'N/A':>8}")

    # Tracé simple de la première CCF (ancienne version pour compatibilité)
    print("\n4. Tracé CCF simple (méthode template):")
    CCFs = compute_ccf_template(
        specs=specs[1:2],  # Un seul spectre
        template=template,
        wavegrid=wavegrid,
        v_grid=v_grid,
        dtype=torch.float32,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(v_grid, CCFs[0])
    plt.xlabel("Vitesse (m/s)")
    plt.ylabel("Cross-Correlation Function")
    plt.title("CCF vs. vitesse radiale")
    plt.grid(True)
    plt.show()


# # * On peut maintenant utiliser cette fonction pour construire les masques CCF
# # * Exemple d'utilisation de la fonction build_CCF_masks_sparse
# v_grid = np.arange(-20000, 20000, 10)  # Exemple de grille de vitesses
# window_size_velocity = 820  # Exemple de taille de fenêtre en espace de vitesse
# start = time.time()
# CCF_masks = build_CCF_masks_sparse(
#     line_pos=mask[:, 0],
#     line_weights=mask[:, 1],
#     v_grid=v_grid,
#     wavegrid=wavegrid,
#     window_size_velocity=window_size_velocity,
# )
# end = time.time()
# print(f"Temps de calcul pour build_CCF_masks_sparse: {end - start:.2f} secondes")

# # Calcul de la mémoire utilisée par les trois tableaux internes
# data_nbytes = CCF_masks.data.nbytes
# indices_nbytes = CCF_masks.indices.nbytes
# indptr_nbytes = CCF_masks.indptr.nbytes
# total_bytes = data_nbytes + indices_nbytes + indptr_nbytes


# specs, wavegrid = get_specs_from_h5(
#     filepath="../data/soapgpu/paper_dataset/spec_cube_tot.h5",
#     idx_spec_start=0,
#     idx_spec_end=99,
#     wavemin=None,
#     wavemax=None,
# )
# # * On peut maintenant utiliser les masques CCF pour calculer les CCF des spectres
# print("Calcul des CCF pour les spectres...")

# tmp = CCF_masks.dot(specs.T)  # shape (n_v, n_specs)
# CCF_spec_vs_v = tmp.T  # shape (n_specs, n_v)

# # Affichage de la forme du tableau CCF
# print(f"Forme du tableau CCF: {CCF_spec_vs_v.shape}")

# # Affichage des CCF pour les 5 premiers spectres
# for i in range(5):
#     plot_ccf(
#         CCF_spec_vs_v[i],
#         v_grid,
#         title=f"CCF pour le spectre {i + 1}",
#     )
