from math import floor
import numpy as np
import torch
from numba import njit


# -----------------------------------------------------------------------------
# Fonctions du facteur de Lorentz
# -----------------------------------------------------------------------------
@njit(inline="always")
def _gamma_numba(v: float) -> np.float64:
    """
    Calcule le facteur de Lorentz pour une vitesse donnée.

    Args:
        v: Vitesse en m/s.

    Returns:
        Facteur de Lorentz en np.float64.
    """
    C_LIGHT = 299_792_458  # Vitesse de la lumière en m/s
    return np.sqrt((1 + v / C_LIGHT) / (1 - v / C_LIGHT))


def gamma_np(v: np.ndarray) -> np.ndarray:
    """
    Calcule le facteur de Lorentz pour un tableau numpy de vitesses.

    Args:
        v: Tableau de vitesses (m/s).

    Returns:
        Tableau des facteurs de Lorentz.
    """
    C_LIGHT = 299_792_458
    return np.sqrt((1 + v / C_LIGHT) / (1 - v / C_LIGHT))


def gamma_torch(v: torch.Tensor) -> torch.Tensor:
    """
    Calcule le facteur de Lorentz pour un tenseur PyTorch de vitesses.

    Args:
        v: Tenseur de vitesses (m/s).

    Returns:
        Tenseur des facteurs de Lorentz.
    """
    C_LIGHT = 299_792_458
    return torch.sqrt((1 + v / C_LIGHT) / (1 - v / C_LIGHT))


# -----------------------------------------------------------------------------
# Extension de la grille de longueurs d’onde pour les décalages Doppler
# -----------------------------------------------------------------------------
@njit(inline="always")
def extend_wavegrid_numba(
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Étend une grille de longueurs d'onde à pas constant pour couvrir les décalages Doppler maximaux.

    Args:
        wavegrid: Tableau trié de longueurs d'onde à pas constant.
        v_grid: Tableau de vitesses radiales pour lesquelles préparer les décalages.

    Returns:
        wavegrid_extended: Grille complète incluant les extensions avant et après.
        wave_before: Portion de la grille avant la longueur d'onde minimale d'origine.
        wave_after: Portion de la grille après la longueur d'onde maximale d'origine.
    """
    C_LIGHT = 299_792_458
    SAFETY_MARGIN = 100
    step = wavegrid[1] - wavegrid[0]
    vmax = v_grid.max() + SAFETY_MARGIN
    vmin = v_grid.min() - SAFETY_MARGIN

    gamma_max = _gamma_numba(vmax)
    gamma_min = _gamma_numba(-vmin)

    wl_max = (wavegrid * gamma_max).max()
    wl_min = (wavegrid * gamma_min).min()

    wave_before = np.arange(wl_min, wavegrid.min(), step)
    wave_after = np.arange(wavegrid.max() + step, wl_max, step)

    total_len = len(wave_before) + len(wavegrid) + len(wave_after)
    extended = np.empty(total_len)
    extended[: len(wave_before)] = wave_before
    extended[len(wave_before) : len(wave_before) + len(wavegrid)] = wavegrid
    extended[len(wave_before) + len(wavegrid) :] = wave_after

    return extended, wave_before, wave_after


# -----------------------------------------------------------------------------
# Interpolation linéaire sur GPU pour spectres décalés
# -----------------------------------------------------------------------------
def linear_interpolate_spec(
    x_ref: torch.Tensor,
    y_ref: torch.Tensor,
    x_new: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolation linéaire par morceaux (et extrapolation) de y_ref(x_ref) en de nouveaux points x_new.

    Args:
        x_ref: Tenseur 1D croissant de valeurs x de référence, forme (L,).
        y_ref: Tenseur 1D des valeurs y de référence, forme (L,).
        x_new: Tenseur des points de requête, forme (..., L).

    Returns:
        Tenseur des valeurs interpolées, même forme que x_new.
    """
    dx = x_ref[1:] - x_ref[:-1]
    m = (y_ref[1:] - y_ref[:-1]) / dx
    b = y_ref[:-1] - m * x_ref[:-1]

    m_pad = torch.cat([m[:1], m, m[-1:]])
    b_pad = torch.cat([b[:1], b, b[-1:]])

    idx = torch.bucketize(x_new, x_ref)
    idx = idx.clamp(0, m_pad.shape[0] - 1)

    m_sel = m_pad[idx]
    b_sel = b_pad[idx]
    return m_sel * x_new + b_sel


# -----------------------------------------------------------------------------
# Décalage Doppler et calcul de la CCF
# -----------------------------------------------------------------------------
def shift_spec(
    spec: np.ndarray,
    wavegrid: np.ndarray,
    velocities: np.ndarray,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Applique le décalage Doppler à un spectre modèle pour plusieurs vitesses,
    puis interpole sur la grille d'onde d'origine.

    Args:
        spec: Tableau 1D de flux (forme (L,)).
        wavegrid: Tableau 1D de longueurs d'onde (forme (L,)).
        velocities: Tableau 1D de vitesses (forme (Nv,)).
        dtype: torch.dtype (float64 ou float32) pour le calcul.

    Returns:
        Tenseur de forme (Nv, L) contenant les spectres décalés.
    """
    device = torch.device("cuda")

    spec_t = torch.as_tensor(spec, dtype=dtype, device=device)
    wave_t = torch.as_tensor(wavegrid, dtype=dtype, device=device)

    gamma_t = gamma_torch(torch.as_tensor(velocities, dtype=dtype, device=device))
    wave_shift = wave_t.unsqueeze(0) * gamma_t.unsqueeze(1)

    return linear_interpolate_spec(x_ref=wave_t, y_ref=spec_t, x_new=wave_shift)


# -----------------------------------------------------------------------------
# Calcul de la taille de batch selon la mémoire GPU
# -----------------------------------------------------------------------------
def get_max_nv(
    L: int,
    free_memory_bytes: float,
    dtype: torch.dtype = torch.float64,
) -> int:
    """
    Estime le nombre maximal de vitesses (Nv) compatibles avec la mémoire GPU.

    Args:
        L: Nombre de longueurs d'onde.
        free_memory_bytes: Mémoire GPU disponible en octets.
        dtype: torch.float32 ou torch.float64.

    Returns:
        Nombre entier maximal de vitesses (Nv).
    """
    # Modèle mémoire : wave_shift & output, idx, m_sel+b_sel, overhead
    if dtype == torch.float64:
        nv = (free_memory_bytes - 56 * L + 8) / (8 + 36 * L)
    elif dtype == torch.float32:
        nv = (free_memory_bytes - 28 * L + 4) / (4 + 28 * L)
    else:
        raise ValueError("dtype doit être torch.float32 ou torch.float64")

    return floor(nv)
