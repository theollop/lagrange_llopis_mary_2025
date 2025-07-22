import numpy as np
import torch
from numba import njit
from io_spec import get_specs_from_h5
from cuda import get_free_memory
import matplotlib.pyplot as plt
import time
from math import floor

# --- Constants ---
c = 299_792_458.0  # speed of light in m/s

# --- Original Utilities ---


@njit(inline="always")
def _gamma(v):
    return np.sqrt((1 + v / c) / (1 - v / c))


def gamma_torch(v: torch.Tensor) -> torch.Tensor:
    """Lorentz factor for a torch.Tensor of velocities."""
    return torch.sqrt((1 + v / c) / (1 - v / c))


def linear_interpolate_spec(
    x_ref: torch.Tensor,  # (L,), ascending
    y_ref: torch.Tensor,  # (L,)
    x_new: torch.Tensor,  # (...,)
) -> torch.Tensor:
    """Piecewise-linear interpolation on GPU via torch.bucketize."""
    dx = x_ref[1:] - x_ref[:-1]  # (L-1)
    m = (y_ref[1:] - y_ref[:-1]) / dx  # (L-1)
    b = y_ref[:-1] - m * x_ref[:-1]  # (L-1)
    m_pad = torch.cat([m[:1], m, m[-1:]])  # (L+1)
    b_pad = torch.cat([b[:1], b, b[-1:]])  # (L+1)
    idx = torch.bucketize(x_new, x_ref)  # (...,)
    idx = idx.clamp(0, m_pad.shape[0] - 1)
    m_sel = m_pad[idx]
    b_sel = b_pad[idx]
    return m_sel * x_new + b_sel


def shift_spec(
    spec: np.ndarray,  # (L,)
    wavegrid: np.ndarray,  # (L,)
    v: np.ndarray,  # (Nv,)
    out_dtype=torch.float32,  # torch dtype
    device="cuda",
) -> torch.Tensor:
    """Shift 'spec' by Doppler v and interpolate back to original wavegrid."""
    # to GPU tensors
    spec_t = torch.as_tensor(spec, dtype=out_dtype, device=device)  # (L,)
    wave_t = torch.as_tensor(wavegrid, dtype=out_dtype, device=device)  # (L,)
    v_t = torch.as_tensor(v, dtype=out_dtype, device=device)  # (Nv,)
    # compute gamma and shifted wavelengths
    gamma_t = gamma_torch(v_t).unsqueeze(1)  # (Nv,1)
    wave_shift = wave_t.unsqueeze(0) * gamma_t  # (Nv,L)
    # interpolate back onto original wavegrid
    return linear_interpolate_spec(
        x_ref=wave_t, y_ref=spec_t, x_new=wave_shift
    )  # → (Nv,L)


# --- New FFT-Based CCF in log‑wavelength space ---


def ccf_log_fft(
    template: np.ndarray,  # (L,)
    spec: np.ndarray,  # (L,)
    wavegrid: np.ndarray,  # (L,)
    v_grid: np.ndarray,  # (Nv,)
    out_dtype=torch.float32,
    device="cuda",
) -> torch.Tensor:
    """
    Compute the CCF by:
      1) resampling template & spec onto a uniform ln(λ) grid,
      2) doing FFT-based cross-correlation,
      3) sampling the correlation at Doppler shifts corresponding to v_grid.
    Returns a torch.Tensor of shape (Nv,).
    """
    # move to torch
    wave_t = torch.as_tensor(wavegrid, dtype=out_dtype, device=device)  # (L,)
    temp_t = torch.as_tensor(template, dtype=out_dtype, device=device)  # (L,)
    spec_t = torch.as_tensor(spec, dtype=out_dtype, device=device)  # (L,)

    # 1) define uniform ln(λ) grid
    ln_wave = torch.log(wave_t)
    ln_min, ln_max = ln_wave.min(), ln_wave.max()
    L = wave_t.numel()
    ln_uniform = torch.linspace(ln_min, ln_max, steps=L, dtype=out_dtype, device=device)
    lam_uniform = torch.exp(ln_uniform)  # (L,)

    # 2) resample both spectra onto that grid
    templ_u = linear_interpolate_spec(wave_t, temp_t, lam_uniform)  # (L,)
    spec_u = linear_interpolate_spec(wave_t, spec_t, lam_uniform)  # (L,)

    # 3) prepare zero-padded FFT size
    nfft = 2 ** ((2 * L - 1).bit_length())
    pad_t = torch.zeros(nfft, dtype=out_dtype, device=device)
    pad_s = torch.zeros(nfft, dtype=out_dtype, device=device)
    pad_t[:L] = templ_u
    pad_s[:L] = spec_u

    # 4) FFT-based correlation: corr[k] = sum spec[i] * templ[i+k]
    S = torch.fft.fft(pad_s)
    T = torch.fft.fft(pad_t)
    corr = torch.fft.ifft(S * torch.conj(T)).real  # (nfft,)

    # 5) extract the 'full' correlation of length 2L-1
    corr_full = torch.cat([corr[-(L - 1) :], corr[:L]])  # (2L-1,)

    # 6) map each velocity to a fractional lag in ln(λ) space
    v_t_grid = torch.as_tensor(v_grid, dtype=out_dtype, device=device)
    gamma_v = torch.sqrt((1 + v_t_grid / c) / (1 - v_t_grid / c))
    delta_ln = torch.log(gamma_v)  # (Nv,)
    dln = (ln_max - ln_min) / (L - 1)
    positions = delta_ln / dln + (L - 1)  # fractional indices in [0,2L-2]

    # 7) linear interpolate corr_full at those positions
    idx_lo = torch.floor(positions).long().clamp(0, 2 * L - 2)
    idx_hi = (idx_lo + 1).clamp(0, 2 * L - 2)
    w = positions - idx_lo.float()
    return corr_full[idx_lo] * (1 - w) + corr_full[idx_hi] * w  # (Nv,)


# --- Direct CCF via shifting & dot-product (batched) ---


def ccf_direct_batched(
    template: np.ndarray,
    spec: np.ndarray,
    wavegrid: np.ndarray,
    v_grid: np.ndarray,
    batch_size: int,
    out_dtype=torch.float32,
    device="cuda",
) -> np.ndarray:
    """
    Compute the CCF by shifting the template and dotting with spec, in batches.
    Returns a NumPy array of length Nv.
    """
    pieces = []
    for batch in (
        v_grid[i : i + batch_size] for i in range(0, len(v_grid), batch_size)
    ):
        shifted = shift_spec(
            template, wavegrid, batch, out_dtype, device=device
        )  # (Nb,L)
        # dot with spec
        spec_t = torch.as_tensor(spec, dtype=out_dtype, device=device)
        ccf_piece = torch.sum(shifted * spec_t.unsqueeze(0), dim=1)  # (Nb,)
        pieces.append(ccf_piece.cpu().numpy())
        del shifted, ccf_piece
    return np.concatenate(pieces, axis=0)  # (Nv,)


# --- Helper to compute max batch size based on free GPU memory ---


def get_max_nv(L, free_memory, dtype=torch.float32):
    """Estimate max Nv for (Nv,L) arrays in GPU, accounting for intermediate tensors."""
    if dtype == torch.float64:
        Nv_max = (free_memory - 56 * L + 8) / (8 + 36 * L)
    elif dtype == torch.float32:
        Nv_max = (free_memory - 28 * L + 4) / (4 + 28 * L)
    else:
        raise ValueError("Unsupported dtype.")
    return floor(Nv_max)


# --- Main execution ---

if __name__ == "__main__":
    # 1) load spectra & define parameters
    specs, wavegrid = get_specs_from_h5(
        wavemax=None, wavemin=None, idx_spec_start=0, idx_spec_end=10
    )
    template = specs[0]  # template spectrum
    spec_for_ccf = specs[5]  # the one we correlate against

    # velocity grid
    v_grid = np.arange(-20_000, 20_000, 100)  # m/s

    # GPU memory & batching
    M = get_free_memory() * 0.9
    L = wavegrid.shape[0]
    dtype = torch.float32
    Nv_max = get_max_nv(L, M, dtype)
    batch_size = max(1, Nv_max)
    print(f"Using batch_size = {batch_size} for direct method.")

    # 2) time the direct method
    t0 = time.time()
    ccf_direct = ccf_direct_batched(
        template, spec_for_ccf, wavegrid, v_grid, batch_size, dtype, device="cuda"
    )
    dt_direct = time.time() - t0
    print(f"Direct method time: {dt_direct:.3f} s")

    # 3) time the FFT-log method
    t1 = time.time()
    ccf_fft = (
        ccf_log_fft(template, spec_for_ccf, wavegrid, v_grid, dtype, device="cuda")
        .cpu()
        .numpy()
    )
    dt_fft = time.time() - t1
    print(f"FFT-log method time: {dt_fft:.3f} s")

    # 4) comparison & plot
    plt.figure(figsize=(8, 5))
    plt.plot(v_grid, ccf_direct, label="Direct shift+dot")
    plt.plot(v_grid, ccf_fft, label="Log‑FFT CCF", linestyle="--")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("CCF")
    plt.title("Comparison of CCF Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5) summary
    speedup = dt_direct / dt_fft if dt_fft > 0 else np.inf
    print(f"Speedup (direct / fft-log): {speedup:.2f}×")

    # cleanup
    del spec_for_ccf, template
    torch.cuda.empty_cache()
