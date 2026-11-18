"""Core 1D NMR processing chain.

Ported from the MATLAB workflow in
mathworks/Chemistry_NMR_Data_Processing_and_Analysis:
DC offset correction -> apodization -> zero-filling -> FFT ->
phase correction (manual + automatic) -> baseline correction.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# Time-domain steps
# ----------------------------------------------------------------------

def dc_offset_correct(fid: np.ndarray, tail_fraction: float = 0.25) -> np.ndarray:
    """Subtract the mean of the FID tail (assumed pure noise + DC offset)."""
    n_tail = max(16, int(len(fid) * tail_fraction))
    return fid - fid[-n_tail:].mean()


def apodization_window(n: int, dwell: float, kind: str = "Exponential",
                       lb: float = 1.0) -> np.ndarray:
    """Return the apodization window sampled at the FID time points.

    kind: "None", "Exponential" (line broadening lb in Hz) or
    "Gaussian" (gb in Hz).
    """
    t = np.arange(n) * dwell
    if kind == "Exponential":
        return np.exp(-np.pi * max(lb, 0.0) * t)
    if kind == "Gaussian":
        return np.exp(-((np.pi * max(lb, 0.0) * t) ** 2) / (2.0 * np.log(2.0)))
    return np.ones(n)


def zero_fill(fid: np.ndarray, factor: int = 2) -> np.ndarray:
    """Zero-fill to `factor` x the next power of two above len(fid)."""
    if factor < 1:
        return fid
    n_target = int(2 ** np.ceil(np.log2(len(fid)))) * factor
    out = np.zeros(n_target, dtype=complex)
    out[: len(fid)] = fid
    return out


def fourier_transform(fid: np.ndarray) -> np.ndarray:
    """FFT with first-point scaling (halve point 1 to suppress baseline offset)."""
    work = fid.copy()
    work[0] *= 0.5
    return np.fft.fftshift(np.fft.fft(work))


def ppm_axis(n: int, sw_hz: float, sf_mhz: float, center_ppm: float) -> np.ndarray:
    """Chemical-shift axis (ppm) for an fftshifted spectrum of n points."""
    f_hz = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sw_hz))
    return center_ppm + f_hz / sf_mhz


# ----------------------------------------------------------------------
# Phase correction
# ----------------------------------------------------------------------

def apply_phase(spec: np.ndarray, ph0_deg: float, ph1_deg: float,
                pivot_index: int | None = None) -> np.ndarray:
    """Zero- and first-order phase correction about a pivot point."""
    n = len(spec)
    if pivot_index is None:
        pivot_index = n // 2
    idx = np.arange(n)
    phase = np.deg2rad(ph0_deg + ph1_deg * (idx - pivot_index) / n)
    return spec * np.exp(1j * phase)


def _phase_penalty(params, spec, pivot_index):
    """ACME-style penalty: derivative entropy + strong negative-area term."""
    s = np.real(apply_phase(spec, params[0], params[1], pivot_index))
    ds = np.abs(np.diff(s))
    p = ds / (ds.sum() + 1e-14)
    entropy = -np.sum(p * np.log(p + 1e-14))
    neg = s[s < 0.0]
    neg_penalty = np.sum(neg ** 2) / (np.sum(s ** 2) + 1e-14)
    return entropy + 5000.0 * neg_penalty


def autophase(spec: np.ndarray, pivot_index: int | None = None):
    """Automatic phase correction. Returns (ph0_deg, ph1_deg).

    Coarse grid search on ph0, then Nelder-Mead refinement of (ph0, ph1).
    """
    n = len(spec)
    if pivot_index is None:
        pivot_index = n // 2
    coarse = np.arange(-180.0, 180.0, 10.0)
    scores = [_phase_penalty((p0, 0.0), spec, pivot_index) for p0 in coarse]
    ph0_start = coarse[int(np.argmin(scores))]
    res = minimize(_phase_penalty, x0=[ph0_start, 0.0],
                   args=(spec, pivot_index), method="Nelder-Mead",
                   options={"xatol": 0.05, "fatol": 1e-8, "maxiter": 600})
    ph0, ph1 = res.x
    ph0 = (ph0 + 180.0) % 360.0 - 180.0
    return float(ph0), float(ph1)


# ----------------------------------------------------------------------
# Baseline correction
# ----------------------------------------------------------------------

def baseline_polynomial(real_spec: np.ndarray, order: int = 3,
                        n_iter: int = 6, k: float = 3.0):
    """Iterative polynomial baseline on signal-free points.

    Fits a polynomial to the whole spectrum, then repeatedly excludes
    points more than k * MAD above the current baseline (signals) and
    refits. Returns (baseline, mask_of_baseline_points).
    """
    n = len(real_spec)
    x = np.linspace(-1.0, 1.0, n)
    mask = np.ones(n, dtype=bool)
    baseline = np.zeros(n)
    for _ in range(n_iter):
        if mask.sum() < (order + 2):
            break
        coeffs = np.polyfit(x[mask], real_spec[mask], order)
        baseline = np.polyval(coeffs, x)
        resid = real_spec - baseline
        mad = np.median(np.abs(resid[mask] - np.median(resid[mask]))) + 1e-14
        mask = resid < k * 1.4826 * mad
    return baseline, mask
