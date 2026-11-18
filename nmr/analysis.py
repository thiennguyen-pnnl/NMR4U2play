"""Signal analysis: peak picking, trapezium-rule integration, Lorentzian fitting.

Ported from the integration and 1D peak-fitting live scripts in
mathworks/Chemistry_NMR_Data_Processing_and_Analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ----------------------------------------------------------------------
# Peak picking
# ----------------------------------------------------------------------

def pick_peaks(ppm: np.ndarray, real_spec: np.ndarray,
               threshold_frac: float = 0.05, min_sep_ppm: float = 0.01):
    """Pick peaks above threshold_frac * max intensity.

    Returns (indices, ppm_positions, heights) sorted by descending ppm.
    """
    height = threshold_frac * real_spec.max()
    dppm = abs(ppm[1] - ppm[0])
    distance = max(1, int(min_sep_ppm / dppm))
    idx, _ = find_peaks(real_spec, height=height, distance=distance)
    order = np.argsort(ppm[idx])[::-1]
    idx = idx[order]
    return idx, ppm[idx], real_spec[idx]


# ----------------------------------------------------------------------
# Integration (trapezium rule)
# ----------------------------------------------------------------------

def integrate_region(ppm: np.ndarray, real_spec: np.ndarray,
                     ppm_lo: float, ppm_hi: float) -> float:
    """Trapezium-rule integral of a ppm region (positive for positive peaks)."""
    lo, hi = sorted((ppm_lo, ppm_hi))
    sel = (ppm >= lo) & (ppm <= hi)
    if sel.sum() < 2:
        return 0.0
    x = ppm[sel]
    y = real_spec[sel]
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    return float(np.trapezoid(y, x))


def integral_table(ppm, real_spec, regions, ref_index: int = 0,
                   ref_value: float = 1.0):
    """Integrate a list of (lo, hi) regions and normalize to one reference.

    Returns list of dicts with region bounds, raw and normalized integrals.
    """
    raw = [integrate_region(ppm, real_spec, lo, hi) for lo, hi in regions]
    ref = raw[ref_index] if (raw and raw[ref_index] != 0) else 1.0
    rows = []
    for (lo, hi), r in zip(regions, raw):
        rows.append({
            "from (ppm)": round(max(lo, hi), 3),
            "to (ppm)": round(min(lo, hi), 3),
            "integral": r,
            "relative": r / ref * ref_value,
        })
    return rows


# ----------------------------------------------------------------------
# Lorentzian peak fitting
# ----------------------------------------------------------------------

def _multi_lorentzian(x, *params):
    """Sum of Lorentzians; params = [amp, x0, fwhm] * n + [offset]."""
    n = (len(params) - 1) // 3
    y = np.full_like(x, params[-1], dtype=float)
    for i in range(n):
        amp, x0, fwhm = params[3 * i: 3 * i + 3]
        hw = max(abs(fwhm), 1e-6) / 2.0
        y += amp * hw ** 2 / ((x - x0) ** 2 + hw ** 2)
    return y


def fit_lorentzians(ppm: np.ndarray, real_spec: np.ndarray,
                    ppm_lo: float, ppm_hi: float, sf_mhz: float,
                    max_peaks: int = 6, threshold_frac: float = 0.05):
    """Fit a sum of Lorentzians to one ppm region.

    Returns (x_region, y_region, y_fit, rows) where rows hold per-peak
    position, height, FWHM (ppm and Hz) and area.
    """
    lo, hi = sorted((ppm_lo, ppm_hi))
    sel = (ppm >= lo) & (ppm <= hi)
    x = ppm[sel]
    y = real_spec[sel]
    if len(x) < 10:
        raise ValueError("Fit region contains too few points.")

    idx, pos, height = pick_peaks(x, y, threshold_frac=threshold_frac,
                                  min_sep_ppm=0.003)
    if len(idx) == 0:
        raise ValueError("No peaks found in the fit region; lower the threshold.")
    if len(idx) > max_peaks:
        keep = np.argsort(height)[::-1][:max_peaks]
        idx, pos, height = idx[keep], pos[keep], height[keep]

    guess_fwhm = max((hi - lo) / 50.0, 3.0 * abs(x[1] - x[0]))
    p0, lbounds, ubounds = [], [], []
    for p, h in zip(pos, height):
        p0 += [h, p, guess_fwhm]
        lbounds += [0.0, lo, abs(x[1] - x[0])]
        ubounds += [np.inf, hi, (hi - lo)]
    p0.append(float(np.median(y)))
    lbounds.append(-np.inf)
    ubounds.append(np.inf)

    popt, _ = curve_fit(_multi_lorentzian, x, y, p0=p0,
                        bounds=(lbounds, ubounds), maxfev=20000)
    y_fit = _multi_lorentzian(x, *popt)

    rows = []
    n = (len(popt) - 1) // 3
    for i in range(n):
        amp, x0, fwhm = popt[3 * i: 3 * i + 3]
        area = np.pi * amp * abs(fwhm) / 2.0  # analytic Lorentzian area
        rows.append({
            "position (ppm)": round(float(x0), 4),
            "height": float(amp),
            "FWHM (ppm)": round(abs(float(fwhm)), 5),
            "FWHM (Hz)": round(abs(float(fwhm)) * sf_mhz, 2),
            "area": float(area),
        })
    rows.sort(key=lambda r: -r["position (ppm)"])
    return x, y, y_fit, rows
