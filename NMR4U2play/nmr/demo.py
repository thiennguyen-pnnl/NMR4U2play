"""Synthetic demo datasets.

Mimics the example compounds in the MathWorks repository (ethyl crotonate
and 4-fluorophenol 1H spectra, plus a small COSY) so the app runs with no
spectrometer data. FIDs are built as sums of damped complex exponentials
with J-coupling multiplets, then given deliberate DC offset, noise and
phase errors for the user to correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset1D:
    name: str
    fid: np.ndarray            # complex FID
    sw_hz: float               # spectral width (Hz)
    sf_mhz: float              # spectrometer frequency (MHz)
    center_ppm: float          # carrier position (ppm)
    nucleus: str = "1H"
    source: str = "synthetic demo"
    default_regions: list = field(default_factory=list)  # (lo, hi, label, nH)

    @property
    def dwell(self) -> float:
        return 1.0 / self.sw_hz

    @property
    def td(self) -> int:
        return len(self.fid)


def _multiplet(delta: float, j_list, intensity: float):
    """Split a line at `delta` ppm by each J (Hz); returns [(ppm, Hz_off, amp)].

    Returns list of (delta_ppm, hz_offset_from_center_line, amplitude).
    """
    lines = [(0.0, intensity)]
    for j in j_list:
        new = []
        for off, amp in lines:
            new.append((off - j / 2.0, amp / 2.0))
            new.append((off + j / 2.0, amp / 2.0))
        lines = new
    return [(delta, off, amp) for off, amp in lines]


def _build_fid(lines, td, sw_hz, sf_mhz, center_ppm, t2=0.8,
               noise=0.002, dc_offset=0.03, ph0_err_deg=35.0,
               ph1_err_deg=60.0, seed=7):
    """Sum damped complex exponentials + imperfections."""
    rng = np.random.default_rng(seed)
    t = np.arange(td) / sw_hz
    fid = np.zeros(td, dtype=complex)
    total = sum(amp for *_ , amp in lines)
    for delta, hz_off, amp in lines:
        f_hz = (delta - center_ppm) * sf_mhz + hz_off
        # first-order phase error grows linearly with offset frequency
        phi = np.deg2rad(ph0_err_deg + ph1_err_deg * f_hz / (sw_hz / 2.0))
        fid += (amp / total) * np.exp(1j * (2 * np.pi * f_hz * t + phi)
                                      - t / t2)
    fid += noise * (rng.standard_normal(td) + 1j * rng.standard_normal(td))
    fid += dc_offset * (1 + 1j)
    return fid


def ethyl_crotonate(td: int = 16384) -> Dataset1D:
    """1H spectrum of ethyl crotonate (CH3-CH=CH-COO-CH2-CH3), 400 MHz."""
    sf, sw, center = 400.13, 4800.0, 5.0
    lines = []
    lines += _multiplet(6.95, [15.5, 6.9, 6.9, 6.9], 1.0)   # =CH-CH3 (dq)
    lines += _multiplet(5.84, [15.5, 1.6, 1.6, 1.6], 1.0)   # =CH-CO (dq)
    lines += _multiplet(4.18, [7.1, 7.1, 7.1], 2.0)         # O-CH2 (q)
    lines += _multiplet(1.88, [6.9, 1.6], 3.0)              # =CH-CH3 (dd)
    lines += _multiplet(1.28, [7.1, 7.1], 3.0)              # CH2-CH3 (t)
    fid = _build_fid(lines, td, sw, sf, center)
    regions = [(7.15, 6.75, "=CH-CH3", 1), (6.05, 5.65, "=CH-CO", 1),
               (4.40, 3.95, "O-CH2", 2), (2.05, 1.70, "CH3 (vinyl)", 3),
               (1.50, 1.05, "CH3 (ethyl)", 3)]
    return Dataset1D("Ethyl crotonate (1H, 400 MHz)", fid, sw, sf, center,
                     default_regions=regions)


def fluorophenol(td: int = 16384) -> Dataset1D:
    """1H spectrum of 4-fluorophenol, 400 MHz (aromatic AA'BB'X + OH)."""
    sf, sw, center = 400.13, 3200.0, 5.0
    lines = []
    lines += _multiplet(6.93, [8.7, 8.1], 2.0)   # H ortho to F
    lines += _multiplet(6.77, [8.7, 4.3], 2.0)   # H ortho to OH
    lines += _multiplet(4.65, [], 1.0)           # OH (broad s)
    fid = _build_fid(lines, td, sw, sf, center, t2=0.6, seed=11)
    regions = [(7.10, 6.86, "Ar-H (o-F)", 2), (6.86, 6.60, "Ar-H (o-OH)", 2),
               (4.90, 4.40, "OH", 1)]
    return Dataset1D("4-Fluorophenol (1H, 400 MHz)", fid, sw, sf, center,
                     default_regions=regions)


DEMOS_1D = {
    "Ethyl crotonate (1H, 400 MHz)": ethyl_crotonate,
    "4-Fluorophenol (1H, 400 MHz)": fluorophenol,
}


# ----------------------------------------------------------------------
# 2D COSY demo
# ----------------------------------------------------------------------

def cosy_demo(n1: int = 256, n2: int = 1024):
    """Synthetic magnitude-COSY of ethyl crotonate.

    Returns (fid2d [n1 x n2 complex], sw_hz, sf_mhz, center_ppm).
    Diagonal peaks for every proton; cross peaks between J-coupled pairs.
    """
    sf, sw, center = 400.13, 4800.0, 5.0
    shifts = {"Ha": 6.95, "Hb": 5.84, "CH2": 4.18, "CH3v": 1.88, "CH3e": 1.28}
    amps = {"Ha": 1.0, "Hb": 1.0, "CH2": 2.0, "CH3v": 3.0, "CH3e": 3.0}
    couples = [("Ha", "Hb"), ("Ha", "CH3v"), ("CH2", "CH3e")]

    def f(name):
        return (shifts[name] - center) * sf

    t1 = (np.arange(n1) / sw)[:, None]
    t2 = (np.arange(n2) / sw)[None, :]
    fid = np.zeros((n1, n2), dtype=complex)
    for name, a in amps.items():  # diagonal
        fid += a * np.exp(1j * 2 * np.pi * f(name) * t1 - t1 / 0.15) \
                 * np.exp(1j * 2 * np.pi * f(name) * t2 - t2 / 0.15)
    for a_name, b_name in couples:  # cross peaks (both symmetric partners)
        amp = 0.45 * np.sqrt(amps[a_name] * amps[b_name])
        for p, q in ((a_name, b_name), (b_name, a_name)):
            fid += amp * np.exp(1j * 2 * np.pi * f(p) * t1 - t1 / 0.15) \
                       * np.exp(1j * 2 * np.pi * f(q) * t2 - t2 / 0.15)
    rng = np.random.default_rng(3)
    fid += 0.01 * (rng.standard_normal(fid.shape)
                   + 1j * rng.standard_normal(fid.shape))
    return fid, sw, sf, center


def process_cosy(fid2d: np.ndarray, sw_hz: float, sf_mhz: float,
                 center_ppm: float, ssb: float = 2.0, zf: int = 2):
    """Sine-bell apodization in both dimensions, 2D FFT, magnitude mode.

    ssb: sine-bell shift parameter (2 = cosine bell, like Bruker SSB=2).
    Returns (ppm_f1, ppm_f2, magnitude_spectrum).
    """
    n1, n2 = fid2d.shape

    def sine_bell(n):
        k = np.arange(n)
        phi = np.pi / ssb if ssb >= 1.0 else 0.0
        return np.sin(phi + (np.pi - phi) * k / (n - 1))

    work = fid2d * sine_bell(n2)[None, :] * sine_bell(n1)[:, None]
    m1, m2 = n1 * zf, n2 * zf
    spec = np.fft.fftshift(np.fft.fft2(work, s=(m1, m2)))
    mag = np.abs(spec)
    f1 = center_ppm + np.fft.fftshift(np.fft.fftfreq(m1, d=1.0 / sw_hz)) / sf_mhz
    f2 = center_ppm + np.fft.fftshift(np.fft.fftfreq(m2, d=1.0 / sw_hz)) / sf_mhz
    return f1, f2, mag
