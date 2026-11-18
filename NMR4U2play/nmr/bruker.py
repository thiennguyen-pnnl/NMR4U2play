"""Minimal Bruker 1D data reader (equivalent to rbnmr.m used by the
MathWorks repository). Reads `acqus` + `fid` from a zipped experiment
folder uploaded through the app.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile

import numpy as np

from .demo import Dataset1D

_PARAM_RE = re.compile(r"^##\$(\w+)=\s*(.+?)\s*$")


def parse_acqus(text: str) -> dict:
    """Parse scalar ##$KEY= VALUE entries from a Bruker acqus file."""
    params = {}
    for line in text.splitlines():
        m = _PARAM_RE.match(line)
        if not m:
            continue
        key, val = m.groups()
        try:
            params[key] = float(val) if "." in val or "e" in val.lower() \
                else int(val)
        except ValueError:
            params[key] = val.strip("<>")
    return params


def read_fid_bytes(raw: bytes, params: dict) -> np.ndarray:
    """Decode a Bruker `fid` file into a complex FID."""
    byteorder = "<" if params.get("BYTORDA", 0) == 0 else ">"
    dtypa = params.get("DTYPA", 0)
    dtype = np.dtype(byteorder + ("f8" if dtypa == 2 else "i4"))
    data = np.frombuffer(raw, dtype=dtype).astype(float)
    td = int(params.get("TD", len(data)))
    data = data[:td]
    fid = data[0::2] + 1j * data[1::2]
    # remove digital-filter group delay points
    grpdly = params.get("GRPDLY", 0) or 0
    if grpdly and grpdly > 0:
        fid = fid[int(round(float(grpdly))):]
    return fid


def load_bruker_zip(file_like) -> Dataset1D:
    """Load a zipped Bruker experiment folder (must contain acqus + fid)."""
    zf = zipfile.ZipFile(io.BytesIO(file_like.read())
                         if hasattr(file_like, "read") else file_like)
    names = zf.namelist()
    acqus_path = None
    fid_path = None
    for n in names:
        if posixpath.basename(n) == "acqus":
            folder = posixpath.dirname(n)
            candidate = posixpath.join(folder, "fid") if folder else "fid"
            if candidate in names:
                acqus_path, fid_path = n, candidate
                break
    if acqus_path is None:
        raise ValueError("Zip must contain 'acqus' and 'fid' in the same "
                         "folder (a 1D Bruker experiment directory).")

    params = parse_acqus(zf.read(acqus_path).decode("latin-1"))
    fid = read_fid_bytes(zf.read(fid_path), params)

    sw_hz = float(params.get("SW_h", params.get("SW", 10.0) *
                             params.get("SFO1", 400.0)))
    sf_mhz = float(params.get("SFO1", 400.0))
    o1_hz = float(params.get("O1", 0.0))
    bf1 = float(params.get("BF1", sf_mhz))
    center_ppm = o1_hz / bf1 if bf1 else 0.0
    nucleus = str(params.get("NUC1", "1H"))

    return Dataset1D(
        name=posixpath.dirname(acqus_path) or "Bruker experiment",
        fid=fid, sw_hz=sw_hz, sf_mhz=sf_mhz, center_ppm=center_ppm,
        nucleus=nucleus, source="Bruker import",
    )
