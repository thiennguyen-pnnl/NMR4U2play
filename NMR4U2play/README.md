# NMR4U2play

Interactive Streamlit app for NMR data processing and analysis. It is a
Python port of the MATLAB workflow in
[mathworks/Chemistry_NMR_Data_Processing_and_Analysis](https://github.com/mathworks/Chemistry_NMR_Data_Processing_and_Analysis),
with a UI styled after the Apple Human Interface Guidelines (light
appearance by default, dark available from the sidebar).

## Features

| Step | What it does |
|------|--------------|
| Import | Zipped Bruker 1D experiment folder (`acqus` + `fid`, group-delay handled), or built-in synthetic demos (ethyl crotonate, 4-fluorophenol) |
| Initial processing | DC offset correction, exponential/Gaussian apodization, zero-filling, FFT |
| Phasing | Manual ph0/ph1 sliders + automatic phasing (entropy minimization with negative-area penalty) |
| Baseline | Iterative polynomial baseline correction with robust signal exclusion |
| Peaks & integrals | Threshold peak picking; editable integration regions; trapezium-rule integrals normalized to a reference (CSV export) |
| Peak fitting | Multi-Lorentzian least-squares fit of any ppm window (position, FWHM in ppm/Hz, area) |
| 2D COSY | Synthetic magnitude COSY: sine-bell apodization, 2D FFT, contour map, F1-slice extraction |

## Run

```bash
cd NMR4U2play
pip install -r requirements.txt
streamlit run app.py
```

The two demo spectra carry deliberate DC offset, noise, and zero/first
order phase errors, so every processing step has something real to fix.

## Layout

```
NMR4U2play/
├── app.py               # Streamlit UI (Apple HIG styling, light/dark)
├── nmr/
│   ├── processing.py    # DC offset, apodization, zero-fill, FFT, phasing, baseline
│   ├── analysis.py      # peak picking, trapezium integration, Lorentzian fitting
│   ├── bruker.py        # minimal Bruker acqus/fid reader (zip upload)
│   └── demo.py          # synthetic 1D datasets + 2D COSY generator
├── .streamlit/config.toml
└── requirements.txt
```
