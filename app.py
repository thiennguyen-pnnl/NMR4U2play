"""NMR4U2play — interactive NMR processing and analysis.

A Streamlit port of the MATLAB workflow in
mathworks/Chemistry_NMR_Data_Processing_and_Analysis, styled after the
Apple Human Interface Guidelines.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nmr import analysis, bruker, demo, processing

# ----------------------------------------------------------------------
# Apple HIG design tokens — light (default) and dark appearance.
# Series palettes validated (CVD + contrast) per mode.
# ----------------------------------------------------------------------
FONT_STACK = ('-apple-system, BlinkMacSystemFont, "SF Pro Text", '
              '"Helvetica Neue", Helvetica, Arial, sans-serif')

THEMES = {
    "Light": dict(
        ink="#1D1D1F", ink2="#6E6E73", canvas="#F5F5F7", card="#FFFFFF",
        separator="#E5E5EA", accent="#0071E3", accent_hover="#0077ED",
        tab_bg="#E8E8ED", input_bg="#FFFFFF",
        sidebar_bg="rgba(255,255,255,0.86)",
        card_border="rgba(0,0,0,0.06)", card_shadow="rgba(0,0,0,0.04)",
        series=["#007AFF", "#E68600", "#AF52DE"],   # data, overlay, aux
        marker_red="#FF3B30",
        contour=[[0, "#A8CCF0"], [0.5, "#4895E8"], [1, "#0A3F7A"]],
    ),
    "Dark": dict(
        ink="#F5F5F7", ink2="#98989D", canvas="#000000", card="#1C1C1E",
        separator="#38383A", accent="#0A84FF", accent_hover="#2E95FF",
        tab_bg="#2C2C2E", input_bg="#2C2C2E",
        sidebar_bg="rgba(28,28,30,0.9)",
        card_border="rgba(255,255,255,0.08)", card_shadow="rgba(0,0,0,0.5)",
        series=["#0A84FF", "#C97100", "#BF5AF2"],
        marker_red="#FF453A",
        contour=[[0, "#274E78"], [0.5, "#4895E8"], [1, "#A8CCF0"]],
    ),
}

st.set_page_config(page_title="NMR4U2play", page_icon="🧲", layout="wide")

with st.sidebar:
    st.markdown("## 🧲 NMR4U2play")
    mode = st.radio("Appearance", ["Light", "Dark"], horizontal=True)

T = THEMES[mode]
INK, INK_2 = T["ink"], T["ink2"]
CANVAS, CARD, SEPARATOR = T["canvas"], T["card"], T["separator"]
ACCENT = T["accent"]
SERIES = T["series"]
MARKER_RED = T["marker_red"]

st.markdown(f"""
<style>
    html, body, [class*="st-"], .stMarkdown, button, input, textarea {{
        font-family: {FONT_STACK} !important;
        color: {INK};
    }}
    .stApp {{ background: {CANVAS}; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1200px; }}

    /* Large title, Apple style */
    .hero-title {{
        font-size: 2.6rem; font-weight: 700; letter-spacing: -0.022em;
        margin-bottom: 0;
    }}
    .hero-sub {{ color: {INK_2}; font-size: 1.05rem; margin-top: 0.15rem; }}

    /* Cards */
    .apple-card {{
        background: {CARD}; border-radius: 18px; padding: 1rem 1.25rem;
        border: 1px solid {T["card_border"]};
        box-shadow: 0 1px 3px {T["card_shadow"]};
    }}
    .stat-label {{ color: {INK_2}; font-size: 0.8rem; font-weight: 500;
                   text-transform: uppercase; letter-spacing: 0.04em; }}
    .stat-value {{ font-size: 1.5rem; font-weight: 600;
                   letter-spacing: -0.01em; }}

    /* Tabs as a segmented control */
    .stTabs [data-baseweb="tab-list"] {{
        background: {T["tab_bg"]}; border-radius: 12px; padding: 3px;
        gap: 2px; width: fit-content;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 9px; padding: 6px 16px; background: transparent;
        font-weight: 500; color: {INK};
    }}
    .stTabs [aria-selected="true"] {{
        background: {CARD} !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
    }}
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{ display: none; }}

    /* Buttons: filled pill */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 980px; border: none; background: {ACCENT};
        color: white; font-weight: 500; padding: 0.45rem 1.3rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {T["accent_hover"]}; color: white;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {T["sidebar_bg"]}; backdrop-filter: blur(20px);
        border-right: 1px solid {SEPARATOR};
    }}

    /* Inputs & widgets follow the appearance */
    [data-baseweb="input"], [data-baseweb="select"] > div,
    [data-baseweb="base-input"], .stNumberInput input,
    .stTextInput input {{
        background: {T["input_bg"]} !important; color: {INK} !important;
        border-color: {SEPARATOR} !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: {T["input_bg"]}; color: {INK};
        border: 1px dashed {SEPARATOR};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] .stMarkdown h3 {{
        font-size: 0.85rem; text-transform: uppercase; color: {INK_2};
        letter-spacing: 0.05em;
    }}
    hr {{ border-color: {SEPARATOR}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Plot helpers (Apple-styled Plotly)
# ----------------------------------------------------------------------
def apple_layout(fig: go.Figure, x_title: str, y_title: str,
                 reverse_x: bool = True, height: int = 420,
                 showlegend: bool = False) -> go.Figure:
    fig.update_layout(
        template=None, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT_STACK, color=INK, size=13),
        margin=dict(l=55, r=20, t=30, b=50), height=height,
        hovermode="x unified", showlegend=showlegend,
        legend=dict(orientation="h", y=1.08, x=0,
                    font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=CARD, font_family=FONT_STACK,
                        font_color=INK, bordercolor=SEPARATOR),
    )
    axis = dict(gridcolor=SEPARATOR, zeroline=False, linecolor=SEPARATOR,
                ticks="outside", tickcolor=SEPARATOR,
                title_font=dict(color=INK_2, size=12))
    fig.update_xaxes(title_text=x_title, autorange="reversed" if reverse_x
                     else True, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


PLOTLY_CONF = {"displaylogo": False,
               "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def line_fig(x, traces, x_title, y_title, reverse_x=True, height=420):
    """traces: list of (y, name, color, dash)."""
    fig = go.Figure()
    for y, name, color, dash in traces:
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="%{y:.4g}<extra>" + name + "</extra>"))
    return apple_layout(fig, x_title, y_title, reverse_x, height,
                        showlegend=len(traces) > 1)


def stat_card(col, label, value):
    col.markdown(
        f'<div class="apple-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar — data source + inspector controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### Data")
    source = st.radio("Source", ["Demo spectrum", "Bruker upload (.zip)"],
                      label_visibility="collapsed")
    if source == "Demo spectrum":
        demo_name = st.selectbox("Demo dataset", list(demo.DEMOS_1D))
        dataset = demo.DEMOS_1D[demo_name]()
    else:
        up = st.file_uploader(
            "Zip of a Bruker experiment folder (acqus + fid)", type="zip")
        if up is None:
            st.info("Upload a zipped Bruker 1D experiment folder, or switch "
                    "to a demo spectrum.")
            st.stop()
        try:
            dataset = bruker.load_bruker_zip(up)
        except Exception as exc:
            st.error(f"Could not read Bruker data: {exc}")
            st.stop()

    st.markdown("---")
    st.markdown("### Processing")
    dc_on = st.toggle("DC offset correction", value=True)
    apod_kind = st.selectbox("Apodization",
                             ["Exponential", "Gaussian", "None"])
    lb = st.slider("Line broadening (Hz)", 0.0, 10.0, 0.5, 0.1,
                   disabled=(apod_kind == "None"))
    zf_factor = st.select_slider("Zero-filling ×", options=[1, 2, 4],
                                 value=2)

    st.markdown("### Phasing")
    if "ph0" not in st.session_state:
        st.session_state.ph0 = 0.0
        st.session_state.ph1 = 0.0
    ph0 = st.slider("Zero order ph0 (°)", -180.0, 180.0, step=0.5,
                    key="ph0")
    ph1 = st.slider("First order ph1 (°)", -360.0, 360.0, step=1.0,
                    key="ph1")

    def _run_autophase():
        spec_pre = st.session_state.get("_spec_prephase")
        if spec_pre is not None:
            p0, p1 = processing.autophase(spec_pre)
            st.session_state.ph0 = round(p0, 1)
            st.session_state.ph1 = round(np.clip(p1, -360, 360), 1)

    st.button("Auto-phase", on_click=_run_autophase, width="stretch")

    st.markdown("### Baseline")
    bl_on = st.toggle("Polynomial baseline correction", value=True)
    bl_order = st.slider("Polynomial order", 0, 6, 3,
                         disabled=not bl_on)


# ----------------------------------------------------------------------
# Processing chain (recomputed from the inspector settings)
# ----------------------------------------------------------------------
fid_raw = dataset.fid
fid = processing.dc_offset_correct(fid_raw) if dc_on else fid_raw
window = processing.apodization_window(len(fid), dataset.dwell,
                                       apod_kind, lb)
fid_apod = fid * window
fid_zf = processing.zero_fill(fid_apod, zf_factor)
spec = processing.fourier_transform(fid_zf)
ppm = processing.ppm_axis(len(spec), dataset.sw_hz, dataset.sf_mhz,
                          dataset.center_ppm)
st.session_state["_spec_prephase"] = spec

spec_ph = processing.apply_phase(spec, ph0, ph1)
real = np.real(spec_ph)
if bl_on:
    baseline, bl_mask = processing.baseline_polynomial(real, bl_order)
    real_final = real - baseline
else:
    baseline, real_final = np.zeros_like(real), real

# reset dataset-dependent state when the spectrum changes
if st.session_state.get("_ds_name") != dataset.name:
    st.session_state["_ds_name"] = dataset.name
    st.session_state.pop("regions_df", None)
    st.session_state.pop("fit_result", None)

# ----------------------------------------------------------------------
# Header + stat row
# ----------------------------------------------------------------------
st.markdown('<p class="hero-title">NMR4U2play</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">FID → spectrum → answers. The classic NMR '
            'processing chain, step by step.</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
stat_card(c1, "Dataset", dataset.name.split(" (")[0])
stat_card(c2, "Nucleus", f"{dataset.nucleus}")
stat_card(c3, "Frequency", f"{dataset.sf_mhz:.2f} MHz")
stat_card(c4, "Spectral width", f"{dataset.sw_hz / dataset.sf_mhz:.1f} ppm")
stat_card(c5, "Points (TD → SI)", f"{dataset.td:,} → {len(spec):,}")
st.write("")

tab_fid, tab_phase, tab_base, tab_int, tab_fit, tab_cosy = st.tabs(
    ["FID & FT", "Phasing", "Baseline", "Peaks & integrals",
     "Peak fitting", "2D COSY"])

# ----------------------------------------------------------------------
# Tab 1 — FID and Fourier transform
# ----------------------------------------------------------------------
with tab_fid:
    t_ms = np.arange(len(fid)) * dataset.dwell * 1e3
    st.markdown("**Time domain** — FID after DC correction, with the "
                "apodization window applied.")
    scale = np.abs(fid).max()
    fig = line_fig(
        t_ms,
        [(np.real(fid_apod), "FID (real, apodized)", SERIES[0], None),
         (window * scale, "window (scaled)", SERIES[1], "dash")],
        "time (ms)", "amplitude", reverse_x=False, height=320)
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONF, key="pc_fid")

    st.markdown("**Frequency domain** — magnitude spectrum straight after "
                "the FFT (phase-insensitive preview).")
    fig2 = line_fig(ppm, [(np.abs(spec), "|S(ω)|", SERIES[0], None)],
                    "chemical shift (ppm)", "magnitude", height=360)
    st.plotly_chart(fig2, width="stretch", theme=None, config=PLOTLY_CONF, key="pc_mag")

# ----------------------------------------------------------------------
# Tab 2 — Phasing
# ----------------------------------------------------------------------
with tab_phase:
    st.markdown("**Real spectrum after phase correction.** Adjust ph0/ph1 "
                "in the sidebar, or press **Auto-phase** (entropy "
                "minimization with a negative-area penalty). A well-phased "
                "spectrum has symmetric, fully positive absorption peaks.")
    fig = line_fig(ppm, [(real, "real part", SERIES[0], None)],
                   "chemical shift (ppm)", "intensity", height=430)
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONF,
                    key="pc_phase")
    st.caption(f"Current phases: ph0 = {ph0:.1f}°, ph1 = {ph1:.1f}° "
               f"(pivot at spectrum center).")

# ----------------------------------------------------------------------
# Tab 3 — Baseline
# ----------------------------------------------------------------------
with tab_base:
    if bl_on:
        st.markdown(f"**Iterative polynomial baseline** (order {bl_order}): "
                    "signal points are excluded by a robust threshold and "
                    "the polynomial refit on the remaining baseline points.")
        fig = line_fig(
            ppm,
            [(real, "before correction", SERIES[2], None),
             (baseline, "fitted baseline", SERIES[1], "dash"),
             (real_final, "corrected", SERIES[0], None)],
            "chemical shift (ppm)", "intensity", height=430)
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONF,
                        key="pc_bl_on")
    else:
        st.info("Baseline correction is switched off in the sidebar; the "
                "phased spectrum passes through unchanged.")
        fig = line_fig(ppm, [(real_final, "real part", SERIES[0], None)],
                       "chemical shift (ppm)", "intensity", height=430)
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONF,
                        key="pc_bl_off")

# ----------------------------------------------------------------------
# Tab 4 — Peak picking + integration
# ----------------------------------------------------------------------
with tab_int:
    left, right = st.columns([1, 2.4])
    with left:
        st.markdown("**Peak picking**")
        thr = st.slider("Threshold (% of max)", 0.5, 50.0, 5.0, 0.5)
        min_sep = st.slider("Min. separation (ppm)", 0.001, 0.1, 0.01,
                            0.001, format="%.3f")
    idx, pk_ppm, pk_h = analysis.pick_peaks(ppm, real_final,
                                            thr / 100.0, min_sep)
    with left:
        st.metric("Peaks found", len(idx))

    with right:
        fig = line_fig(ppm, [(real_final, "spectrum", SERIES[0], None)],
                       "chemical shift (ppm)", "intensity", height=380)
        if len(idx):
            fig.add_trace(go.Scatter(
                x=pk_ppm, y=pk_h, mode="markers+text", name="peaks",
                marker=dict(color=MARKER_RED, size=8, symbol="circle"),
                text=[f"{p:.2f}" for p in pk_ppm], textposition="top center",
                textfont=dict(color=INK_2, size=11),
                hovertemplate="%{x:.3f} ppm<extra>peak</extra>"))
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONF,
                        key="pc_peaks")

    st.markdown("---")
    st.markdown("**Integration (trapezium rule).** Edit the regions below; "
                "integrals are normalized to the reference row.")
    if "regions_df" not in st.session_state:
        if dataset.default_regions:
            st.session_state["regions_df"] = pd.DataFrame(
                [{"from (ppm)": r[0], "to (ppm)": r[1], "label": r[2]}
                 for r in dataset.default_regions])
        else:
            lo_default = float(np.round(ppm.min(), 1))
            hi_default = float(np.round(ppm.max(), 1))
            st.session_state["regions_df"] = pd.DataFrame(
                [{"from (ppm)": hi_default, "to (ppm)": lo_default,
                  "label": "region 1"}])

    ctl, tbl = st.columns([1, 2.4])
    with ctl:
        regions_df = st.data_editor(
            st.session_state["regions_df"], num_rows="dynamic",
            width="stretch", key="regions_editor")
        n_rows = len(regions_df.dropna(subset=["from (ppm)", "to (ppm)"]))
        ref_row = st.number_input("Reference row (0-based)", 0,
                                  max(n_rows - 1, 0), 0)
        ref_val = st.number_input("Reference value (e.g. nH)", 0.1, 100.0,
                                  1.0, 0.5)
    regions_clean = regions_df.dropna(subset=["from (ppm)", "to (ppm)"])
    with tbl:
        if len(regions_clean):
            rows = analysis.integral_table(
                ppm, real_final,
                [(r["from (ppm)"], r["to (ppm)"])
                 for _, r in regions_clean.iterrows()],
                ref_index=int(min(ref_row, len(regions_clean) - 1)),
                ref_value=ref_val)
            out = pd.DataFrame(rows)
            out.insert(2, "label", regions_clean["label"].values)
            out["integral"] = out["integral"].map(lambda v: f"{v:.3g}")
            out["relative"] = out["relative"].map(lambda v: f"{v:.2f}")
            st.dataframe(out, width="stretch", hide_index=True)
            st.download_button(
                "Download integrals (CSV)",
                pd.DataFrame(rows).to_csv(index=False).encode(),
                "integrals.csv", "text/csv")
        else:
            st.info("Add at least one integration region.")

# ----------------------------------------------------------------------
# Tab 5 — Lorentzian peak fitting
# ----------------------------------------------------------------------
with tab_fit:
    st.markdown("**Lorentzian line fitting.** Choose a ppm window; peaks "
                "inside it are detected and a sum of Lorentzians is fitted "
                "by non-linear least squares.")
    f1c, f2c, f3c, f4c = st.columns(4)
    fit_hi = f1c.number_input("From (ppm)", value=float(
        np.round(ppm.max() if not dataset.default_regions
                 else dataset.default_regions[0][0], 2)))
    fit_lo = f2c.number_input("To (ppm)", value=float(
        np.round(ppm.min() if not dataset.default_regions
                 else dataset.default_regions[0][1], 2)))
    max_pk = f3c.number_input("Max peaks", 1, 12, 6)
    fit_thr = f4c.number_input("Detection threshold (% of max)", 0.5, 50.0,
                               5.0)
    if st.button("Fit region"):
        try:
            st.session_state["fit_result"] = analysis.fit_lorentzians(
                ppm, real_final, fit_lo, fit_hi, dataset.sf_mhz,
                max_peaks=int(max_pk), threshold_frac=fit_thr / 100.0)
        except Exception as exc:
            st.session_state.pop("fit_result", None)
            st.warning(f"Fit failed: {exc}")
    if "fit_result" in st.session_state:
            x, y, y_fit, rows = st.session_state["fit_result"]
            fig = line_fig(
                x,
                [(y, "data", SERIES[0], None),
                 (y_fit, "Lorentzian fit", SERIES[1], None),
                 (y - y_fit, "residual", SERIES[2], None)],
                "chemical shift (ppm)", "intensity", height=420)
            st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONF,
                            key="pc_fit")
            df = pd.DataFrame(rows)
            df["height"] = df["height"].map(lambda v: f"{v:.3g}")
            df["area"] = df["area"].map(lambda v: f"{v:.3g}")
            st.dataframe(df, width="stretch", hide_index=True)
            d1, d2 = st.columns(2)
            curve_df = pd.DataFrame({"ppm": x, "data": y, "fit": y_fit,
                                     "residual": y - y_fit})
            d1.download_button(
                "Download fit curve (CSV)",
                curve_df.to_csv(index=False).encode(),
                "lorentzian_fit_curve.csv", "text/csv", width="stretch")
            d2.download_button(
                "Download fit parameters (CSV)",
                pd.DataFrame(rows).to_csv(index=False).encode(),
                "lorentzian_fit_parameters.csv", "text/csv",
                width="stretch")

# ----------------------------------------------------------------------
# Tab 6 — 2D COSY demo
# ----------------------------------------------------------------------
with tab_cosy:
    st.markdown("**Magnitude COSY (synthetic ethyl crotonate).** Sine-bell "
                "apodization in both dimensions, 2D FFT, magnitude mode. "
                "Cross peaks link J-coupled protons; extract an F2 slice at "
                "any F1 shift.")
    cc1, cc2, cc3 = st.columns(3)
    ssb = cc1.select_slider("Sine-bell shift (SSB)", options=[1.0, 2.0, 3.0,
                            4.0], value=2.0)
    floor_pct = cc2.slider("Contour floor (% of max)", 1.0, 30.0, 6.0)
    slice_ppm = cc3.slider("F1 slice position (ppm)", 0.5, 7.5, 6.95, 0.01)

    @st.cache_data(show_spinner="Processing 2D COSY…")
    def _cosy(ssb_val):
        fid2d, sw, sf, ctr = demo.cosy_demo()
        return demo.process_cosy(fid2d, sw, sf, ctr, ssb=ssb_val)

    f1_ppm, f2_ppm, mag = _cosy(ssb)
    floor = floor_pct / 100.0 * mag.max()

    fig = go.Figure(go.Contour(
        x=f2_ppm, y=f1_ppm, z=mag,
        contours=dict(start=floor, end=float(mag.max()),
                      size=(mag.max() - floor) / 12, coloring="lines"),
        line_width=1.5,
        colorscale=T["contour"],
        showscale=False,
        hovertemplate=("F2 %{x:.2f} ppm · F1 %{y:.2f} ppm"
                       "<extra>COSY</extra>")))
    fig.add_hline(y=slice_ppm, line=dict(color=MARKER_RED, width=1,
                                         dash="dot"))
    fig = apple_layout(fig, "F2 (ppm)", "F1 (ppm)", reverse_x=True,
                       height=560)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONF,
                    key="pc_cosy")

    i_slice = int(np.argmin(np.abs(f1_ppm - slice_ppm)))
    fig_s = line_fig(f2_ppm, [(mag[i_slice, :],
                               f"F2 slice @ F1 = {f1_ppm[i_slice]:.2f} ppm",
                               SERIES[0], None)],
                     "F2 (ppm)", "magnitude", height=260)
    st.plotly_chart(fig_s, width="stretch", theme=None, config=PLOTLY_CONF,
                    key="pc_cosy_slice")
    slice_df = pd.DataFrame({"F2_ppm": f2_ppm, "magnitude": mag[i_slice, :]})
    st.download_button(
        f"Download F2 slice @ {f1_ppm[i_slice]:.2f} ppm (CSV)",
        slice_df.to_csv(index=False).encode(), "cosy_f2_slice.csv",
        "text/csv")

# ----------------------------------------------------------------------
# Sidebar — export of the processed data (appended below the inspector)
# ----------------------------------------------------------------------
_slug = (dataset.name.split(" (")[0].lower()
         .replace(" ", "_").replace("/", "-"))
with st.sidebar:
    st.markdown("---")
    st.markdown("### Export")
    fmt = st.radio("File format", ["CSV", "TXT (tab-separated)"],
                   horizontal=True, label_visibility="collapsed")
    sep, ext, mime = ((",", "csv", "text/csv") if fmt == "CSV"
                      else ("\t", "txt", "text/plain"))

    spec_df = pd.DataFrame({"ppm": ppm, "intensity": real_final})
    st.download_button(
        f"Spectrum (.{ext})",
        spec_df.to_csv(index=False, sep=sep).encode(),
        f"{_slug}_spectrum.{ext}", mime, width="stretch")

    fid_df = pd.DataFrame({"time_ms": t_ms,
                           "fid_real": np.real(fid_apod),
                           "fid_imag": np.imag(fid_apod)})
    st.download_button(
        f"Processed FID (.{ext})",
        fid_df.to_csv(index=False, sep=sep).encode(),
        f"{_slug}_fid.{ext}", mime, width="stretch")

    if len(idx):
        peaks_df = pd.DataFrame({"ppm": np.round(pk_ppm, 4),
                                 "height": pk_h})
        st.download_button(
            f"Peak list ({len(idx)} peaks, .{ext})",
            peaks_df.to_csv(index=False, sep=sep).encode(),
            f"{_slug}_peaks.{ext}", mime, width="stretch")
    st.caption("Spectrum and FID reflect the current processing, phasing "
               "and baseline settings. Integrals and fit results download "
               "from their own tabs.")

st.markdown(
    f'<p style="color:{INK_2}; font-size:0.85rem; margin-top:2rem;">'
    'NMR4U2play · processing chain ported from '
    '<a href="https://github.com/mathworks/'
    'Chemistry_NMR_Data_Processing_and_Analysis" style="color:#0071E3; '
    'text-decoration:none;">mathworks/Chemistry_NMR_Data_Processing_and_'
    'Analysis</a></p>', unsafe_allow_html=True)
