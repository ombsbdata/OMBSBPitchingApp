#!/usr/bin/env python3
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os, re, tempfile
import gdown

#stuff calculator render
from stuff_calculator_lgbm import render_stuff_calculator_tab

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def _max_width(px=1700):
    st.markdown(
        f"""
        <style>
          .block-container {{ max-width: {px}px; padding-left: 1rem; padding-right: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
_max_width(1700)  # bump to taste (1600–1800 is nice)


# -----------------------------------------------------------------------------------
# Config / constants
# -----------------------------------------------------------------------------------
SEASON_FILE = "Fall_2025_wRV_with_stuff.csv"     # main dataset (plays)
ROLLING_FILE = "Fall_2025_wRV_with_stuff.csv"    # if you truly have a separate rolling file, point it here
STUFFPLUS_FILE = "Fall_2025_wRV_with_stuff.csv"  # Stuff+ source; can be the same file in your case

TEAM_FILTER = "OLE_REB"  # change to None to allow all teams

# Minimal columns we expect (app will degrade gracefully if some are missing)
REQUIRED_COLS = [
    "Pitcher", "PitcherTeam", "Date",
    "TaggedPitchType", "AutoPitchType",
    "PlateLocSide", "PlateLocHeight",
    "RelSpeed", "SpinRate", "Tilt", "RelHeight", "RelSide", "Extension",
    "InducedVertBreak", "HorzBreak", "VertApprAngle", "HorzRelAngle", "VertRelAngle", "HorzApprAngle",
    "ExitSpeed", "Angle", "PitchNo", "PitchCall", "Balls", "Strikes", "BatterSide"
]

# Canonical mapping for pitch types — extend as needed
PITCH_NORMALIZE_MAP = {
    "FF": "Four-Seam", "Four-Seam": "Four-Seam", "FourSeam": "Four-Seam", "Fastball": "Four-Seam",
    "SI": "Sinker", "Sinker": "Sinker", "Two-Seam": "Sinker",
    "SL": "Slider", "Slider": "Slider",
    "CU": "Curveball", "Curveball": "Curveball", "KC": "Curveball",
    "FC": "Cutter", "Cutter": "Cutter",
    "CH": "Changeup", "ChangeUp": "Changeup", "Changeup": "Changeup",
    "FS": "Splitter", "Splitter": "Splitter",
    "KN": "Knuckleball", "Knuckleball": "Knuckleball",
    "PO": "PitchOut", "PitchOut": "PitchOut",
    "UN": "Unknown", "Undefined": "Unknown", "Unknown": "Unknown", "Other": "Other"
}

PLOTLY_COLORS = {
    "Four-Seam": "royalblue",
    "Sinker": "goldenrod",
    "Slider": "mediumseagreen",
    "Curveball": "firebrick",
    "Cutter": "darkorange",
    "Changeup": "mediumpurple",
    "Splitter": "teal",
    "Knuckleball": "gray",
    "PitchOut": "lightgray",
    "Unknown": "black",
    "Other": "black",
}

NUMERIC_COLS = [
    "RelSpeed", "SpinRate", "Tilt", "RelHeight", "RelSide",
    "Extension", "InducedVertBreak", "HorzBreak", "VertApprAngle",
    "ExitSpeed", "PlateLocSide", "PlateLocHeight", "PitchNo",
    "HorzRelAngle", "VertRelAngle", "HorzApprAngle"
]

# --- Bio-Mech Google Sheet (download as XLSX via gdown) ---
BIOMECH_FILE_ID = "1dR8K0QPugSz34IpfndbFiSyp8OpYCOrbX0xHntwTtO8"
BIOMECH_EXPORT_XLSX_URL = f"https://docs.google.com/spreadsheets/d/{BIOMECH_FILE_ID}/export?format=xlsx"



# -----------------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------------
def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _ensure_date(df: pd.DataFrame, col="Date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df[df[col].notna()]
    return df

def _has_cols(df: pd.DataFrame, cols) -> bool:
    return all(c in df.columns for c in cols)

def canonical_pitch_type(row) -> str:
    """
    Build a canonical PitchType using TaggedPitchType, fallback to AutoPitchType,
    then normalize via PITCH_NORMALIZE_MAP.
    """
    raw = str(row.get("TaggedPitchType") or "").strip()
    if not raw or raw.lower().startswith("undefined"):
        raw = str(row.get("AutoPitchType") or "").strip()
    return PITCH_NORMALIZE_MAP.get(raw, raw if raw else "Unknown")

# Bio mech util

# ==== Bio-Mech loader that prefers local combined CSV (from parse_biomech.py) ====
@st.cache_data(show_spinner=False)
def load_biomech_data() -> pd.DataFrame:
    local_path = os.path.join("biomech_clean", "biomech_all.csv")
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        # Clean base columns
        if "Player" in df.columns:
            df["Player"] = df["Player"].astype(str).str.strip()
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        # Ensure PlayerKey exists and is normalized
        if "PlayerKey" not in df.columns:
            df["PlayerKey"] = df["Player"].apply(canon_key_last_first)
        else:
            df["PlayerKey"] = df["PlayerKey"].fillna(df["Player"].apply(canon_key_last_first))
            df["PlayerKey"] = df["PlayerKey"].astype(str).str.strip()
        # Numeric coercion
        for c in df.columns:
            if c not in ("Player","PlayerKey","DATE","PREP_TIME"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values(["Player","DATE"], na_position="last")
    # fallback: live workbook (uses canon_key_last_first inside)
    return load_biomech_workbook()



def _biomech_summary(one: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a single-row DF with Events (unique dates) and mean of Fresh/Post metrics.
    """
    if one.empty:
        return pd.DataFrame()

    metrics = ["RSI","PP","ECC_PP","JH"]
    res = {}

    # Events = unique non-null dates within the filtered set
    if "DATE" in one.columns:
        res["Events"] = int(one["DATE"].dropna().dt.normalize().nunique())
    else:
        res["Events"] = int(len(one))

    for m in metrics:
        fcol = f"Fresh_{m}"
        pcol = f"Post_{m}"
        if fcol in one.columns:
            res[f"Fresh_{m}"] = float(np.nanmean(one[fcol].values))
        if pcol in one.columns:
            res[f"Post_{m}"] = float(np.nanmean(one[pcol].values))
        # Optional delta:
        if fcol in one.columns and pcol in one.columns:
            fmean = np.nanmean(one[fcol].values)
            pmean = np.nanmean(one[pcol].values)
            res[f"Δ_{m}"] = float(pmean - fmean) if not (np.isnan(fmean) or np.isnan(pmean)) else np.nan

    # include velo/weight averages if present
    for extra in ["TOP_VELO","lbs"]:
        if extra in one.columns:
            res[f"Avg_{extra}"] = float(np.nanmean(one[extra].values))

    # tidy display order
    cols = ["Events",
            "Fresh_RSI","Post_RSI","Δ_RSI",
            "Fresh_PP","Post_PP","Δ_PP",
            "Fresh_ECC_PP","Post_ECC_PP","Δ_ECC_PP",
            "Fresh_JH","Post_JH","Δ_JH",
            "Avg_TOP_VELO","Avg_lbs"]
    cols = [c for c in cols if c in res]
    out = pd.DataFrame([{k: res.get(k, np.nan) for k in cols}])
    # round nicely
    for c in out.columns:
        if c != "Events":
            out[c] = out[c].round(2)
    return out

def canon_key_last_first(name: str) -> str:
    """
    Normalize 'Last, First' or 'First Last' to 'lastfirst' (letters only, lowercase).
    Also tolerates 'Last,First' (no space) and extra whitespace.
    """
    if not name:
        return ""
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)

    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
    else:
        parts = s.split(" ")
        if len(parts) >= 2:
            first = " ".join(parts[1:])
            last  = parts[0]
        else:
            first, last = s, ""

    last  = re.sub(r"[^A-Za-z]", "", last)
    first = re.sub(r"[^A-Za-z]", "", first)
    return (last + first).lower()



# End of bio mech util

# -----------------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------------
@st.cache_data
def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = _ensure_date(df, "Date")
    df = _coerce_numeric(df, NUMERIC_COLS)
    # add a canonical 'PitchType' for grouping/merges
    if "PitchType" not in df.columns:
        df["PitchType"] = df.apply(canonical_pitch_type, axis=1)
    if "Season" not in df.columns:
        df["Season"] = "2025 Season"
    return df

# -----------------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------------
season_df = load_csv(SEASON_FILE)
rolling_df = load_csv(ROLLING_FILE)
# --- Build stuff_df ONCE, with canonical PitchType and numeric StuffPlus
stuff_src = load_csv(STUFFPLUS_FILE)  # has PitchType from load_csv()
if "StuffPlus" in stuff_src.columns:
    stuff_src["StuffPlus"] = pd.to_numeric(stuff_src["StuffPlus"], errors="coerce")
    # keep only the columns we need
    stuff_df = stuff_src[["Pitcher", "PitchType", "StuffPlus"]].dropna(subset=["PitchType"])
else:
    stuff_df = pd.DataFrame(columns=["Pitcher", "PitchType", "StuffPlus"])

if TEAM_FILTER and "PitcherTeam" in season_df.columns:
    season_df = season_df[season_df["PitcherTeam"] == TEAM_FILTER]


### Bio Mech Functions

_PLAYER_RE = re.compile(r"^[A-Za-z .'\-]+,\s*[A-Za-z .'\-]+$")  # LAST, FIRST or LAST,FIRST

def _is_last_first(s: str) -> bool:
    return bool(_PLAYER_RE.match((s or "").strip()))

def _norm_band(x: str) -> str:
    x = str(x).strip().upper()
    if x.startswith("FRESH"): return "Fresh"
    if x.startswith("POST"):  return "Post"
    return ""

def _flatten_cols(bands_row: pd.Series, leaves_row: pd.Series):
    bands = list(bands_row.fillna(method="ffill"))
    leaves = [str(x).strip() if pd.notna(x) else "" for x in leaves_row]
    cols = []
    for b, l in zip(bands, leaves):
        l2 = re.sub(r"\s+", "_", l).replace("%", "Pct")
        b2 = _norm_band(b)
        cols.append(l2 if not b2 else f"{b2}_{l2}")
    return cols

@st.cache_data(show_spinner=False)
def load_biomech_workbook() -> pd.DataFrame:
    # download once
    td = tempfile.mkdtemp()
    xlsx_path = os.path.join(td, "bio_mech.xlsx")
    gdown.download(url=BIOMECH_EXPORT_XLSX_URL, output=xlsx_path, quiet=True)

    sheets = pd.read_excel(xlsx_path, sheet_name=None, header=None, engine="openpyxl")
    frames = []

    for sheet_name, raw in sheets.items():
        if raw is None or raw.empty or raw.shape[0] < 3:
            continue

        title_cell = str(raw.iat[0, 0]).strip() if pd.notna(raw.iat[0, 0]) else ""
        # Accept player if either A1 or the tab name is LAST, FIRST/LAST,FIRST
        if _is_last_first(title_cell):
            player_display = title_cell
        elif _is_last_first(str(sheet_name)):
            player_display = str(sheet_name)
        else:
            # skip non-player tabs (e.g., OLE MISS AVERAGES)
            continue

        band_row = raw.iloc[1].astype(str).str.upper()
        if not band_row.str.contains("FRESH|POST", regex=True).any():
            continue

        cols = _flatten_cols(raw.iloc[1], raw.iloc[2])
        df = raw.iloc[3:].copy()
        df.columns = cols
        df = df.dropna(how="all")
        if df.empty:
            continue

        # Base cols & fixes
        if "Ibs" in df.columns and "lbs" not in df.columns:
            df = df.rename(columns={"Ibs": "lbs"})
        base_cols = [c for c in ["TOP_VELO", "lbs", "DATE", "PREP_TIME"] if c in df.columns]
        fresh_cols = [c for c in df.columns if c.startswith("Fresh_")]
        post_cols  = [c for c in df.columns if c.startswith("Post_")]
        keep = base_cols + fresh_cols + post_cols
        if not keep:
            continue

        df = df[keep].copy()
        df.insert(0, "Player", player_display)
        df.insert(1, "PlayerKey", canon_key_last_first(player_display))

        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        if "PREP_TIME" in df.columns:
            df["PREP_TIME"] = df["PREP_TIME"].astype(str).str.strip()

        for c in df.columns:
            if c not in ("Player", "PlayerKey", "DATE", "PREP_TIME"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if "DATE" in out.columns:
        out = out.sort_values(["Player", "DATE"], na_position="last")
    return out


# -----------------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------------
st.title("Pitcher Reports (2025 Season)")
st.sidebar.header("Filters")

pitchers = sorted(season_df["Pitcher"].dropna().unique().tolist()) if "Pitcher" in season_df.columns else []
if not pitchers:
    st.error("No pitchers found. Check your CSV columns and filters.")
    st.stop()

pitcher_name = st.sidebar.selectbox("Select Pitcher:", pitchers, index=0)
heatmap_type = st.sidebar.selectbox("Select Heatmap Type:", ["Frequency", "Whiff", "Exit Velocity"])
batter_side = st.sidebar.selectbox("Select Batter Side:", ["Both", "Right", "Left"])
strikes = st.sidebar.selectbox("Select Strikes:", ["All", 0, 1, 2])
balls = st.sidebar.selectbox("Select Balls:", ["All", 0, 1, 2, 3])

st.sidebar.header("Date Filtering")
date_filter_option = st.sidebar.selectbox("Select Date Filter:", ["All", "Single Date", "Date Range"])
selected_date = None
start_date = None
end_date = None
if date_filter_option == "Single Date":
    selected_date = st.sidebar.date_input("Select a Date", value=datetime.today())
elif date_filter_option == "Date Range":
    date_range = st.sidebar.date_input("Select Date Range", value=[datetime.today(), datetime.today()])
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        st.sidebar.warning("Please select a valid date range.")

# --- Pitch Type filter (left sidebar)
st.sidebar.header("Pitch Types")

if "PitchType" in season_df.columns:
    # Limit options to the currently selected pitcher so the list stays tidy
    _pt_src = season_df
    if "Pitcher" in _pt_src.columns:
        _pt_src = _pt_src[_pt_src["Pitcher"] == pitcher_name]

    available_types = sorted(
        _pt_src["PitchType"].dropna().unique().tolist()
    )
else:
    available_types = []

# If nothing is available (edge case), keep an empty selection to avoid KeyErrors
if not available_types:
    selected_types = []
    st.sidebar.caption("No pitch types available for this selection.")
else:
    # Multiselect drives all visuals via filter_data()
    selected_types = st.sidebar.multiselect(
        "Include pitch types:",
        options=available_types,
        default=available_types,
        help="Filters all visuals in Pitch Flight Data by these pitch types."
    )

    # Quick actions
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("All", use_container_width=True):
        selected_types = available_types
    if col_b.button("None", use_container_width=True):
        selected_types = []


# --- Rolling view control (explicit, independent of the date filter above)
st.sidebar.header("Rolling View")
rolling_view_mode = st.sidebar.radio(
    "How should rolling charts be plotted?",
    ["Date-by-Date Rolling Averages", "Pitch-by-Pitch (Single Date)"],
    index=0,
    help="Date-by-Date shows daily averages over time. Pitch-by-Pitch shows sequential pitches for a single date."
)

# If user wants pitch-by-pitch but hasn't set Single Date above, give them a dedicated date picker here
pp_selected_date = None
if rolling_view_mode == "Pitch-by-Pitch (Single Date)":
    if date_filter_option == "Single Date" and selected_date:
        pp_selected_date = selected_date
    else:
        pp_selected_date = st.sidebar.date_input(
            "Select a date for Pitch-by-Pitch view",
            value=datetime.today()
        )


# -----------------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------------
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[df["Pitcher"] == pitcher_name] if "Pitcher" in df.columns else df.copy()
    if batter_side != "Both" and "BatterSide" in out.columns:
        out = out[out["BatterSide"] == batter_side]
    if strikes != "All" and "Strikes" in out.columns:
        out = out[out["Strikes"] == strikes]
    if balls != "All" and "Balls" in out.columns:
        out = out[out["Balls"] == balls]
    if date_filter_option == "Single Date" and selected_date:
        out = out[out["Date"].dt.date == pd.to_datetime(selected_date).date()]
    elif date_filter_option == "Date Range" and start_date and end_date:
        out = out[(out["Date"] >= pd.to_datetime(start_date)) & (out["Date"] <= pd.to_datetime(end_date))]
    if selected_types and "PitchType" in out.columns:
        out = out[out["PitchType"].isin(selected_types)]
    return out

filtered_df = filter_data(season_df)

# -----------------------------------------------------------------------------------
# Helpers for metrics
# -----------------------------------------------------------------------------------
def calculate_in_zone(df: pd.DataFrame) -> pd.DataFrame:
    # falls back if missing
    reqs = {"PlateLocHeight", "PlateLocSide"}
    if not reqs.issubset(df.columns):
        return df.iloc[0:0]
    return df[
        (df["PlateLocHeight"] >= 1.5) &
        (df["PlateLocHeight"] <= 3.3775) &
        (df["PlateLocSide"] >= -0.83083) &
        (df["PlateLocSide"] <= 0.83083)
    ]

def fmt_percent(x: float) -> str:
    return f"{round(x, 2)}%" if pd.notna(x) else "N/A"

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    percent_cols = ['InZone%', 'Swing%','Whiff%', 'SwStr%', 'Chase%', 'InZoneWhiff%', 'Pitch%', 'Hard%', 'Soft%', 'GB%', 'FB%', 'Contact%']
    for c in out.columns:
        if c in percent_cols:
            out[c] = out[c].apply(lambda v: fmt_percent(v) if isinstance(v, (int, float, np.floating)) else v)
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(1)
        else:
            out[c] = out[c].fillna("N/A")
    return out

# -----------------------------------------------------------------------------------
# Visuals
# -----------------------------------------------------------------------------------
def _render_two_cols(figs, header=None):
    if header:
        st.subheader(header)
    if not figs:
        st.info("No charts to display.")
        return
    for i in range(0, len(figs), 2):
        # if your Streamlit doesn’t support 'gap', just drop the arg
        c1, c2 = st.columns([1, 1], gap="small")
        with c1:
            st.plotly_chart(figs[i], use_container_width=True)
        if i + 1 < len(figs):
            with c2:
                st.plotly_chart(figs[i + 1], use_container_width=True)




def plot_heatmaps(map_type: str, grid=(60, 60), sigma=1.2, min_cell=4):
    """
    TruMedia-ish heatmaps via binned fields + Gaussian smoothing.
    map_type: 'Frequency' | 'Whiff' | 'Exit Velocity'
    grid:     (nx, ny) bins across the plotting window
    sigma:    Gaussian std (in cells); ~1.0–1.6 looks good
    min_cell: show blank if smoothed denom < min_cell (de-noise)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors

    df = filtered_df.copy()
    if df.empty:
        st.info("No data available for the selected parameters.")
        return

    need_cols = {"PlateLocSide","PlateLocHeight","PitchType","PitchCall"}
    if map_type == "Exit Velocity":
        need_cols.add("ExitSpeed")
    if not need_cols.issubset(df.columns):
        st.info(f"{map_type} heatmap needs columns: {', '.join(sorted(need_cols))}")
        return

    # --- plotting window (slightly larger than the zone) ---
    zx = 0.83083     # half width
    z_ymin, z_ymax = 1.5, 3.3775
    x_min, x_max = -1.25, 1.25
    y_min, y_max = 1.0, 4.0

    # keep rows with valid coords
    df = df.dropna(subset=["PlateLocSide","PlateLocHeight"]).copy()
    if df.empty:
        st.info("No data to plot after filtering.")
        return

    # swing flags for whiff
    swing_flags = {"StrikeSwinging","FoulBallFieldable","FoulBallNotFieldable","InPlay"}

    # --- small Gaussian blur (separable, no SciPy) ---
    def _gauss1d(s, r=None):
        if r is None: r = int(3*s)
        x = np.arange(-r, r+1, dtype=float)
        k = np.exp(-(x*x)/(2*s*s))
        k /= k.sum()
        return k

    def _blur2d(M, s=1.2):
        if M.size == 0: return M
        r = int(3*s)
        k = _gauss1d(s, r)
        pad = ((r, r), (r, r))
        A = np.pad(M, pad, mode="edge")
        # convolve columns
        tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis=0, arr=A)
        # convolve rows
        tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis=1, arr=tmp)
        return tmp[r:-r, r:-r]

    # --- helpers to make a smoothed grid ---
    def _hist2d(x, y, weights=None):
        H, xe, ye = np.histogram2d(
            x, y, bins=grid,
            range=[[x_min, x_max],[y_min, y_max]],
            weights=weights
        )
        # transpose so heatmap aligns with imshow(origin='lower')
        return H.T, xe, ye

    def _rate_smooth(num, den):
        num_b = _blur2d(num, sigma)
        den_b = _blur2d(den, sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(den_b >= min_cell, num_b / np.maximum(den_b, 1e-9), np.nan)
        return R, den_b

    # --- per-pitch-type panels ---
    pts = [pt for pt in df["PitchType"].dropna().unique()]
    n = len(pts)
    if n == 0:
        st.info("No pitch types to plot.")
        return
    cols = 3
    rows = int(np.ceil(n/cols))
    fig_w, fig_h = 5.6*cols, 5.6*rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    for ax, pt in zip(axes, pts):
        sub = df[df["PitchType"] == pt]
        x = sub["PlateLocSide"].to_numpy()
        y = sub["PlateLocHeight"].to_numpy()

        if map_type == "Frequency":
            count, xe, ye = _hist2d(x, y)
            count_b = _blur2d(count, sigma)
            # normalize to % of this pitch type
            pct = 100 * count_b / max(count_b.sum(), 1e-9)
            vmin, vmax = 0, np.nanpercentile(pct, 98)
            cmap = "inferno"
            Z = pct
            cbar_label = "% of pitches"

        elif map_type == "Whiff":
            swings = sub[sub["PitchCall"].isin(swing_flags)]
            whiffs = swings[swings["PitchCall"] == "StrikeSwinging"]
            den, xe, ye = _hist2d(swings["PlateLocSide"], swings["PlateLocHeight"])
            num, _, _ = _hist2d(whiffs["PlateLocSide"], whiffs["PlateLocHeight"])
            Z, den_b = _rate_smooth(num, den)
            vmin, vmax = 0.0, 0.6  # 0–60% typical
            cmap = "RdYlBu_r"
            cbar_label = "Whiff rate"

        else:  # Exit Velocity
            bip = sub[sub["PitchCall"] == "InPlay"].dropna(subset=["ExitSpeed"])
            den, xe, ye = _hist2d(bip["PlateLocSide"], bip["PlateLocHeight"])
            num, _, _ = _hist2d(bip["PlateLocSide"], bip["PlateLocHeight"], weights=bip["ExitSpeed"])
            Z_raw, den_b = _rate_smooth(num, den)
            Z = Z_raw
            vmin, vmax = 70, 100
            cmap = "magma"
            cbar_label = "Avg EV (mph)"

        # draw
        im = ax.imshow(
            Z, origin="lower", extent=[x_min, x_max, y_min, y_max],
            aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear"
        )

        # optional light contour for shape (skip when too sparse)
        if np.isfinite(Z).sum() > 50:
            try:
                cs = ax.contour(Z, levels=4, colors="k", alpha=0.15,
                                extent=[x_min, x_max, y_min, y_max])
            except Exception:
                pass

        # strike zone + 3x3 inner grid
        ax.add_patch(patches.Rectangle((-zx, z_ymin), 2*zx, z_ymax - z_ymin,
                                       fill=False, ec="black", lw=1.3))
        for fx in np.linspace(-zx, zx, 4)[1:-1]:
            ax.plot([fx, fx], [z_ymin, z_ymax], color="#000", alpha=0.2, lw=1)
        for fy in np.linspace(z_ymin, z_ymax, 4)[1:-1]:
            ax.plot([-zx, zx], [fy, fy], color="#000", alpha=0.2, lw=1)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([]); ax.set_yticks([])
        N = len(sub)
        ax.set_title(f"{pt} — {map_type}  (n={N})", fontsize=12)

        # tiny colorbar per panel (clean look)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=9)
        cb.set_label(cbar_label, size=9)

    # remove extra axes
    for j in range(len(pts), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{pitcher_name} • {map_type} Heatmaps", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    plt.close(fig)


def generate_pitch_traits_table():
    try:
        df = filtered_df.copy()
        if df.empty:
            st.info("No data available for the selected parameters.")
            return
        needed = {"PitchType", "RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "RelHeight", "RelSide", "Extension", "VertApprAngle"}
        if not needed.issubset(df.columns):
            st.info("Pitch Traits requires standard pitch metrics; some columns are missing.")
            return

        # Build the table entirely from the filtered slice
        grouped = (df
            .groupby("PitchType", dropna=False)
            .agg(
                Count=("PitchType", "size"),
                Velo=("RelSpeed", "mean"),
                iVB=("InducedVertBreak", "mean"),
                HB=("HorzBreak", "mean"),
                Spin=("SpinRate", "mean"),
                RelH=("RelHeight", "mean"),
                RelS=("RelSide", "mean"),
                Ext=("Extension", "mean"),
                VAA=("VertApprAngle", "mean"),
                StuffPlus=("StuffPlus", "mean")  # stays aligned with filters
            )
            .reset_index()
        )

        



        # sort by usage
        grouped = grouped.sort_values("Count", ascending=False)
        
        # weighted "All" row
        total = grouped["Count"].sum()
        
        def wavg(col):
            if col not in grouped.columns:
                return np.nan
            vals = grouped[col]
            mask = vals.notna()
            if not mask.any():
                return np.nan
            return np.average(vals[mask], weights=grouped.loc[mask, "Count"])
        
        all_row = {
            "PitchType": "All",
            "Count": int(total),
            "Velo": round(wavg("Velo"), 1),
            "iVB": round(wavg("iVB"), 1),
            "HB": round(wavg("HB"), 1),
            "Spin": round(wavg("Spin"), 1),
            "RelH": round(wavg("RelH"), 1),
            "RelS": round(wavg("RelS"), 1),
            "Ext": round(wavg("Ext"), 1),
            "VAA": round(wavg("VAA"), 1),
            "StuffPlus": round(wavg("StuffPlus"), 1)
        }
        grouped = pd.concat([grouped, pd.DataFrame([all_row])], ignore_index=True)


        display = grouped.rename(columns={"PitchType": "Pitch"})
        st.subheader("Pitch Traits")
        st.dataframe(format_dataframe(display))
    except Exception as e:
        st.error(f"An error occurred while generating the pitch traits table: {e}")

def generate_plate_discipline_table():
    try:
        df = filtered_df.copy()
        if df.empty:
            st.info("No data available for the selected parameters.")
            return
        needed = {"PitchType", "PitchCall", "Balls", "Strikes"}
        if not needed.issubset(df.columns):
            st.info("Plate Discipline needs PitchCall, Balls, Strikes.")
            return

        total_pitches = len(df)

        def calc_metrics(slice_df: pd.DataFrame) -> dict:
            in_zone = calculate_in_zone(slice_df)
            swing_flags = ["StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"]
            strike_flags = ["StrikeCalled", "FoulBallFieldable", "FoulBallNotFieldable", "StrikeSwinging", "InPlay"]
        
            fp_df = slice_df[(slice_df["Balls"] == 0) & (slice_df["Strikes"] == 0)]
            fp_total = len(fp_df)
            fp_strikes = fp_df[~fp_df["PitchCall"].isin(["HitByPitch", "BallCalled", "BallInDirt", "BallinDirt"])].shape[0]
            fp_strike_pct = (fp_strikes / fp_total * 100) if fp_total > 0 else 0
        
            swings = slice_df[slice_df["PitchCall"].isin(swing_flags)].shape[0]
            whiffs = slice_df[slice_df["PitchCall"] == "StrikeSwinging"].shape[0]
            chase = slice_df[(~slice_df.index.isin(in_zone.index)) & (slice_df["PitchCall"].isin(swing_flags))].shape[0]
            in_zone_whiffs = in_zone[in_zone["PitchCall"] == "StrikeSwinging"].shape[0]
            strikes_all = slice_df[slice_df["PitchCall"].isin(strike_flags)].shape[0]
            total = len(slice_df)
        
            return {
                "InZone%": (len(in_zone) / total * 100) if total else 0,
                "Swing%": (swings / total * 100) if total else 0,
                "Whiff%": (whiffs / swings * 100) if swings else 0,
                "SwStr%": (whiffs / total * 100) if total else 0,   # <-- new
                "Chase%": (chase / swings * 100) if swings else 0,
                "InZoneWhiff%": (in_zone_whiffs / len(in_zone) * 100) if len(in_zone) else 0,
                "Strike%": (strikes_all / total * 100) if total else 0,
                "FP Strike%": fp_strike_pct
            }


        grp = df.groupby("PitchType").apply(lambda x: pd.Series(calc_metrics(x))).reset_index()
        counts = df.groupby("PitchType")["PitchType"].count().rename("Count").reset_index()
        table = grp.merge(counts, on="PitchType", how="left")
        table["Pitch%"] = (table["Count"] / total_pitches * 100 if total_pitches else 0)

        # build "All"
        all_metrics = calc_metrics(df)
        all_row = {
            "PitchType": "All",
            "Count": total_pitches,
            "Pitch%": 100.0,
            **all_metrics
        }
        table = pd.concat([table, pd.DataFrame([all_row])], ignore_index=True)
        
        display = table.rename(columns={"PitchType": "Pitch"})
        st.subheader("Plate Discipline")
        cols = ["Pitch","Count","Pitch%","Strike%","InZone%","Swing%","Whiff%","SwStr%","Chase%","InZoneWhiff%","FP Strike%"]
        st.dataframe(format_dataframe(display[cols]))

    except Exception as e:
        st.error(f"An error occurred while generating the plate discipline table: {e}")

def generate_batted_ball_table():
    try:
        df = filtered_df.copy()
        if df.empty:
            st.info("No data available for the selected parameters.")
            return
        needed = {"PitchType", "PitchCall", "ExitSpeed", "Angle"}
        if not needed.issubset(df.columns):
            st.info("Batted Ball Summary needs ExitSpeed, Angle, PitchCall.")
            return

        def categorize_batted_type(angle):
            if pd.isna(angle):
                return np.nan
            angle = float(angle)
            if angle < 10:
                return "GroundBall"
            elif 10 <= angle < 25:
                return "LineDrive"
            elif 25 <= angle < 50:
                return "FlyBall"
            else:
                return "PopUp"

        df["BattedType"] = df["Angle"].apply(categorize_batted_type)
        bip = df[df["PitchCall"] == "InPlay"]

        # Build skeleton of all pitch types to avoid missing rows
        base = pd.DataFrame({"PitchType": sorted(df["PitchType"].unique())})

        agg = (bip.groupby("PitchType")
                  .agg(
                      BIP=("PitchCall", "size"),
                      GB=("BattedType", lambda x: (x == "GroundBall").sum()),
                      FB=("BattedType", lambda x: (x == "FlyBall").sum()),
                      EV=("ExitSpeed", "mean"),
                      Hard=("ExitSpeed", lambda x: (x >= 95).sum()),
                      Soft=("ExitSpeed", lambda x: (x < 95).sum()),
                  )
                  .reset_index())

        agg = base.merge(agg, on="PitchType", how="left").fillna(0)

        counts = df.groupby("PitchType")["PitchType"].count().rename("Count").reset_index()
        agg = agg.merge(counts, on="PitchType", how="left")

        # percentages
        with np.errstate(divide="ignore", invalid="ignore"):
            agg["GB%"] = np.where(agg["BIP"] > 0, agg["GB"] / agg["BIP"] * 100, 0.0)
            agg["FB%"] = np.where(agg["BIP"] > 0, agg["FB"] / agg["BIP"] * 100, 0.0)
            agg["Hard%"] = np.where(agg["BIP"] > 0, agg["Hard"] / agg["BIP"] * 100, 0.0)
            agg["Soft%"] = np.where(agg["BIP"] > 0, agg["Soft"] / agg["BIP"] * 100, 0.0)

        # contact%
        swing_flags = ["StrikeSwinging", "InPlay", "FoulBallNotFieldable", "FoulBallFieldable"]
        def contact_pct(slice_df):
            swings = slice_df[slice_df["PitchCall"].isin(swing_flags)].shape[0]
            contact = slice_df[slice_df["PitchCall"].isin(["InPlay","FoulBallNotFieldable","FoulBallFieldable"])].shape[0]
            return (contact / swings * 100) if swings else 0.0

        agg["Contact%"] = agg["PitchType"].map(lambda pt: contact_pct(df[df["PitchType"] == pt]))

        # all row
        all_row = {
            "PitchType": "All",
            "Count": len(df),
            "BIP": len(bip),
            "EV": bip["ExitSpeed"].mean() if len(bip) else 0.0,
            "GB%": (bip["BattedType"].eq("GroundBall").mean() * 100) if len(bip) else 0.0,
            "FB%": (bip["BattedType"].eq("FlyBall").mean() * 100) if len(bip) else 0.0,
            "Hard%": (bip["ExitSpeed"].ge(95).mean() * 100) if len(bip) else 0.0,
            "Soft%": (bip["ExitSpeed"].lt(95).mean() * 100) if len(bip) else 0.0,
            "Contact%": contact_pct(df),
        }
        agg = pd.concat([agg, pd.DataFrame([all_row])], ignore_index=True)

        display = agg.drop(columns=["GB","FB","Hard","Soft"], errors="ignore").rename(columns={"PitchType":"Pitch"})
        st.subheader("Batted Ball Summary")
        st.dataframe(format_dataframe(display))
    except Exception as e:
        st.error(f"Error generating batted ball table: {e}")

def plot_pitch_movement():
    try:
        df = filtered_df.copy()
        needed = {"PitchType", "InducedVertBreak", "HorzBreak", "RelSpeed"}
        if df.empty or not needed.issubset(df.columns):
            st.info("Pitch Movement needs iVB, HB, RelSpeed.")
            return

        # Keep only rows that have movement coordinates
        data = df.dropna(subset=["InducedVertBreak", "HorzBreak"]).copy()
        if data.empty:
            st.info("No pitch movement data available for plotting.")
            return

        # Ensure numeric + rounding for display
        for c in ["PitchNo","RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "RelHeight", "RelSide", "StuffPlus"]:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")

        # Build per-point hover fields (strings so we can show '—' when missing)
        def fmt1(x):
            return "—" if pd.isna(x) else f"{x:.1f}"

        # Date formatting for hover
        if "Date" in data.columns:
            data["Date_str"] = data["Date"].dt.strftime("%Y-%m-%d")
            data["Date_str"] = data["Date_str"].fillna("—")
        else:
            data["Date_str"] = "—"

        # Formatted fields
        data["PitchNo_disp"] = data["PitchNo"].apply(fmt1) if "PitchNo" in data.columns else "—"
        data["RelSpeed_disp"] = data["RelSpeed"].apply(fmt1) if "RelSpeed" in data.columns else "—"
        data["iVB_disp"]      = data["InducedVertBreak"].apply(fmt1)
        data["HB_disp"]       = data["HorzBreak"].apply(fmt1)
        data["Spin_disp"]     = data["SpinRate"].apply(fmt1) if "SpinRate" in data.columns else "—"
        data["Ext_disp"]     = data["Extension"].apply(fmt1) if "Extension" in data.columns else "—"
        data["RelH_disp"]    = data["RelHeight"].apply(fmt1) if "RelHeight" in data.columns else "-"
        data["RelS_disp"]    = data["RelSide"].apply(fmt1) if "RelSide" in data.columns else "-"
        data["SP_disp"]       = data["StuffPlus"].apply(lambda v: "—" if pd.isna(v) else f"{v:.1f}") if "StuffPlus" in data.columns else "—"

        # Plotly
        fig = go.Figure()
        for pt in data["PitchType"].dropna().unique():
            sub = data[data["PitchType"] == pt].copy()

            # Customdata order must match hovertemplate placeholders
            sub_customdata = np.stack([
                sub["Date_str"].values,
                sub["PitchNo_disp"].values,
                sub["RelSpeed_disp"].values,
                sub["iVB_disp"].values,
                sub["HB_disp"].values,
                sub["Spin_disp"].values,
                sub["Ext_disp"].values,
                sub["RelH_disp"].values,
                sub["RelS_disp"].values,
                sub["SP_disp"].values,   # StuffPlus (rounded .1)
            ], axis=-1)

            fig.add_trace(go.Scatter(
                x=sub["HorzBreak"], y=sub["InducedVertBreak"],
                mode="markers", name=pt,
                marker=dict(
                    size=9,
                    color=PLOTLY_COLORS.get(pt, "black"),
                    opacity=0.85,
                    line=dict(width=1, color="white")
                ),
                customdata=sub_customdata,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Date: %{customdata[0]}<br>"
                    "Pitch#: %{customdata[1]}<br>"
                    "Velo: %{customdata[2]} mph<br>"
                    "iVB: %{customdata[3]} in<br>"
                    "HB: %{customdata[4]} in<br>"
                    "Spin: %{customdata[5]} rpm<br>"
                    "Extension: %{customdata[6]} ft<br>"
                    "RelH: %{customdata[7]} ft<br>"
                    "RelS: %{customdata[8]} ft<br>"
                    "Stuff+: <b>%{customdata[9]}</b><extra></extra>"
                )
            ))

        # Crosshairs
        fig.add_shape(type="line", x0=0, x1=0, y0=-25, y1=25, line=dict(color="black", width=2), layer="below")
        fig.add_shape(type="line", x0=-25, x1=25, y0=0, y1=0, line=dict(color="black", width=2), layer="below")

        fig.update_xaxes(title="Horizontal Break (inches)", range=[-30, 30], zeroline=True, zerolinewidth=2, zerolinecolor="black")
        fig.update_yaxes(title="Induced Vertical Break (inches)", range=[-30, 30], zeroline=True, zerolinewidth=2, zerolinecolor="black")
        fig.update_layout(
            title=f"Pitch Movement for {pitcher_name}",
            template="plotly_white",
            legend_title="Pitch Type",
            width=900, height=700,
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred while generating the pitch movement graph: {e}")


def plot_release_and_approach_angles():
    try:
        df = filtered_df.copy()
        needed_release = {"PitchType", "HorzRelAngle", "VertRelAngle"}
        needed_approach = {"PitchType", "HorzApprAngle", "VertApprAngle"}

        def make_scatter(data, x_col, y_col, title, x_lim, y_lim):
            fig = go.Figure()
            for pt in data["PitchType"].unique():
                sub = data[data["PitchType"] == pt]
                mx, my = sub[x_col].mean(), sub[y_col].mean()
                sx, sy = sub[x_col].std(), sub[y_col].std()
                fig.add_trace(go.Scatter(
                    x=sub[x_col], y=sub[y_col], mode="markers", name=f"{pt} ({mx:.1f}, {my:.1f})",
                    marker=dict(size=8, color=PLOTLY_COLORS.get(pt, "black"), opacity=0.7)
                ))
                if not (pd.isna(mx) or pd.isna(my) or pd.isna(sx) or pd.isna(sy)):
                    r = max(sx, sy)
                    fig.add_shape(type="circle", xref="x", yref="y",
                                  x0=mx - r, y0=my - r, x1=mx + r, y1=my + r,
                                  line=dict(color=PLOTLY_COLORS.get(pt, "black"), width=2), opacity=0.3)
            fig.update_layout(title=title, xaxis=dict(title=x_col, range=x_lim),
                              yaxis=dict(title=y_col, range=y_lim),
                              template="plotly_white", showlegend=True, width=900, height=700)
            return fig

        if needed_release.issubset(df.columns):
            rel = df.dropna(subset=["HorzRelAngle", "VertRelAngle"])
            if not rel.empty:
                st.plotly_chart(make_scatter(rel, "HorzRelAngle", "VertRelAngle",
                                             "Release Angles by Pitch Type", [-7.5, 7.5], [-5, 3]),
                                use_container_width=True)

        if needed_approach.issubset(df.columns):
            appr = df.dropna(subset=["HorzApprAngle", "VertApprAngle"])
            if not appr.empty:
                st.plotly_chart(make_scatter(appr, "HorzApprAngle", "VertApprAngle",
                                             "Approach Angles by Pitch Type", [-6, 6], [-12, 0]),
                                use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred while generating the angle plots: {e}")

def _tight_layout(fig):
    fig.update_layout(autosize=True, margin=dict(l=20, r=10, t=60, b=40))
    return fig


def generate_rolling_line_graphs(view_mode: str, pitch_by_pitch_date=None):
    try:
        df = rolling_df.copy()
        if df.empty or "Pitcher" not in df.columns:
            st.info("No data available for rolling view.")
            return

        # keep only this pitcher; keep all dates for proper rolling context
        df = df[df["Pitcher"] == pitcher_name]
        if df.empty:
            st.info("No data for the selected pitcher.")
            return

        for c in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "StuffPlus"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = _ensure_date(df, "Date")
        if "PitchType" not in df.columns:
            df["PitchType"] = df.apply(canonical_pitch_type, axis=1)

        color_map = {pt: PLOTLY_COLORS.get(pt, "black") for pt in df["PitchType"].unique()}

        metrics = [
            ("RelSpeed", "Velocity"),
            ("InducedVertBreak", "iVB"),
            ("HorzBreak", "HB"),
            ("SpinRate", "Spin"),
            ("Extension", "Extension"),
        ]
        if "StuffPlus" in df.columns:
            metrics.append(("StuffPlus", "StuffPlus"))

        if view_mode == "Date-by-Date Rolling Averages":
            # daily means by pitch type
            roll = (df.groupby(["Date", "PitchType"])
                      .agg({m[0]: "mean" for m in metrics if m[0] in df.columns})
                      .reset_index()
                      .sort_values("Date"))

            figs = []
            for metric, label in metrics:
                if metric not in roll.columns:
                    continue
                fig = px.line(
                    roll, x="Date", y=metric, color="PitchType",
                    title=f"{label} Rolling Averages by Pitch Type (Date-by-Date)",
                    labels={"Date": "Date", metric: label, "PitchType": "Pitch Type"},
                    color_discrete_map=color_map, hover_data={"Date": "|%b %d, %Y", metric: ":.2f"},
                )
                # scatter overlay
                for pt in roll["PitchType"].unique():
                    sub = roll[roll["PitchType"] == pt]
                    fig.add_scatter(
                        x=sub["Date"], y=sub[metric], mode="markers",
                        marker=dict(size=6, color=color_map.get(pt, "black"), opacity=0.6),
                        name=f"{pt} points", showlegend=False
                    )
                # highlight active filter window, if any
                if date_filter_option == "Single Date" and selected_date:
                    xdt = pd.to_datetime(selected_date)
                    fig.add_vrect(x0=xdt, x1=xdt, fillcolor="gray", opacity=0.25, line_width=0)
                elif date_filter_option == "Date Range" and start_date and end_date:
                    sdt, edt = pd.to_datetime(start_date), pd.to_datetime(end_date)
                    fig.add_vrect(x0=sdt, x1=edt, fillcolor="gray", opacity=0.2, line_width=0)
            
                fig.update_layout(xaxis_title="Date", yaxis_title=label, legend_title="Pitch Type",
                                  template="plotly_white", hovermode="x unified")
                fig = _tight_layout(fig)
                figs.append(fig)
            
            _render_two_cols(figs, header="Rolling Averages Across Full Database (Date-by-Date)")


        else:  # Pitch-by-Pitch (Single Date)
            if pitch_by_pitch_date is None:
                st.info("Choose a date for Pitch-by-Pitch view in the sidebar.")
                return

            xdt = pd.to_datetime(pitch_by_pitch_date)
            day = df[df["Date"].dt.date == xdt.date()].copy()
            if day.empty:
                st.info(f"No data available for {xdt.strftime('%B %d, %Y')}.")
                return

            if "PitchNo" in day.columns:
                day["PitchNo"] = pd.to_numeric(day["PitchNo"], errors="coerce")
                day = day.dropna(subset=["PitchNo"]).sort_values("PitchNo")


            figs = []
            for metric, label in metrics:
                if metric not in day.columns:
                    continue
                fig = px.line(
                    day, x="PitchNo", y=metric, color="PitchType",
                    title=f"{label} Pitch-by-Pitch",
                    labels={"PitchNo": "Pitch #", metric: label, "PitchType": "Pitch Type"},
                    color_discrete_map=color_map, hover_data={"PitchNo": ":.0f", metric: ":.2f"},
                )
                for pt in day["PitchType"].unique():
                    sub = day[day["PitchType"] == pt]
                    fig.add_scatter(
                        x=sub["PitchNo"], y=sub[metric], mode="markers",
                        marker=dict(size=8, color=color_map.get(pt, "black")),
                        name=f"{pt} pts", showlegend=False
                    )
                fig.update_xaxes(range=[day["PitchNo"].min() - 1, day["PitchNo"].max() + 1])
                fig.update_layout(xaxis_title="Pitch #", yaxis_title=label, legend_title="Pitch Type",
                                  template="plotly_white", hovermode="x unified")
                fig = _tight_layout(fig)
                figs.append(fig)
            
            _render_two_cols(figs, header=f"Pitch-by-Pitch View for {xdt.strftime('%B %d, %Y')}")


    except Exception as e:
        st.error(f"An error occurred while generating rolling line graphs: {e}")


def _normalize_block(g: pd.DataFrame, cols: list, mode: str) -> pd.DataFrame:
    """
    Normalize selected columns within each PitchType group.
    mode: 'zscore', 'minmax', or 'none'
    """
    g = g.copy()
    if mode == "zscore":
        for c in cols:
            mu = g[c].mean()
            sd = g[c].std(ddof=0)
            g[f"{c}_norm"] = (g[c] - mu) / sd if (sd and not np.isnan(sd)) else g[c] * 0.0
    elif mode == "minmax":
        for c in cols:
            mn, mx = g[c].min(), g[c].max()
            rng = (mx - mn)
            g[f"{c}_norm"] = (g[c] - mn) / rng if (rng and not np.isnan(rng)) else g[c] * 0.0
    else:
        for c in cols:
            g[f"{c}_norm"] = g[c]
    return g

def _overlay_date_chart(df_in: pd.DataFrame, xcol: str, pitch_types: list, metrics_keys: list,
                        color_map: dict, norm_mode: str, window_label: str, title_suffix: str):
    """
    Build a single overlay chart for multiple metrics (Date-by-Date).
    If norm_mode == 'none' and len(metrics_keys) == 2, use dual axes. Else, overlay normalized to one axis.
    """
    import plotly.graph_objects as go
    gdf = df_in[df_in["PitchType"].isin(pitch_types)].copy()
    # Normalize within PitchType so lines are comparable per type
    blocks = []
    for pt, g in gdf.groupby("PitchType", as_index=False):
        blocks.append(_normalize_block(g, [f"{k}_roll" for k in metrics_keys], norm_mode))
    gdf = pd.concat(blocks, ignore_index=True) if blocks else gdf

    fig = go.Figure()

    # Dual-axis path (only if exactly two metrics and no normalization requested)
    if norm_mode == "none" and len(metrics_keys) == 2:
        m1, m2 = metrics_keys
        for pt in sorted(gdf["PitchType"].unique()):
            sub = gdf[gdf["PitchType"] == pt]
            fig.add_trace(go.Scatter(
                x=sub[xcol], y=sub[f"{m1}_roll"], mode="lines+markers",
                name=f"{pt} — {m1}", marker=dict(size=5),
                line=dict(width=2, color=color_map.get(pt, "black"))
            ))
            fig.add_trace(go.Scatter(
                x=sub[xcol], y=sub[f"{m2}_roll"], mode="lines+markers",
                name=f"{pt} — {m2}", marker=dict(size=5),
                line=dict(width=2, dash="dash"), yaxis="y2",
                marker_color=color_map.get(pt, "black")
            ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"{metrics_keys[0]} ({window_label})",
            yaxis2=dict(title=f"{metrics_keys[1]} ({window_label})", overlaying="y", side="right"),
            template="plotly_white", hovermode="x unified",
            title=f"Overlay: {metrics_keys[0]} & {metrics_keys[1]} — {title_suffix}"
        )
    else:
        # Single-axis with normalized values, legend includes metric label
        for pt in sorted(gdf["PitchType"].unique()):
            sub = gdf[gdf["PitchType"] == pt]
            for k in metrics_keys:
                fig.add_trace(go.Scatter(
                    x=sub[xcol], y=sub[f"{k}_roll_norm"], mode="lines+markers",
                    name=f"{pt} — {k}", marker=dict(size=5),
                    line=dict(width=2), marker_color=color_map.get(pt, "black")
                ))
        ylab = "Normalized Value" if norm_mode != "none" else f"Value ({window_label})"
        fig.update_layout(
            xaxis_title="Date", yaxis_title=ylab, template="plotly_white",
            hovermode="x unified", title=f"Overlay: {', '.join(metrics_keys)} — {title_suffix}"
        )

    # Shade selected date/range if the global date filter is set
    from pandas import to_datetime
    if date_filter_option == "Single Date" and selected_date:
        xdt = to_datetime(selected_date)
        fig.add_vrect(x0=xdt, x1=xdt, fillcolor="gray", opacity=0.25, line_width=0)
    elif date_filter_option == "Date Range" and start_date and end_date:
        sdt, edt = to_datetime(start_date), to_datetime(end_date)
        fig.add_vrect(x0=sdt, x1=edt, fillcolor="gray", opacity=0.2, line_width=0)

    return fig

def _overlay_pitch_chart(df_in: pd.DataFrame, xcol: str, pitch_types: list, metrics_keys: list,
                         color_map: dict, norm_mode: str, window_label: str, title_suffix: str):
    """
    Build a single overlay chart for multiple metrics (Pitch-by-Pitch).
    Same dual-axis vs normalized logic as above.
    """
    import plotly.graph_objects as go
    gdf = df_in[df_in["PitchType"].isin(pitch_types)].copy()
    blocks = []
    for pt, g in gdf.groupby("PitchType", as_index=False):
        blocks.append(_normalize_block(g, [f"{k}_roll" for k in metrics_keys], norm_mode))
    gdf = pd.concat(blocks, ignore_index=True) if blocks else gdf

    fig = go.Figure()

    if norm_mode == "none" and len(metrics_keys) == 2:
        m1, m2 = metrics_keys
        for pt in sorted(gdf["PitchType"].unique()):
            sub = gdf[gdf["PitchType"] == pt]
            fig.add_trace(go.Scatter(
                x=sub[xcol], y=sub[f"{m1}_roll"], mode="lines+markers",
                name=f"{pt} — {m1}", marker=dict(size=6),
                line=dict(width=2), marker_color=color_map.get(pt, "black")
            ))
            fig.add_trace(go.Scatter(
                x=sub[xcol], y=sub[f"{m2}_roll"], mode="lines+markers",
                name=f"{pt} — {m2}", marker=dict(size=6),
                line=dict(width=2, dash="dash"), yaxis="y2",
                marker_color=color_map.get(pt, "black")
            ))
        fig.update_layout(
            xaxis_title="Pitch #",
            yaxis_title=f"{metrics_keys[0]} ({window_label})",
            yaxis2=dict(title=f"{metrics_keys[1]} ({window_label})", overlaying="y", side="right"),
            template="plotly_white", hovermode="x unified",
            title=f"Overlay: {metrics_keys[0]} & {metrics_keys[1]} — {title_suffix}"
        )
    else:
        for pt in sorted(gdf["PitchType"].unique()):
            sub = gdf[gdf["PitchType"] == pt]
            for k in metrics_keys:
                fig.add_trace(go.Scatter(
                    x=sub[xcol], y=sub[f"{k}_roll_norm"], mode="lines+markers",
                    name=f"{pt} — {k}", marker=dict(size=6),
                    line=dict(width=2), marker_color=color_map.get(pt, "black")
                ))
        ylab = "Normalized Value" if norm_mode != "none" else f"Value ({window_label})"
        fig.update_layout(
            xaxis_title="Pitch #", yaxis_title=ylab, template="plotly_white",
            hovermode="x unified", title=f"Overlay: {', '.join(metrics_keys)} — {title_suffix}"
        )
    return fig

def render_rolling_average_charts_tab():
    """
    Flexible rolling charts:
      - Select multiple metrics
      - Choose Date-by-Date or Pitch-by-Pitch (Single Date)
      - Pick rolling window size
      - Filter by pitch types for the lines
      - Overlay multiple metrics on one chart (dual-axis if 2 metrics + no normalization,
        or normalized overlays for 2+ metrics)
    """
    try:
        df = rolling_df.copy()
        if df.empty or "Pitcher" not in df.columns:
            st.info("No data available for rolling charts.")
            return

        df = df[df["Pitcher"] == pitcher_name].copy()
        if df.empty:
            st.info("No data for the selected pitcher.")
            return

        # Ensure datatypes
        df = _ensure_date(df, "Date")
        if "PitchType" not in df.columns:
            df["PitchType"] = df.apply(canonical_pitch_type, axis=1)

        # Candidate metrics (only keep those that exist in df)
        metric_options_all = [
            ("RelSpeed", "Velocity"),
            ("InducedVertBreak", "iVB"),
            ("HorzBreak", "HB"),
            ("RelHeight", "RelH"),
            ("SpinRate", "Spin"),
            ("Extension", "Extension"),
            ("StuffPlus", "StuffPlus"),
            ("VertApprAngle", "VAA"),
            ("HorzApprAngle", "HAA"),
        ]
        present_metrics = [(k, lbl) for k, lbl in metric_options_all if k in df.columns]

        for k, _ in present_metrics:
            df[k] = pd.to_numeric(df[k], errors="coerce")

        st.subheader("Configure Rolling Average Charts")

        # Controls
        metrics_selected = st.multiselect(
            "Metrics",
            options=[f"{lbl} ({k})" for k, lbl in present_metrics],
            default=[f"{lbl} ({k})" for k, lbl in present_metrics[:3]]  # first three by default
        )
        # Map back to column keys
        metrics_selected_keys = [m.split("(")[-1].strip(")") for m in metrics_selected]

        view_mode = st.radio(
            "View Mode",
            ["Date-by-Date Rolling Averages", "Pitch-by-Pitch (Single Date)"],
            index=0,
            horizontal=False
        )

        # Pitch type filter inside the tab (independent of left sidebar)
        pt_available = sorted(df["PitchType"].dropna().unique().tolist())
        pt_selected = st.multiselect("Pitch Types to include", options=pt_available, default=pt_available)

        color_map = {pt: PLOTLY_COLORS.get(pt, "black") for pt in pt_available}

        # Overlay controls
        overlay_mode = st.radio(
            "Chart Layout",
            ["One chart per metric", "Overlay metrics in one chart"],
            index=0
        )

        norm_choice = "none"
        if overlay_mode == "Overlay metrics in one chart":
            norm_choice = st.selectbox(
                "Normalization for overlay",
                ["none (dual axis if 2 metrics)", "z-score (within pitch type)", "min-max [0,1] (within pitch type)"],
                index=1
            )
            norm_choice = {"none (dual axis if 2 metrics)": "none",
                           "z-score (within pitch type)": "zscore",
                           "min-max [0,1] (within pitch type)": "minmax"}[norm_choice]

        if not metrics_selected_keys:
            st.info("Select at least one metric to plot.")
            return

        if view_mode == "Date-by-Date Rolling Averages":
            window_days = st.slider("Rolling window (in days/rows)", min_value=1, max_value=30, value=7, step=1,
                                    help="Rolling window size over daily rows per pitch type.")

            # Daily mean by date & pitch type
            daily = (df.groupby(["Date", "PitchType"])
                       .agg({k: "mean" for k in metrics_selected_keys})
                       .reset_index()
                       .sort_values(["PitchType", "Date"]))

            daily = daily[daily["PitchType"].isin(pt_selected)]

            # Apply rolling per pitch type
            daily_roll = []
            for pt, g in daily.groupby("PitchType", as_index=False):
                g = g.sort_values("Date").copy()
                for k in metrics_selected_keys:
                    if k in g.columns:
                        g[k] = pd.to_numeric(g[k], errors="coerce")
                        g[f"{k}_roll"] = g[k].rolling(window=window_days, min_periods=1).mean()
                daily_roll.append(g)
            daily_roll = pd.concat(daily_roll, ignore_index=True) if daily_roll else pd.DataFrame()

            if overlay_mode == "Overlay metrics in one chart" and len(metrics_selected_keys) >= 2:
                fig = _overlay_date_chart(
                    daily_roll, "Date", pt_selected, metrics_selected_keys,
                    color_map, norm_choice, f"{window_days}-day", "Date-by-Date"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # One chart per metric (original behavior)
                for k in metrics_selected_keys:
                    colname = f"{k}_roll"
                    if colname not in daily_roll.columns:
                        continue
                    fig = px.line(
                        daily_roll, x="Date", y=colname, color="PitchType",
                        title=f"{k} — {window_days}-day Rolling Average",
                        labels={"Date": "Date", colname: k, "PitchType": "Pitch Type"},
                        color_discrete_map=color_map
                    )
                    for pt in daily_roll["PitchType"].unique():
                        sub = daily_roll[daily_roll["PitchType"] == pt]
                        fig.add_scatter(
                            x=sub["Date"], y=sub[colname], mode="markers",
                            marker=dict(size=6, color=color_map.get(pt, "black"), opacity=0.6),
                            name=f"{pt} points", showlegend=False
                        )
                    # Gray highlight from global date filter
                    if date_filter_option == "Single Date" and selected_date:
                        xdt = pd.to_datetime(selected_date)
                        fig.add_vrect(x0=xdt, x1=xdt, fillcolor="gray", opacity=0.25, line_width=0)
                    elif date_filter_option == "Date Range" and start_date and end_date:
                        sdt, edt = pd.to_datetime(start_date), pd.to_datetime(end_date)
                        fig.add_vrect(x0=sdt, x1=edt, fillcolor="gray", opacity=0.2, line_width=0)
                    fig.update_layout(template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

        else:
            # Pitch-by-pitch needs a single date + window size in pitches
            p_window = st.slider("Rolling window (in pitches)", min_value=1, max_value=50, value=5, step=1)
            p_date = st.date_input("Select a date for Pitch-by-Pitch view", value=datetime.today())
            xdt = pd.to_datetime(p_date)

            day = df[df["Date"].dt.date == xdt.date()].copy()
            if day.empty:
                st.info(f"No data for {xdt.strftime('%B %d, %Y')}.")
                return

            if "PitchNo" not in day.columns:
                st.info("This view requires a PitchNo column.")
                return

            day["PitchNo"] = pd.to_numeric(day["PitchNo"], errors="coerce")
            day = day.dropna(subset=["PitchNo"]).sort_values(["PitchType", "PitchNo"])
            day = day[day["PitchType"].isin(pt_selected)]

            # Rolling per PitchType over sequential pitches
            rolled = []
            for pt, g in day.groupby("PitchType", as_index=False):
                g = g.sort_values("PitchNo").copy()
                for k in metrics_selected_keys:
                    if k in g.columns:
                        g[k] = pd.to_numeric(g[k], errors="coerce")
                        g[f"{k}_roll"] = g[k].rolling(window=p_window, min_periods=1).mean()
                rolled.append(g)
            rolled = pd.concat(rolled, ignore_index=True) if rolled else pd.DataFrame()

            if overlay_mode == "Overlay metrics in one chart" and len(metrics_selected_keys) >= 2:
                fig = _overlay_pitch_chart(
                    rolled, "PitchNo", pt_selected, metrics_selected_keys,
                    color_map, norm_choice, f"{p_window}-pitch", xdt.strftime('%b %d, %Y')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                for k in metrics_selected_keys:
                    colname = f"{k}_roll"
                    if colname not in rolled.columns:
                        continue
                    fig = px.line(
                        rolled, x="PitchNo", y=colname, color="PitchType",
                        title=f"{k} — {p_window}-pitch Rolling Average ({xdt.strftime('%b %d, %Y')})",
                        labels={"PitchNo": "Pitch #", colname: k, "PitchType": "Pitch Type"},
                        color_discrete_map=color_map
                    )
                    for pt in rolled["PitchType"].unique():
                        sub = rolled[rolled["PitchType"] == pt]
                        fig.add_scatter(
                            x=sub["PitchNo"], y=sub[colname], mode="markers",
                            marker=dict(size=7, color=color_map.get(pt, "black"), opacity=0.7),
                            name=f"{pt} points", showlegend=False
                        )
                    fig.update_layout(template="plotly_white", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error in Rolling Average Charts: {e}")






# === RENDER ===
tab_flight, tab_biomech, tab_roll, tab_calc = st.tabs(
    ["Pitch Flight Data", "Workload/30-Day", "Rolling Average Charts", "Stuff+ Calculator"]
)


with tab_flight:
    plot_heatmaps(heatmap_type)
    generate_plate_discipline_table()
    generate_pitch_traits_table()
    generate_batted_ball_table()
    plot_pitch_movement()
    generate_rolling_line_graphs(rolling_view_mode, pitch_by_pitch_date=pp_selected_date)
    plot_release_and_approach_angles()

with tab_biomech:
    st.subheader("Workload / Pitch Counter")

    df_src = season_df.copy()
    if df_src.empty:
        st.info("No data available.")
        st.stop()

    df_src = _ensure_date(df_src, "Date")

    # -----------------------
    # Controls
    # -----------------------
    teams = sorted(df_src["PitcherTeam"].dropna().unique().tolist()) if "PitcherTeam" in df_src.columns else []
    default_team = TEAM_FILTER if TEAM_FILTER in teams else (teams[0] if teams else None)

    from datetime import timedelta
    c0, c1, c2, c3, c4 = st.columns([1.2, 0.9, 0.55, 0.55, 1.2])
    with c0:
        team_sel = st.selectbox("PitcherTeam", teams, index=(teams.index(default_team) if default_team in teams else 0)) if teams else None
    with c1:
        days_back = st.number_input("Window (days)", min_value=7, max_value=180, value=30, step=1)

    # window navigation
    data_max = df_src["Date"].max().date() if "Date" in df_src.columns and not df_src["Date"].isna().all() else datetime.today().date()
    if "wl_end_date" not in st.session_state:
        st.session_state.wl_end_date = data_max
    with c2:
        prev_btn = st.button("◀ Prev")
    with c3:
        next_btn = st.button("Next ▶")
    with c4:
        picked_end = st.date_input("End date", value=st.session_state.wl_end_date)
        st.session_state.wl_end_date = picked_end
    if prev_btn:
        st.session_state.wl_end_date = st.session_state.wl_end_date - timedelta(days=int(days_back))
    if next_btn:
        st.session_state.wl_end_date = st.session_state.wl_end_date + timedelta(days=int(days_back))

    if team_sel is None:
        st.info("No PitcherTeam found in data.")
        st.stop()

    end_date = st.session_state.wl_end_date
    start_date = end_date - timedelta(days=int(days_back) - 1)
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")

    mask = (df_src["PitcherTeam"] == team_sel) & (df_src["Date"].dt.date.between(start_date, end_date))
    df = df_src.loc[mask].copy()
    if df.empty:
        st.info(f"No rows for {team_sel} between {start_date} and {end_date}.")
        st.stop()

    # -----------------------
    # Daily P and IP
    # -----------------------
    grp_p = (df.assign(DateOnly=df["Date"].dt.date)
               .groupby(["Pitcher", "DateOnly"], dropna=False)
               .size().rename("P").reset_index())

    has_korbb = "KorBB" in df.columns
    has_playres = "PlayResult" in df.columns
    if has_korbb or has_playres:
        df["is_out"] = False
        if has_korbb:
            df.loc[df["KorBB"].astype(str).str.strip().str.lower() == "strikeout", "is_out"] = True
        if has_playres:
            df.loc[df["PlayResult"].astype(str).str.strip().isin(["Out", "Sacrifice"]), "is_out"] = True
        grp_outs = (df.assign(DateOnly=df["Date"].dt.date)
                      .groupby(["Pitcher", "DateOnly"], dropna=False)["is_out"]
                      .sum().rename("Outs").reset_index())
    else:
        grp_outs = pd.DataFrame(columns=["Pitcher", "DateOnly", "Outs"])

    daily = pd.merge(grp_p, grp_outs, on=["Pitcher", "DateOnly"], how="left")
    daily["Outs"] = pd.to_numeric(daily["Outs"], errors="coerce").fillna(0).astype(int)

    def outs_to_ip_str(outs: int) -> str:
        innings = outs // 3
        rem = outs % 3
        return f"{innings}.{rem}"

    daily["IP"] = daily["Outs"].apply(outs_to_ip_str)

    pitchers = sorted(df["Pitcher"].dropna().unique().tolist())

    # complete grid pitcher × date
    idx = pd.MultiIndex.from_product([pitchers, date_index.date], names=["Pitcher", "DateOnly"])
    base = pd.DataFrame(index=idx).reset_index()
    daily = base.merge(daily, on=["Pitcher", "DateOnly"], how="left")
    daily["P"] = daily["P"].fillna(0).astype(int)
    daily["Outs"] = daily["Outs"].fillna(0).astype(int)
    daily["IP"] = daily["IP"].fillna("").astype(str)

    # Totals for window
    tot_p = daily.groupby("Pitcher")["P"].sum()
    tot_outs = daily.groupby("Pitcher")["Outs"].sum()
    totals = pd.DataFrame({"Pitcher": pitchers,
                           "TotalP": tot_p.reindex(pitchers).fillna(0).astype(int).values,
                           "TotalIP": tot_outs.apply(outs_to_ip_str).reindex(pitchers).fillna("0.0").values})

    # -----------------------
    # EOW (End of Week) after each Sunday
    # -----------------------
    sundays = [d.date() for d in date_index if d.weekday() == 6]  # 6 = Sunday

    # precompute EOW P/IP for each (pitcher, sunday)
    eow_vals = {}
    for p in pitchers:
        p_rows = daily[daily["Pitcher"] == p]
        for s in sundays:
            week_start = max(start_date, s - timedelta(days=6))
            mask_week = (p_rows["DateOnly"] >= week_start) & (p_rows["DateOnly"] <= s)
            p_sum = int(p_rows.loc[mask_week, "P"].sum())
            outs_sum = int(p_rows.loc[mask_week, "Outs"].sum())
            eow_vals[(p, s)] = (p_sum, outs_sum)

    # Build column sequence: [(kind, date)] where kind in {"day", "eow"}
    col_seq = []
    for d in date_index:
        col_seq.append(("day", d.date()))
        if d.weekday() == 6:  # add EOW after Sunday
            col_seq.append(("eow", d.date()))

    # -----------------------
    # Render HTML table
    # -----------------------
    from collections import OrderedDict
    # month spans need to include both day and eow columns
    month_spans = OrderedDict()
    for kind, d in col_seq:
        key = pd.Timestamp(d).strftime("%B %Y")
        month_spans.setdefault(key, 0)
        month_spans[key] += 2  # each entry contributes 2 columns (P, IP)

    def dow(d): return pd.Timestamp(d).strftime("%a")

    # ===== Appearance (tweak here) =====
    name_col_width = 180
    tot_col_width  = 96
    day_cell_w     = 32    # << narrower P/IP cells
    row_h          = 28

    highlight_name = (pitcher_name or "").strip().lower()

    styles = f"""
    <style>
      .scroll-x {{
        position: relative;            /* make this the sticky containing block */
        overflow-x: auto;
        border: 1px solid #e5e7eb; border-radius: 10px;
      }}
    
      /* Safari-friendly: no collapse when using sticky left */
      table.workload {{
        border-collapse: separate;     /* <-- changed from collapse */
        border-spacing: 0;             /* keep grid look */
        width: max-content;
        table-layout: fixed;
      }}
    
      table.workload th, table.workload td {{
        border: 1px solid #e6e6e6;
        padding: 3px 4px;
        text-align: center;
        font-size: 12px;
        height: {row_h}px;
        white-space: nowrap;
        background-clip: padding-box;  /* avoids bleed under sticky edges */
      }}
    
      /* sticky headers */
      table.workload thead th {{ position: sticky; top: 0; background: #f8fafc; z-index: 2; }}
      table.workload thead tr:nth-child(2) th {{ top: 26px; }}
      table.workload thead tr:nth-child(3) th {{ top: 52px; }}
    
      /* sticky left columns (both th & td) */
      .sticky-left   {{ position: sticky; left: 0px;  background: #fff; z-index: 3; text-align: left; font-weight: 600; }}
      .sticky-left-2 {{ position: sticky; left: {name_col_width}px; background: #fff; z-index: 3; }}
      .sticky-left-3 {{ position: sticky; left: {name_col_width + tot_col_width}px; background: #fff; z-index: 3; }}
    
      /* ensure header versions sit above month/day headers */
      table.workload thead th.sticky-left,
      table.workload thead th.sticky-left-2,
      table.workload thead th.sticky-left-3 {{ z-index: 6; }}
    
      /* totals styling */
      thead .tot-head {{ background: #eef6ff; font-weight: 700; }}
      tbody td.tot    {{ background: #f5faff; font-weight: 700; }}
    
      /* EOW styling */
      thead .eow-head {{ background: #fff0e6; }}
      tbody td.eow    {{ background: #fff7f0; font-weight: 600; }}
    
      tbody tr:nth-child(even) {{ background: #fcfcfc; }}
      tbody tr.hl {{ background: #fff7cc !important; font-weight: 700; }}
    
      /* FIX: give both th and td identical, fixed widths */
      .name  {{ width: {name_col_width}px; min-width: {name_col_width}px; max-width: {name_col_width}px; }}
      .totcol{{ width: {tot_col_width}px;  min-width: {tot_col_width}px;  max-width: {tot_col_width}px;  }}
      .daycol{{ width: {day_cell_w}px;     min-width: {day_cell_w}px;     max-width: {day_cell_w}px;     }}
    </style>
    """


    # Header rows
    h1 = (
        f"<tr>"
        f"<th class='sticky-left name' rowspan='3'>Pitcher</th>"
        f"<th class='sticky-left-2 tot-head totcol' rowspan='3' title='Total pitches in window'>Total P ({int(days_back)}d)</th>"
        f"<th class='sticky-left-3 tot-head totcol' rowspan='3' title='Total innings pitched in window'>Total IP ({int(days_back)}d)</th>"
    )

    for mon, span in month_spans.items():
        h1 += f"<th colspan='{span}'>{mon}</th>"
    h1 += "</tr>"

    h2 = "<tr>"
    for kind, d in col_seq:
        if kind == "day":
            h2 += f"<th colspan='2'>{pd.Timestamp(d).day}<br>{dow(d)}</th>"
        else:
            # End of Week label (after Sun)
            h2 += f"<th class='eow-head' colspan='2' title='Totals Mon–Sun ending {d}'>EOW<br>{pd.Timestamp(d).strftime('%b %d')}</th>"
    h2 += "</tr>"

    h3 = "<tr>"
    for kind, _ in col_seq:
        if kind == "day":
            h3 += "<th>P</th><th>IP</th>"
        else:
            h3 += "<th class='eow-head'>P</th><th class='eow-head'>IP</th>"
    h3 += "</tr>"

    # mapping for quick daily lookups
    daily_key = {(r["Pitcher"], r["DateOnly"]): (int(r["P"]), r["IP"]) for _, r in daily.iterrows()}

    # order rows by TotalP desc
    totals_sorted = totals.sort_values(["TotalP", "Pitcher"], ascending=[False, True])

    # Body rows
    body_rows = []
    for _, rr in totals_sorted.iterrows():
        p_name = rr["Pitcher"]
        rowcls = "hl" if p_name and p_name.strip().lower() == highlight_name else ""
        tr = f"<tr class='{rowcls}'>"
        tr += f"<td class='name sticky-left'>{p_name}</td>"
        tr += f"<td class='tot sticky-left-2 totcol'>{int(rr['TotalP'])}</td>"
        tr += f"<td class='tot sticky-left-3 totcol'>{rr['TotalIP']}</td>"

        for kind, d in col_seq:
            if kind == "day":
                P, IP = daily_key.get((p_name, d), (0, ""))
                tr += f"<td class='daycol'>{P if P>0 else ''}</td>"
                tr += f"<td class='daycol'>{IP}</td>"
            else:
                P_week, Outs_week = eow_vals.get((p_name, d), (0, 0))
                tr += f"<td class='daycol eow'>{P_week if P_week>0 else ''}</td>"
                tr += f"<td class='daycol eow'>{outs_to_ip_str(Outs_week) if Outs_week>0 else ''}</td>"
        tr += "</tr>"
        body_rows.append(tr)

    table_html = styles + "<div class='scroll-x'><table class='workload'>"
    table_html += f"<thead>{h1}{h2}{h3}</thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.caption(
        "Totals reflect the selected window. EOW = totals Mon–Sun ending on Sunday columns. "
        "IP heuristic: counts **Strikeout** in `KorBB` and **Out/Sacrifice** in `PlayResult` as 1 out each; "
        "IP shown in baseball decimals (e.g., 1.2 = 5 outs)."
    )




with tab_roll:
    render_rolling_average_charts_tab()


with tab_calc:
    # Try to pass the currently selected pitcher's most-used pitch type as default
    default_pt = None
    if "PitchType" in filtered_df.columns and not filtered_df.empty:
        counts = (filtered_df.groupby("PitchType")["PitchType"].size().sort_values(ascending=False))
        if not counts.empty:
            default_pt = counts.index[0]
    render_stuff_calculator_tab(season_df, default_pitcher=pitcher_name, default_pitch_type=default_pt)



