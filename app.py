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
    "Changeup": "hotpink",
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

# after
def _apply_game_type_filter(df: pd.DataFrame, choice: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if choice and "game_type" in df.columns and choice != "All":
        return df[df["game_type"] == choice]
    return df






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
# --- Streamlit UI (put this BEFORE applying the filter) ---
st.title("Pitcher Reports (2025 Season)")
st.sidebar.header("Filters")

# ---- Sidebar: Game Type (multi-select) ----
st.sidebar.subheader("Game Type")

def _game_types_in(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "game_type" not in df.columns:
        return []
    vals = (
        df["game_type"]
        .astype(str).str.strip()
        .replace({"nan": None})
        .dropna()
        .unique()
        .tolist()
    )
    # keep a preferred ordering if present
    preferred = ["LBP", "IS", "FallGame"]
    ordered = [g for g in preferred if g in vals] + [g for g in vals if g not in preferred]
    return ordered

# collect options from both frames so nothing disappears if only one has a value
_all_types = sorted(set(_game_types_in(season_df)) | set(_game_types_in(rolling_df)))

# Default to IS + FallGame (omit LBP). If none detected, show all.
_default_types = [t for t in _all_types if t in {"IS", "FallGame"}] or _all_types

game_type_choices = st.sidebar.multiselect(
    "Show pitches from (leave empty for All):",
    options=_all_types,
    default=_default_types,
    help="Filters by 'game_type'. Select multiple. Leave empty to show all."
)

def _apply_game_type_filter_multi(df: pd.DataFrame, choices: list[str]) -> pd.DataFrame:
    """
    If 'choices' is empty or 'game_type' column missing, returns df unchanged.
    Else keeps rows where game_type is in choices.
    """
    if df is None or df.empty:
        return df
    if "game_type" not in df.columns:
        return df
    if not choices:  # treat empty selection as "All"
        return df

    norms = set(str(s).strip() for s in choices)
    col = df["game_type"].astype(str).str.strip()
    return df.loc[col.isin(norms)].copy()


season_df  = _apply_game_type_filter_multi(season_df,  game_type_choices)
rolling_df = _apply_game_type_filter_multi(rolling_df, game_type_choices)





pitchers = sorted(season_df["Pitcher"].dropna().unique().tolist()) if "Pitcher" in season_df.columns else []
if not pitchers:
    st.error("No pitchers found. Check your CSV columns and filters.")
    st.stop()

pitcher_name = st.sidebar.selectbox("Select Pitcher:", pitchers, index=0)
heatmap_type = st.sidebar.selectbox(
    "Select Heatmap Type:",
    ["Frequency", "Swing Rate", "Whiff Rate", "Exit Velocity", "wOBA", "xwOBA"]
)

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




# ================= TruMedia-style HEATMAPS v2 (bigger + smoother + shared scales) =================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---- separable gaussian (no SciPy) ----
def _gauss_kernel_1d(sigma: float, truncate: float = 3.5) -> np.ndarray:
    sigma = max(1e-6, float(sigma))
    r = max(1, int(round(truncate * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k

def _conv1d_reflect(img2d: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    pad = len(k) // 2
    if axis == 0:
        apad = np.pad(img2d, ((pad, pad), (0, 0)), mode="reflect")
        out = np.empty_like(img2d, dtype=np.float64)
        # vectorized via dot on rolling windows is overkill; simple loop is fast enough here
        for j in range(apad.shape[1]):
            out[:, j] = np.convolve(apad[:, j], k, mode="valid")
        return out
    else:
        apad = np.pad(img2d, ((0, 0), (pad, pad)), mode="reflect")
        out = np.empty_like(img2d, dtype=np.float64)
        for i in range(apad.shape[0]):
            out[i, :] = np.convolve(apad[i, :], k, mode="valid")
        return out

def _gaussian_blur_2d(img2d: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img2d
    k = _gauss_kernel_1d(sigma)
    return _conv1d_reflect(_conv1d_reflect(img2d, k, axis=0), k, axis=1)

def _strike_zone(ax):
    zone_w = 1.66166
    rect = patches.Rectangle((-zone_w/2, 1.5), zone_w, 3.3775 - 1.5,
                             linewidth=2.2, edgecolor="black", facecolor="none")
    ax.add_patch(rect)
    # faint inner grid
    for y in np.linspace(1.5, 3.3775, 5)[1:-1]:
        ax.plot([-zone_w/2, zone_w/2], [y, y], color="k", alpha=0.12, lw=1)
    for x in np.linspace(-zone_w/2, zone_w/2, 5)[1:-1]:
        ax.plot([x, x], [1.5, 3.3775], color="k", alpha=0.12, lw=1)

def _axes_style(ax, xlim, ylim, title=None):
    ax.set_facecolor("white")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=13, weight="bold")

def _build_grid(df, xcol, ycol, xlim, ylim, bins):
    H, *_ = np.histogram2d(df[xcol].values, df[ycol].values,
                           bins=[bins, bins], range=[xlim, ylim], density=False)
    return H.T.astype(np.float64)

def plot_heatmaps(map_type: str):
    """
    Heatmaps by PitchType over plate location.

    Modes:
      - Frequency            : % of this pitch type (per panel scale, feet-based sigma)  <-- restored
      - Swing Rate           : swings / pitches            (ratio-of-smooths)
      - Whiff Rate           : whiffs / swings             (ratio-of-smooths)
      - Exit Velocity        : mean EV on BIP              (ratio-of-smooths, weighted sums)
      - wOBA                 : mean wOBA_value/_result     (ratio-of-smooths)
      - xwOBA                : mean xwOBA_value/_result    (ratio-of-smooths)
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import Normalize

        pitcher_data = filtered_df.copy()
        if pitcher_data.empty:
            st.info("No data available for the selected parameters.")
            return

        needed = {"PlateLocSide", "PlateLocHeight", "PitchType"}
        if not needed.issubset(pitcher_data.columns):
            st.info("Heatmap needs PlateLocSide, PlateLocHeight, PitchType columns.")
            return

        # --- plate extents (keep as your app uses) ---
        x_min, x_max = -2.0, 2.0
        y_min, y_max =  1.0, 4.0

        # strike zone
        zone_w = 1.66166
        z_x0 = -zone_w / 2
        z_y0 = 1.5
        z_h  = 3.3775 - 1.5

        # Common grid
        width_ft  = (x_max - x_min)
        height_ft = (y_max - y_min)

        # =========================
        # FREQUENCY — RESTORED BEHAVIOR
        # =========================
        if map_type == "Frequency":
            # Your previous “feel”: high grid density in ft + sigma in ft, panel-by-panel vmax
            GRID_PER_FT = 140
            SIGMA_FT_BASE = 0.20
            SIGMA_FT_TINYBOOST = 0.35
            TINY_N = 8
            CMAP = "turbo"
            GLOBAL_VMAX_Q = 0.97   # per-panel here, not shared

            NX = max(64, int(width_ft  * GRID_PER_FT))
            NY = max(64, int(height_ft * GRID_PER_FT))
            x_edges = np.linspace(x_min, x_max, NX + 1)
            y_edges = np.linspace(y_min, y_max, NY + 1)

            # smoothing (sigma in pixels derived from feet)
            try:
                from scipy.ndimage import gaussian_filter
                def _smooth(H, n):
                    sx_px = max(1.0, SIGMA_FT_BASE * (NX / width_ft))
                    sy_px = max(1.0, SIGMA_FT_BASE * (NY / height_ft))
                    if n <= TINY_N:
                        sx_px += SIGMA_FT_TINYBOOST * (NX / width_ft)
                        sy_px += SIGMA_FT_TINYBOOST * (NY / height_ft)
                    return gaussian_filter(H, sigma=(sx_px, sy_px), mode="nearest")
            except Exception:
                def _smooth(H, n):
                    k = int(max(3, round((SIGMA_FT_BASE * (NX / width_ft)) * 3)))
                    ker = np.ones((k, k), dtype=float); ker /= ker.sum()
                    pad = k // 2
                    P = np.pad(H, pad, mode="edge")
                    out = np.zeros_like(H)
                    for i in range(H.shape[0]):
                        for j in range(H.shape[1]):
                            out[i, j] = np.sum(P[i:i+k, j:j+k] * ker)
                    return out

            base = pitcher_data.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
            if base.empty:
                st.info("No data available to plot after filtering.")
                return

            pts = base["PitchType"].dropna().unique().tolist()

            # layout
            N_COLS = 3
            PANEL_IN = 5.5
            n_types = len(pts)
            n_rows = int(np.ceil(n_types / N_COLS))
            fig, axes = plt.subplots(n_rows, N_COLS, figsize=(PANEL_IN * N_COLS, PANEL_IN * n_rows),
                                     constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for ax in axes: ax.axis("off")

            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

            for i, pt in enumerate(pts):
                sub = base[base["PitchType"] == pt]
                n = len(sub)
                if n == 0:
                    continue

                xs = sub["PlateLocSide"].values
                ys = sub["PlateLocHeight"].values

                H, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
                H = (H / max(n, 1)) * 100.0  # percent of that pitch type
                Hs = _smooth(H.T, n=n).T     # smooth

                ax = axes[i]; ax.axis("on")
                # per-panel normalization (NO shared vmax)
                vmax_i = np.percentile(Hs[np.isfinite(Hs)], GLOBAL_VMAX_Q * 100.0) if np.isfinite(Hs).any() else 1.0
                norm_i = Normalize(vmin=0.0, vmax=max(1e-9, float(vmax_i)))

                im = ax.imshow(Hs.T, origin="lower", extent=extent, cmap=CMAP, norm=norm_i,
                               interpolation="bilinear", aspect="equal")

                # light raw points
                ax.scatter(sub["PlateLocSide"], sub["PlateLocHeight"], s=6, c="white", alpha=0.25, linewidths=0)

                # strike zone
                ax.add_patch(patches.Rectangle((z_x0, z_y0), zone_w, z_h,
                                               edgecolor="black", facecolor="none", linewidth=2))

                ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{pt} — Frequency (n={n})", fontsize=11, pad=6)

                cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
                cbar.ax.set_ylabel("% of pitches", rotation=90, labelpad=8)
                cbar.ax.tick_params(labelsize=8)

            # hide unused axes
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            fig.suptitle(f"{pitcher_name} • Frequency Heatmaps", fontsize=16, y=0.995)
            st.pyplot(fig)
            plt.close(fig)
            return

        # =========================
        # NON-FREQUENCY — ACCURATE RATIO MAPS
        # =========================
        # Coarser, stable bin sigma in *bins* for robust ratios
        NX, NY = 120, 120
        x_edges = np.linspace(x_min, x_max, NX + 1)
        y_edges = np.linspace(y_min, y_max, NY + 1)
        CMAP = "turbo"
        MIN_DEN = 1.0  # minimum smoothed denominator to color a bin

        try:
            from scipy.ndimage import gaussian_filter
            def _smooth(img):
                return gaussian_filter(img, sigma=2.0, mode="nearest")
        except Exception:
            def _smooth(img):
                k = 5
                ker = np.ones((k, k), dtype=float) / (k * k)
                pad = k // 2
                P = np.pad(img, pad, mode="edge")
                out = np.zeros_like(img)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        out[i, j] = np.sum(P[i:i+k, j:j+k] * ker)
                return out

        def hist2d(x, y, w=None):
            H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
            return H

        def ratio_of_smooths(num, den):
            num_s = _smooth(num)
            den_s = _smooth(den)
            with np.errstate(divide="ignore", invalid="ignore"):
                R = np.where(den_s >= MIN_DEN, num_s / den_s, np.nan)
            return R

        base = pitcher_data.dropna(subset=["PlateLocSide", "PlateLocHeight"]).copy()
        if base.empty:
            st.info("No data available to plot after filtering.")
            return

        # Flags
        SWING = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        WHIFF = "StrikeSwinging"
        BIP   = "InPlay"

        # Determine metric
        key = map_type
        val_col = None
        cbar_label = ""
        title_suffix_fmt = ""

        if map_type == "Swing Rate":
            key = "swing"; cbar_label = "%"; title_suffix_fmt = "Swing Rate (swings={sw}, n={n})"
        elif map_type == "Whiff Rate":
            key = "whiff"; cbar_label = "%"; title_suffix_fmt = "Whiff Rate (whiffs={wf}, swings={sw})"
        elif map_type == "Exit Velocity":
            key = "ev"; cbar_label = "mph"; title_suffix_fmt = "Exit Velocity (BIP={bip})"
            if "ExitSpeed" not in base.columns:
                st.info("No ExitSpeed column available for Exit Velocity heatmap.")
                return
            val_col = "ExitSpeed"
        elif map_type == "wOBA":
            key = "woba"; cbar_label = "wOBA"; title_suffix_fmt = "wOBA (n={nn})"
            val_col = "wOBA_value" if "wOBA_value" in base.columns else ("wOBA_result" if "wOBA_result" in base.columns else None)
            if val_col is None:
                st.info("No wOBA_value (or wOBA_result) column available.")
                return
        elif map_type == "xwOBA":
            key = "xwoba"; cbar_label = "xwOBA"; title_suffix_fmt = "xwOBA (n={nn})"
            val_col = "xwOBA_value" if "xwOBA_value" in base.columns else ("xwOBA_result" if "xwOBA_result" in base.columns else None)
            if val_col is None:
                st.info("No xwOBA_value (or xwOBA_result) column available.")
                return
        else:
            st.info("Unknown heatmap type.")
            return

        fields = []
        vmax_pool = []
        vmin_pool = []

        pts = base["PitchType"].dropna().unique().tolist()
        for pt in pts:
            sub = base[base["PitchType"] == pt].copy()
            xs = sub["PlateLocSide"].values
            ys = sub["PlateLocHeight"].values
            n = len(sub)

            if key == "swing":
                if "PitchCall" not in sub.columns:
                    continue
                swings_mask = sub["PitchCall"].isin(SWING)
                den = hist2d(xs, ys)                           # pitches
                num = hist2d(xs[swings_mask], ys[swings_mask]) # swings
                grid = ratio_of_smooths(num, den) * 100.0
                title_suffix = title_suffix_fmt.format(sw=int(swings_mask.sum()), n=n)

            elif key == "whiff":
                if "PitchCall" not in sub.columns:
                    continue
                swings_mask = sub["PitchCall"].isin(SWING)
                whiff_mask  = sub["PitchCall"].eq(WHIFF)
                den = hist2d(xs[swings_mask], ys[swings_mask]) # swings
                num = hist2d(xs[whiff_mask],  ys[whiff_mask])  # whiffs
                grid = ratio_of_smooths(num, den) * 100.0
                title_suffix = title_suffix_fmt.format(wf=int(whiff_mask.sum()), sw=int(swings_mask.sum()))

            elif key == "ev":
                if "PitchCall" not in sub.columns:
                    continue
                bip_mask = sub["PitchCall"].eq(BIP)
                if not bip_mask.any():
                    grid = np.full((NX, NY), np.nan)
                else:
                    v = sub.loc[bip_mask, val_col].astype(float).values
                    num = hist2d(xs[bip_mask], ys[bip_mask], w=v)  # sum EV
                    den = hist2d(xs[bip_mask], ys[bip_mask])       # BIP count
                    grid = ratio_of_smooths(num, den)              # mean EV
                title_suffix = title_suffix_fmt.format(bip=int(bip_mask.sum()))

            elif key in ("woba", "xwoba"):
                vals = sub[val_col].astype(float)
                mask = vals.notna()
                if not mask.any():
                    grid = np.full((NX, NY), np.nan)
                else:
                    v = vals[mask].values
                    num = hist2d(xs[mask], ys[mask], w=v)  # sum metric
                    den = hist2d(xs[mask], ys[mask])       # count
                    grid = ratio_of_smooths(num, den)      # mean
                title_suffix = title_suffix_fmt.format(nn=int(mask.sum()))
            else:
                continue

            fields.append((pt, grid, title_suffix))

            finite_vals = grid[np.isfinite(grid)]
            if finite_vals.size:
                vmax_pool.append(np.percentile(finite_vals, 97))
                vmin_pool.append(np.percentile(finite_vals, 3))

        if not fields:
            st.info("Nothing to plot with the current filters.")
            return

        # Shared scale for non-frequency so panels are comparable
        vmin = float(np.nanmin(vmin_pool)) if len(vmin_pool) else 0.0
        vmax = float(np.nanmax(vmax_pool)) if len(vmax_pool) else 1.0
        norm = Normalize(vmin=vmin, vmax=max(vmin + 1e-9, vmax))

        # layout
        N_COLS = 3
        PANEL_IN = 5.5
        n_types = len(fields)
        n_rows = int(np.ceil(n_types / N_COLS))
        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(PANEL_IN * N_COLS, PANEL_IN * n_rows),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax in axes: ax.axis("off")

        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        for i, (pt, grid, title_suffix) in enumerate(fields):
            ax = axes[i]; ax.axis("on")
            im = ax.imshow(grid.T, origin="lower", extent=extent, cmap="turbo", norm=norm,
                           interpolation="bilinear", aspect="equal")

            sub = base[base["PitchType"] == pt]
            ax.scatter(sub["PlateLocSide"], sub["PlateLocHeight"], s=6, c="white", alpha=0.25, linewidths=0)

            ax.add_patch(patches.Rectangle((z_x0, z_y0), zone_w, z_h, edgecolor="black", facecolor="none", linewidth=2))
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{pt} — {title_suffix}", fontsize=11, pad=6)

            cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
            cbar.ax.set_ylabel(cbar_label, rotation=90, labelpad=8)
            cbar.ax.tick_params(labelsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{pitcher_name} • {map_type} Heatmaps", fontsize=16, y=0.995)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Error generating heatmaps: {e}")



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

        # ensure numeric for the result metrics if present
        for c in ["wOBA_result", "xwOBA_result"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        def categorize_batted_type(angle):
            if pd.isna(angle):
                return np.nan
            a = float(angle)
            if a < 10: return "GroundBall"
            if a < 25: return "LineDrive"
            if a < 50: return "FlyBall"
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

        # Add wOBA / xwOBA means (use all rows where the metric is non-null)
        if "wOBA_result" in df.columns:
            woba_by = df.groupby("PitchType")["wOBA_result"].mean().rename("wOBA").reset_index()
            agg = agg.merge(woba_by, on="PitchType", how="left")
        else:
            agg["wOBA"] = np.nan

        if "xwOBA_result" in df.columns:
            xwoba_by = df.groupby("PitchType")["xwOBA_result"].mean().rename("xwOBA").reset_index()
            agg = agg.merge(xwoba_by, on="PitchType", how="left")
        else:
            agg["xwOBA"] = np.nan

        # === ROUNDING FIX (do this right after computing/merging) ===
        if "wOBA" in agg.columns:
            agg["wOBA"] = agg["wOBA"].round(3)
        if "xwOBA" in agg.columns:
            agg["xwOBA"] = agg["xwOBA"].round(3)

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

        # "All" row (round here too)
        all_row = {
            "PitchType": "All",
            "Count": len(df),
            "BIP": len(bip),
            "EV": bip["ExitSpeed"].mean() if len(bip) else 0.0,
            "GB%": (bip["BattedType"].eq("GroundBall").mean() * 100) if len(bip) else 0.0,
            "FB%": (bip["BattedType"].eq("FlyBall").mean() * 100) if len(bip) else 0.0,
            "Hard%": (bip["ExitSpeed"].ge(95).mean() * 100) if ("ExitSpeed" in bip.columns and len(bip)) else 0.0,
            "Soft%": (bip["ExitSpeed"].lt(95).mean() * 100) if ("ExitSpeed" in bip.columns and len(bip)) else 0.0,
            "Contact%": contact_pct(df),
            "wOBA": round(df["wOBA_result"].mean(), 3) if "wOBA_result" in df.columns else np.nan,
            "xwOBA": round(df["xwOBA_result"].mean(), 3) if "xwOBA_result" in df.columns else np.nan,
        }
        agg = pd.concat([agg, pd.DataFrame([all_row])], ignore_index=True)

        # tidy display + put wOBA/xwOBA at the end
        display = (agg
                   .drop(columns=["GB","FB","Hard","Soft"], errors="ignore")
                   .rename(columns={"PitchType":"Pitch"}))

        end_cols = ["wOBA", "xwOBA"]
        cols = [c for c in display.columns if c not in end_cols] + [c for c in end_cols if c in display.columns]

        # OPTIONAL: force fixed 3-decimal text (e.g., 0.340 not 0.34)
        for c in ["wOBA", "xwOBA"]:
             if c in display.columns:
                 display[c] = display[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        st.subheader("Batted Ball Summary")

        st.dataframe(format_dataframe(display[cols]))
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

def generate_by_date_overall_table():
    """
    One row per Date (date-only).
    By default, IGNORE the sidebar date filter and show all outings for the current pitcher
    (and other active filters). A toggle lets you switch back to the date-filtered slice.
    """
    try:
        # --- UI toggle (default ON) ---
        show_all = st.checkbox(
            "Show all outings (ignore date filter)",
            value=True,
            help="When ON, this table ignores the sidebar date selection and shows every outing in the season file."
        )

        # --- Build the source slice ---
        if show_all:
            # Start from the full season_df and apply all filters EXCEPT date
            df = season_df.copy()
            if df.empty or "Date" not in df.columns:
                st.info("No data available for By-Date table.")
                return

            # Apply the same non-date filters as filter_data()
            if "Pitcher" in df.columns:
                df = df[df["Pitcher"] == pitcher_name]
            if batter_side != "Both" and "BatterSide" in df.columns:
                df = df[df["BatterSide"] == batter_side]
            if strikes != "All" and "Strikes" in df.columns:
                df = df[df["Strikes"] == strikes]
            if balls != "All" and "Balls" in df.columns:
                df = df[df["Balls"] == balls]
            if selected_types and "PitchType" in df.columns:
                df = df[df["PitchType"].isin(selected_types)]
        else:
            # Respect the full sidebar filter set (including date) by using the global filtered_df
            df = filtered_df.copy()

        if df.empty or "Date" not in df.columns:
            st.info("No data available for By-Date table.")
            return

        # Ensure datatypes
        if "StuffPlus" in df.columns:
            df["StuffPlus"] = pd.to_numeric(df["StuffPlus"], errors="coerce")
        if "ExitSpeed" in df.columns:
            df["ExitSpeed"] = pd.to_numeric(df["ExitSpeed"], errors="coerce")
        if "Angle" in df.columns:
            df["Angle"] = pd.to_numeric(df["Angle"], errors="coerce")

        # Batted-ball categorization
        def _batted_type(angle):
            if pd.isna(angle): return np.nan
            a = float(angle)
            if a < 10: return "GroundBall"
            if a < 25: return "LineDrive"
            if a < 50: return "FlyBall"
            return "PopUp"

        df["BattedType"] = df["Angle"].apply(_batted_type) if "Angle" in df.columns else np.nan

        # Date-only key
        df["DateOnly"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df[df["DateOnly"].notna()]
        if df.empty:
            st.info("No valid dates to display.")
            return

        # Flags
        swing_flags  = ["StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"]
        strike_flags = ["StrikeCalled", "FoulBallFieldable", "FoulBallNotFieldable", "StrikeSwinging", "InPlay"]

        # Aggregate per Date
        rows = []
        for d, day in df.groupby("DateOnly"):
            total = len(day)
            if total == 0: continue

            in_zone = calculate_in_zone(day) if {"PlateLocHeight","PlateLocSide"}.issubset(day.columns) else day.iloc[0:0]
            swings = day["PitchCall"].isin(swing_flags).sum()
            whiffs = (day["PitchCall"] == "StrikeSwinging").sum()
            in_zone_whiffs = in_zone[in_zone["PitchCall"] == "StrikeSwinging"].shape[0] if not in_zone.empty else 0
            strikes_all = day["PitchCall"].isin(strike_flags).sum()
            bip_df = day[day["PitchCall"] == "InPlay"]

            # First-pitch strike
            if {"Balls","Strikes"}.issubset(day.columns):
                fp_df = day[(day["Balls"] == 0) & (day["Strikes"] == 0)]
                fp_total = len(fp_df)
                fp_strikes = fp_df[~fp_df["PitchCall"].isin(["HitByPitch", "BallCalled", "BallInDirt", "BallinDirt"])].shape[0]
                fp_strike_pct = (fp_strikes / fp_total * 100) if fp_total > 0 else 0.0
            else:
                fp_strike_pct = 0.0

            # Contact% (of swings)
            contact = day["PitchCall"].isin(["InPlay","FoulBallNotFieldable","FoulBallFieldable"]).sum()
            contact_pct = (contact / swings * 100) if swings else 0.0

            # BIP splits
            gb_pct = (bip_df["BattedType"].eq("GroundBall").mean() * 100) if len(bip_df) else 0.0
            fb_pct = (bip_df["BattedType"].eq("FlyBall").mean() * 100) if len(bip_df) else 0.0
            hard_pct = (bip_df["ExitSpeed"].ge(95).mean() * 100) if ("ExitSpeed" in bip_df.columns and len(bip_df)) else 0.0
            soft_pct = (bip_df["ExitSpeed"].lt(95).mean() * 100) if ("ExitSpeed" in bip_df.columns and len(bip_df)) else 0.0

            # Stuff+ mean (use all pitches that day; fallback safe)
            stuff_mean = float(day["StuffPlus"].mean()) if "StuffPlus" in day.columns and day["StuffPlus"].notna().any() else np.nan

            # --- PA-aware daily wOBA/xwOBA ---
            # A PA ends on: InPlay, HBP, Walk, Strikeout (when those columns exist)
            pa_mask = (
                day["PitchCall"].isin(["InPlay", "HitByPitch"])
                | (day["KorBB"].isin(["Walk", "Strikeout"]) if "KorBB" in day.columns else False)
            )
            pa_n = int(pa_mask.sum())
            
            woba_mean = float(day["wOBA_result"].sum() / pa_n) if ("wOBA_result" in day.columns and pa_n > 0) else np.nan
            xwoba_mean = float(day["xwOBA_result"].sum() / pa_n) if ("xwOBA_result" in day.columns and pa_n > 0) else np.nan


            rows.append({
                "Date": d,
                "Pitches": int(total),
                "BIP": int(len(bip_df)),
                "Strike%": (strikes_all / total * 100),
                "InZone%": (len(in_zone) / total * 100) if total and not in_zone.empty else 0.0,
                "Swing%": (swings / total * 100),
                "SwStr%": (whiffs / total * 100),
                "InZoneWhiff%": (in_zone_whiffs / len(in_zone) * 100) if (not in_zone.empty and len(in_zone) > 0) else 0.0,
                "FP Strike%": fp_strike_pct,
                "Stuff+": round(stuff_mean, 1) if pd.notna(stuff_mean) else np.nan,
                "Contact%": contact_pct,
                "GB%": gb_pct,
                "FB%": fb_pct,
                "Soft%": soft_pct,
                "Hard%": hard_pct,
                "wOBA": round(woba_mean, 3) if pd.notna(woba_mean) else np.nan,
                "xwOBA": round(xwoba_mean, 3) if pd.notna(xwoba_mean) else np.nan,
            })

        if not rows:
            st.info("No rows to display for By-Date table after filtering.")
            return

        out = pd.DataFrame(rows).sort_values("Date")
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")


        # Format wOBA/xwOBA for display as 0.000 (not 0.0)
        for col in ["wOBA", "xwOBA"]:
            if col in out.columns:
                out[col] = out[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "0.000")

        st.subheader("By-Date (Overall) — Pitch Flight Data")
        cols = ["Date","Pitches","BIP","Strike%","InZone%","Swing%","SwStr%","InZoneWhiff%","FP Strike%","Stuff+","Contact%","GB%","FB%","Soft%","Hard%","wOBA","xwOBA"]
        st.dataframe(format_dataframe(out[cols]), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error generating By-Date table: {e}")


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

        for c in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension", "StuffPlus", "RelHeight", "RelSide"]:
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
            ("RelHeight", "RelH"),   # <-- add
            ("RelSide", "RelS"),     # <-- add
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


from datetime import datetime, timedelta




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
    generate_by_date_overall_table()
    plot_release_and_approach_angles()

with tab_biomech:
    st.subheader("Workload / Pitch Counter")

    df_src = season_df.copy()
    if df_src.empty:
        st.info("No data available.")
        st.stop()


        # --- Warm-up Pitches: load from Google Sheets (cached) ---
    @st.cache_data(show_spinner=False)
    def load_warmups_from_sheet(sheet_url: str) -> pd.DataFrame:
        """
        Reads a public Google Sheet with columns: Date, Pitcher, Count.
        Returns a DataFrame with columns: Pitcher (str), DateOnly (date), WU (int).
        """
        try:
            # Convert a normal "edit" URL to a CSV export URL
            # e.g., https://docs.google.com/spreadsheets/d/<ID>/export?format=csv
            if "/edit" in sheet_url:
                base = sheet_url.split("/edit", 1)[0]
                csv_url = base + "/export?format=csv"
            elif "/view" in sheet_url:
                base = sheet_url.split("/view", 1)[0]
                csv_url = base + "/export?format=csv"
            else:
                # if it's already an export or ends with the ID, try appending export
                csv_url = sheet_url.rstrip("/") + "/export?format=csv"
    
            wu = pd.read_csv(csv_url)
        except Exception:
            # Fallback: let user upload a CSV with same columns if the URL isn't public
            wu = pd.DataFrame(columns=["Date", "Pitcher", "Count"])
    
        # Normalize columns
        cmap = {c.lower().strip(): c for c in wu.columns}
        need = {"date", "pitcher", "count"}
        if not need.issubset(set(cmap.keys())):
            return pd.DataFrame(columns=["Pitcher", "DateOnly", "WU"])
    
        wu = wu.rename(columns={cmap["date"]: "Date",
                                cmap["pitcher"]: "Pitcher",
                                cmap["count"]: "Count"})
        wu["DateOnly"] = pd.to_datetime(wu["Date"], errors="coerce").dt.date
        wu["Pitcher"]  = wu["Pitcher"].astype(str).str.strip()
        wu["WU"]       = pd.to_numeric(wu["Count"], errors="coerce").fillna(0).astype(int)
    
        # Aggregate in case there are multiple rows per pitcher/date
        wu = (wu.groupby(["Pitcher", "DateOnly"], as_index=False)["WU"].sum())
        return wu[["Pitcher", "DateOnly", "WU"]]


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
    
    # Warm-up data (Google Sheet)
    WARMUP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1xa9GUjw3SJ0GMU61ZMWLlp7mM4SY6BRqCVjVRoa80AE/edit?usp=sharing"
    warmups_df = load_warmups_from_sheet(WARMUP_SHEET_URL)


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

    totals_sorted = totals.sort_values(["TotalP", "Pitcher"], ascending=[False, True])

        # Ensure warmups are limited to the window + pitchers in scope
    if warmups_df is not None and not warmups_df.empty:
        wu_masked = warmups_df[
            (warmups_df["DateOnly"].between(start_date, end_date)) &
            (warmups_df["Pitcher"].isin(pitchers))
        ].copy()
    else:
        wu_masked = pd.DataFrame(columns=["Pitcher", "DateOnly", "WU"])
    
    # Merge WU into the daily grid (default 0)
    daily = daily.merge(wu_masked, on=["Pitcher", "DateOnly"], how="left")
    daily["WU"] = daily["WU"].fillna(0).astype(int)


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
    
    # month spans need to include all subcolumns:
    # - day: P, IP, WU (3)
    # - eow: P, IP (2)
    month_spans = OrderedDict()
    for kind, d in col_seq:
        key = pd.Timestamp(d).strftime("%B %Y")
        month_spans.setdefault(key, 0)
        month_spans[key] += (3 if kind == "day" else 2)
    
    def dow(d): return pd.Timestamp(d).strftime("%a")
    
    # ===== Appearance additions =====
    name_col_width = 180
    tot_col_width  = 96
    day_cell_w     = 32
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

        /* Warm-up styling */
      thead .wu-head {{ background: #efeaff; }}
      tbody td.wu     {{ background: #f7f3ff; font-weight: 600; font-size: 11px; }}

    
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
    
    # Row 2: per-day/EOW headers
    h2 = "<tr>"
    for kind, d in col_seq:
        if kind == "day":
            h2 += f"<th colspan='3'>{pd.Timestamp(d).day}<br>{dow(d)}</th>"
        else:
            h2 += f"<th class='eow-head' colspan='2' title='Totals Mon–Sun ending {d}'>EOW<br>{pd.Timestamp(d).strftime('%b %d')}</th>"
    h2 += "</tr>"
    
    # Row 3: subheaders (P, IP, WU for days; P, IP for EOW)
    h3 = "<tr>"
    for kind, _ in col_seq:
        if kind == "day":
            h3 += "<th>P</th><th>IP</th><th class='wu-head' title='Warm-up pitches (not in totals/EOW)'>WU</th>"
        else:
            h3 += "<th class='eow-head'>P</th><th class='eow-head'>IP</th>"
    h3 += "</tr>"

    # mapping for quick daily lookups (now includes WU)
    daily_key = {(r["Pitcher"], r["DateOnly"]): (int(r["P"]), r["IP"], int(r["WU"])) for _, r in daily.iterrows()}
    
    # Body rows (unchanged order/sorting)
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
                P, IP, WU = daily_key.get((p_name, d), (0, "", 0))
                tr += f"<td class='daycol'>{P if P>0 else ''}</td>"
                tr += f"<td class='daycol'>{IP}</td>"
                tr += f"<td class='daycol wu'>{WU if WU>0 else ''}</td>"
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
     # ==== VALD (ForceDecks) section — robust name matching (dual keys, suffix tolerant) ====
    st.markdown("---")
    st.subheader("VALD (ForceDecks)")
    
    @st.cache_data(show_spinner=False)
    def load_vald_df():
        # common locations
        cand = ["/mnt/data/VALD_Master.csv", "data/vald/VALD_Master.csv", "VALD_Master.csv"]
        path = next((p for p in cand if os.path.exists(p)), None)
        if not path:
            return pd.DataFrame(), None, None, None
    
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
    
        # ---- NAME (prefer Name_LastFirst; fallback to Name) ----
        name_col = "Name_LastFirst" if "Name_LastFirst" in df.columns else (
            "Name" if "Name" in df.columns else None
        )
        if not name_col:
            return pd.DataFrame(), None, None, None
    
        # ---- DATETIME (combine Date + Time if present) ----
        # create a single datetime column VALD_DateTime and also a date-only VALD_Date
        if "Date" in df.columns and "Time" in df.columns:
            dt_raw = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
            dt = pd.to_datetime(dt_raw, errors="coerce", infer_datetime_format=True)
        elif "Date" in df.columns:
            dt = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
        else:
            dt = pd.NaT
    
        df["VALD_DateTime"] = dt
        df["VALD_Date"] = pd.to_datetime(df["VALD_DateTime"]).dt.date
        df = df[df["VALD_DateTime"].notna()].copy()
    
        # ---- FRESH/POST detection & normalization ----
        # 1) Start with FP_Tag (P/F), map to Fresh/Post
        def _map_fp(x: str):
            s = str(x or "").strip().lower()
            if s in ("f", "fresh"): return "Fresh"
            if s in ("p", "post"):  return "Post"
            return np.nan
    
        df["FP_Tag_clean"] = np.nan
        if "FP_Tag" in df.columns:
            df["FP_Tag_clean"] = df["FP_Tag"].map(_map_fp)
    
        # 2) If still missing, try Tags column (Fresh/Post strings)
        if "Tags" in df.columns:
            tags_norm = df["Tags"].astype(str).str.strip().str.lower()
            df.loc[df["FP_Tag_clean"].isna() & tags_norm.eq("fresh"), "FP_Tag_clean"] = "Fresh"
            df.loc[df["FP_Tag_clean"].isna() & tags_norm.eq("post"),  "FP_Tag_clean"] = "Post"
    
        # If nothing found, leave NaN (UI lets you choose "Both" anyway)
    
        # ---- robust dual keys for matching to season_df ----
        import re
        SUFFIXES = {"jr","sr","ii","iii","iv","v"}
        def _letters_only(s: str) -> str:
            return re.sub(r"[^A-Za-z]", "", s or "").lower()
        def _strip_suffix_token(tok: str) -> bool:
            w = re.sub(r"[^\w]", "", (tok or "")).lower()
            return w in SUFFIXES
    
        def key_last_first(name: str) -> str:
            s = str(name or "").strip()
            if "," in s:
                last, first = [p.strip() for p in s.split(",", 1)]
            else:
                parts = s.split()
                if len(parts) >= 2:
                    last, first = parts[0], " ".join(parts[1:])
                else:
                    last, first = s, ""
            lp = [p for p in last.split()  if not _strip_suffix_token(p)]
            fp = [p for p in first.split() if not _strip_suffix_token(p)]
            return _letters_only("".join(lp) + "".join(fp))
    
        def key_first_last(name: str) -> str:
            s = str(name or "").strip().replace(",", " ")
            parts = [p for p in s.split() if not _strip_suffix_token(p)]
            if len(parts) >= 2:
                last = parts[-1]; first = " ".join(parts[:-1])
            else:
                last, first = s, ""
            return _letters_only(last + first)
    
        df["VALD_Player"] = df[name_col].astype(str).str.strip()
        df["Key_LF"] = df["VALD_Player"].apply(key_last_first)
        df["Key_FL"] = df["VALD_Player"].apply(key_first_last)
    
        # Make common metric columns numeric (safe convert)
        for c in df.columns:
            if c in (name_col, "VALD_Player", "Key_LF", "Key_FL", "VALD_DateTime", "VALD_Date", "FP_Tag_clean"):
                continue
            df[c] = pd.to_numeric(df[c], errors="ignore")
    
        # tell caller which columns to use
        vald_date_col = "VALD_DateTime"     # used by tables + charts
        vald_fp_col   = "FP_Tag_clean"      # used for filtering + line_dash
        return df, name_col, vald_date_col, vald_fp_col
    
        
        
    vald_df, vald_name_col, vald_date_col, vald_fp_col = load_vald_df()
    
    if vald_df is None or vald_df.empty:
        st.info("No VALD file found or it is empty. Place `VALD_Master.csv` in `/mnt/data/` or `data/vald/`.")
    else:
        # ---- build season pitcher key map with both variants ----
        import re as _re
        all_pitchers = sorted(season_df["Pitcher"].dropna().unique().tolist())
    
        def season_key_first_last(p):
            s = _re.sub(r"\s+", " ", str(p).strip().replace(",", " "))
            parts = s.split()
            if len(parts) >= 2:
                last = parts[-1]
                first = " ".join(parts[:-1])
            else:
                last, first = s, ""
            return _re.sub(r"[^A-Za-z]", "", (last + first)).lower()
    
        key_to_pitcher = {}
        for p in all_pitchers:
            for k in {canon_key_last_first(p), season_key_first_last(p)}:
                if k:
                    key_to_pitcher.setdefault(k, p)
    
        # ---- match VALD rows by either key ----
        match_mask = vald_df["Key_LF"].isin(key_to_pitcher) | vald_df["Key_FL"].isin(key_to_pitcher)
        vald_cut = vald_df.loc[match_mask].copy()
    
        # Helpful debug expander
        with st.expander("VALD name matching debug", expanded=False):
            st.write(f"Pitchers in season_df: {len(all_pitchers)}")
            st.write(f"VALD rows: {len(vald_df)}  •  matched: {len(vald_cut)}")
            if len(vald_cut) < len(vald_df):
                sample_unmatched = (
                    vald_df.loc[~match_mask, [vald_name_col, "Key_LF", "Key_FL"]]
                           .drop_duplicates()
                           .head(20)
                )
                st.write("Unmatched (sample):")
                st.dataframe(sample_unmatched, use_container_width=True)
    
        if vald_cut.empty:
            st.warning("No VALD rows matched any pitcher names after robust normalization.")
        else:
            # Map matched rows to canonical season pitcher names
            def to_pitcher(row):
                return key_to_pitcher.get(row["Key_LF"]) or key_to_pitcher.get(row["Key_FL"]) or row["VALD_Player"]
    
            vald_cut["Pitcher"] = vald_cut.apply(to_pitcher, axis=1)
    
            # ---------- Controls ----------
            c1, c2, c3 = st.columns([1.2, 0.9, 1.7])
            with c1:
                view_mode = st.radio(
                    "Table view",
                    ["Cumulative Average", "Individual Tests"],
                    index=0,
                    horizontal=True,
                    help="Cumulative Average aggregates in the selected date range (and F/P filter) per player."
                )
            with c2:
                if vald_fp_col:
                    fp_opt = st.selectbox("Fresh/Post filter", ["Both","F","P","Fresh","Post"], index=0)
                else:
                    fp_opt = "Both"
            with c3:
                maxd = pd.to_datetime(vald_cut[vald_date_col].max()) if vald_date_col else None
                mind = pd.to_datetime(vald_cut[vald_date_col].min()) if vald_date_col else None
                if pd.notna(mind) and pd.notna(maxd):
                    default_start = maxd - pd.Timedelta(days=90)
                    date_range = st.date_input("Date range",
                                               value=[default_start.date(), maxd.date()],
                                               min_value=mind.date(), max_value=maxd.date())
                else:
                    date_range = None
    
            work = vald_cut.copy()
            if vald_fp_col and fp_opt != "Both":
                want_fresh = fp_opt.lower().startswith("f")
                mask = work[vald_fp_col].str.lower().isin(["f","fresh"]) if want_fresh else work[vald_fp_col].str.lower().isin(["p","post"])
                work = work[mask]
            if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                sdt, edt = [pd.to_datetime(d) for d in date_range]
                work = work[(work[vald_date_col] >= sdt) & (work[vald_date_col] <= edt)]
    
            # Identify numeric metric columns
            non_metric = {vald_name_col, "VALD_Player", "Key_LF", "Key_FL", "Pitcher", vald_date_col}
            if vald_fp_col: non_metric.add(vald_fp_col)
            metric_cols = [c for c in work.columns if c not in non_metric and pd.api.types.is_numeric_dtype(work[c])]
    
            # Shorten headers like "Concentric Mean Force [N] (L)" -> "CMF[N](L)"
            import re
            def short_header(col: str) -> str:
                s = col.strip()
                m = re.match(r"^\s*([^\[\(]+)\s*(\[[^\]]+\])?\s*(\([^\)]+\))?\s*$", s)
                if not m: return s
                phrase, units, side = m.group(1) or "", m.group(2) or "", m.group(3) or ""
                words = re.findall(r"[A-Za-z]+", phrase)
                initials = "".join(w[0].upper() for w in words if w)
                return f"{(initials or phrase.strip().replace(' ', ''))}{units or ''}{side or ''}"
    
            short_map, used = {}, set()
            for c in metric_cols:
                base = short_header(c)
                alias = base
                k = 2
                while alias in used:
                    alias = f"{base}_{k}"
                    k += 1
                short_map[c] = alias
                used.add(alias)
    
            # ---------- Table ----------
            if view_mode == "Individual Tests":
                disp = work[["Pitcher", vald_date_col] + ([vald_fp_col] if vald_fp_col else []) + metric_cols].copy()
            else:
                agg = (work.groupby("Pitcher", as_index=False)
                            .agg({vald_date_col: "max", **{c: "mean" for c in metric_cols}}))
                disp = agg[["Pitcher", vald_date_col] + metric_cols].copy()
                if vald_fp_col:
                    mode_fp = (work.groupby("Pitcher")[vald_fp_col]
                                   .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)).reset_index()
                    disp = disp.merge(mode_fp, on="Pitcher", how="left")
                    disp = disp[["Pitcher", vald_date_col, vald_fp_col] + metric_cols]
    
            col_cfg = {
                "Pitcher": st.column_config.Column("Pitcher", help="Matched by robust normalization (First/Last variants; suffix tolerant)"),
                vald_date_col: st.column_config.DateColumn(vald_date_col, help="Test date")
            }
            if vald_fp_col:
                col_cfg[vald_fp_col] = st.column_config.Column(vald_fp_col, help="Fresh/Post label")
    
            rename_map = {c: short_map[c] for c in metric_cols}
            disp = disp.rename(columns=rename_map)
            for short, full in {short_map[c]: c for c in metric_cols}.items():
                col_cfg[short] = st.column_config.NumberColumn(short, help=full, format="%.2f")
    
            st.markdown("**VALD Tests Table**")
            st.dataframe(disp.sort_values(["Pitcher", vald_date_col]),
                         use_container_width=True, hide_index=True, column_config=col_cfg)
    
            # ---------- Rolling charts ----------
            st.markdown("**Rolling charts**")
            rc1, rc2, rc3 = st.columns([1.2, 1.5, 0.9])
            with rc1:
                p_for_chart = st.selectbox(
                    "Player",
                    options=sorted(disp["Pitcher"].dropna().unique().tolist()),
                    index=(sorted(disp["Pitcher"].dropna().unique().tolist()).index(pitcher_name)
                           if pitcher_name in disp["Pitcher"].values else 0)
                )
            with rc2:
                # build UI options from the short labels
                options = [short_map[c] for c in metric_cols]
            
                # pick defaults by prefix so it works even if units or wording vary
                def pick_by_prefix(prefix, pool):
                    prefix = prefix.upper().replace(" ", "")
                    for label in pool:
                        if label.upper().replace(" ", "").startswith(prefix):
                            return label
                    return None
            
                wanted_prefixes = ["PP", "EPP", "RM", "JH"]  # Propulsive Power, Ecc. PP, Reactive Movement, Jump Height
                default_labels = []
                for p in wanted_prefixes:
                    lab = pick_by_prefix(p, options)
                    if lab and lab not in default_labels:
                        default_labels.append(lab)
            
                metrics_to_plot = st.multiselect(
                    "Metrics to include",
                    options=options,
                    default=default_labels
                )

            with rc3:
                roll_win = st.slider("Rolling window (tests)", 1, 10, 3, 1)
    
            series = work[work["Pitcher"] == p_for_chart].sort_values(vald_date_col).copy()
            if series.empty:
                st.info("No VALD tests for this player in the selected filters/date range.")
            else:
                tidy = []
                inv = {v: k for k, v in short_map.items()}
                for short in metrics_to_plot:
                    full = inv.get(short)
                    if full not in series.columns:
                        continue
                    s = series[[vald_date_col, full] + ([vald_fp_col] if vald_fp_col else [])].rename(columns={full: "Value"})
                    s["Metric"] = short
                    tidy.append(s)
                if tidy:
                    tidy = pd.concat(tidy, ignore_index=True)
                    grp_keys = ["Metric"] + ([vald_fp_col] if vald_fp_col else [])
                    out = []
                    for _, g in tidy.groupby(grp_keys, dropna=False):
                        g = g.sort_values(vald_date_col).copy()
                        g["Roll"] = g["Value"].rolling(window=roll_win, min_periods=1).mean()
                        out.append(g)
                    tidy = pd.concat(out, ignore_index=True)
    
                    fig = px.line(
                        tidy, x=vald_date_col, y="Roll",
                        color="Metric",
                        line_dash=(vald_fp_col if vald_fp_col else None),
                        markers=True,
                        title=f"Rolling ({roll_win}-test) — {p_for_chart}"
                    )
                    fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Score", legend_title=None)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select at least one metric to plot.")
    # ==== End VALD (ForceDecks) section ====





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



