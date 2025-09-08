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
    percent_cols = ['InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%', 'Pitch%', 'Hard%', 'Soft%', 'GB%', 'FB%', 'Contact%']
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
def plot_heatmaps(map_type: str):
    try:
        pitcher_data = filtered_df.copy()
        if pitcher_data.empty:
            st.info("No data available for the selected parameters.")
            return

        if not {"PlateLocSide", "PlateLocHeight", "PitchCall", "PitchType"}.issubset(pitcher_data.columns):
            st.info("Heatmap needs PlateLocSide, PlateLocHeight, PitchCall, PitchType columns.")
            return

        plot_data = pitcher_data.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        if plot_data.empty:
            st.info("No data available to plot after filtering.")
            return

        unique_pitch_types = plot_data["PitchType"].unique()
        n_pitch_types = len(unique_pitch_types)
        plots_per_row = 3
        n_rows = math.ceil(n_pitch_types / plots_per_row)

        fig_width = 6 * plots_per_row
        fig_height = 6 * n_rows
        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(fig_width, fig_height))
        if n_pitch_types == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, pitch_type in zip(axes, unique_pitch_types):
            pitch_type_data = plot_data[plot_data["PitchType"] == pitch_type]
            if map_type == "Frequency":
                heatmap_data = pitch_type_data
            elif map_type == "Whiff":
                heatmap_data = pitch_type_data[pitch_type_data["PitchCall"] == "StrikeSwinging"]
            elif map_type == "Exit Velocity":
                heatmap_data = pitch_type_data
            else:
                heatmap_data = pitch_type_data

            # base scatter
            ax.scatter(
                pitch_type_data["PlateLocSide"],
                pitch_type_data["PlateLocHeight"],
                color="black", edgecolor="white", s=40, alpha=0.6
            )

            if len(heatmap_data) >= 5:
                bw_adjust_value = 0.6 if len(heatmap_data) > 50 else 1.0
                cmap = "Spectral_r" if map_type == "Frequency" else "coolwarm"
                sns.kdeplot(
                    x=heatmap_data["PlateLocSide"],
                    y=heatmap_data["PlateLocHeight"],
                    fill=True, cmap=cmap, levels=6, ax=ax, bw_adjust=bw_adjust_value, thresh=0.05
                )

            # strike zone
            strike_zone_width = 1.66166
            strike_zone = patches.Rectangle(
                (-strike_zone_width / 2, 1.5),
                strike_zone_width, 3.3775 - 1.5,
                edgecolor="black", facecolor="none", linewidth=1.5
            )
            ax.add_patch(strike_zone)

            ax.set_xlim(-2, 2)
            ax.set_ylim(1, 4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.set_title(f"{pitch_type} ({pitcher_name})", fontsize=14)
            ax.set_aspect("equal", adjustable="box")

        # remove unused axes
        for j in range(len(unique_pitch_types), len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"{pitcher_name} {map_type} Heatmap (2025 College Season)", fontsize=18, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error generating {map_type} heatmaps: {e}")

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
            )
            .reset_index()
        )

        # --- FIX 1: StuffPlus merge on canonical keys (Pitcher + PitchType)
        # --- Stuff+ merge: collapse to one row per PitchType for this pitcher
        sp = stuff_df.copy()
        if not sp.empty:
            sp = sp[sp["Pitcher"] == pitcher_name]
        
            # If StuffPlus exists per-pitch/per-game, collapse to a single value per PitchType.
            # Mean is common; switch to median or last if you prefer.
            sp = (sp
                  .groupby("PitchType", as_index=False, observed=True)
                  .agg(StuffPlus=("StuffPlus", "mean")))
        
            # Now this is 1:1 by PitchType → safe to merge
            grouped = grouped.merge(sp, on="PitchType", how="left", validate="one_to_one")


        # sort by usage
        grouped = grouped.sort_values("Count", ascending=False)

        # weighted "All" row
        total = grouped["Count"].sum()
        def wavg(col):
            vals = grouped[col]
            mask = vals.notna()
            if not mask.any():
                return np.nan
            return np.average(vals[mask], weights=grouped.loc[mask, "Count"])

        all_row = {
            "PitchType": "All",
            "Count": total,
            "Velo": round(wavg("Velo"), 1) if pd.notna(wavg("Velo")) else np.nan,
            "iVB": round(wavg("iVB"), 1) if pd.notna(wavg("iVB")) else np.nan,
            "HB": round(wavg("HB"), 1) if pd.notna(wavg("HB")) else np.nan,
            "Spin": round(wavg("Spin"), 1) if pd.notna(wavg("Spin")) else np.nan,
            "RelH": round(wavg("RelH"), 1) if pd.notna(wavg("RelH")) else np.nan,
            "RelS": round(wavg("RelS"), 1) if pd.notna(wavg("RelS")) else np.nan,
            "Ext": round(wavg("Ext"), 1) if pd.notna(wavg("Ext")) else np.nan,
            "VAA": round(wavg("VAA"), 1) if pd.notna(wavg("VAA")) else np.nan,
            "StuffPlus": round(wavg("StuffPlus"), 1) if ("StuffPlus" in grouped.columns and pd.notna(wavg("StuffPlus"))) else np.nan
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

            # First pitch
            fp_df = slice_df[(slice_df["Balls"] == 0) & (slice_df["Strikes"] == 0)]
            fp_total = len(fp_df)
            fp_strikes = fp_df[~fp_df["PitchCall"].isin(["HitByPitch", "BallCalled", "BallInDirt", "BallinDirt"])].shape[0]
            fp_strike_pct = (fp_strikes / fp_total * 100) if fp_total > 0 else 0

            swings = slice_df[slice_df["PitchCall"].isin(swing_flags)].shape[0]
            whiffs = slice_df[slice_df["PitchCall"] == "StrikeSwinging"].shape[0]
            chase = slice_df[(~slice_df.index.isin(in_zone.index)) & (slice_df["PitchCall"].isin(swing_flags))].shape[0]
            in_zone_whiffs = in_zone[in_zone["PitchCall"] == "StrikeSwinging"].shape[0]
            strikes_all = slice_df[slice_df["PitchCall"].isin(strike_flags)].shape[0]

            return {
                "InZone%": (len(in_zone) / len(slice_df) * 100) if len(slice_df) else 0,
                "Swing%": (swings / len(slice_df) * 100) if len(slice_df) else 0,
                "Whiff%": (whiffs / swings * 100) if swings else 0,
                "Chase%": (chase / swings * 100) if swings else 0,
                "InZoneWhiff%": (in_zone_whiffs / len(in_zone) * 100) if len(in_zone) else 0,
                "Strike%": (strikes_all / len(slice_df) * 100) if len(slice_df) else 0,
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
        st.dataframe(format_dataframe(display[["Pitch","Count","Pitch%","Strike%","InZone%","Swing%","Whiff%","Chase%","InZoneWhiff%","FP Strike%"]]))
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

        data = df.dropna(subset=["InducedVertBreak", "HorzBreak"])
        if data.empty:
            st.info("No pitch movement data available for plotting.")
            return

        fig = go.Figure()
        for pt in data["PitchType"].unique():
            sub = data[data["PitchType"] == pt].copy()
            sub["RelSpeed"] = sub["RelSpeed"].round(1)
            sub["InducedVertBreak"] = sub["InducedVertBreak"].round(1)
            sub["HorzBreak"] = sub["HorzBreak"].round(1)
            fig.add_trace(go.Scatter(
                x=sub["HorzBreak"], y=sub["InducedVertBreak"],
                mode="markers", name=pt,
                marker=dict(size=9, color=PLOTLY_COLORS.get(pt, "black"), opacity=0.85, line=dict(width=1, color="white")),
                text=sub.apply(lambda r: f"Date: {r['Date']}<br>Velo: {r['RelSpeed']}<br>iVB: {r['InducedVertBreak']}<br>HB: {r['HorzBreak']}<br>Spin: {r.get('SpinRate','')}", axis=1),
                hoverinfo="text"
            ))
        fig.add_shape(type="line", x0=0, x1=0, y0=-25, y1=25, line=dict(color="black", width=2), layer="below")
        fig.add_shape(type="line", x0=-25, x1=25, y0=0, y1=0, line=dict(color="black", width=2), layer="below")
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="black")
        fig.update_layout(
            title=f"Pitch Movement for {pitcher_name}",
            xaxis=dict(title="Horizontal Break (inches)", range=[-30, 30]),
            yaxis=dict(title="Induced Vertical Break (inches)", range=[-30, 30]),
            template="plotly_white", legend_title="Pitch Type", width=900, height=700
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

            st.subheader("Rolling Averages Across Full Database (Date-by-Date)")
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
                st.plotly_chart(fig, use_container_width=True)

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
            st.subheader(f"Pitch-by-Pitch View for {xdt.strftime('%B %d, %Y')}")

            for metric, label in metrics:
                if metric not in day.columns:
                    continue
                fig = px.line(
                    day, x="PitchNo", y=metric, color="PitchType",
                    title=f"{label} Pitch-by-Pitch", labels={"PitchNo": "Pitch #", metric: label, "PitchType": "Pitch Type"},
                    color_discrete_map=color_map, hover_data={"PitchNo": ":.0f", metric: ":.2f"},
                )
                for pt in day["PitchType"].unique():
                    sub = day[day["PitchType"] == pt]
                    fig.add_scatter(
                        x=sub["PitchNo"], y=sub[metric], mode="markers",
                        marker=dict(size=8, color=color_map.get(pt, "black")), name=f"{pt} pts", showlegend=False
                    )
                fig.update_xaxes(range=[day["PitchNo"].min() - 1, day["PitchNo"].max() + 1])
                fig.update_layout(xaxis_title="Pitch #", yaxis_title=label, legend_title="Pitch Type",
                                  template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while generating rolling line graphs: {e}")





# === RENDER ===
tab_flight, tab_biomech = st.tabs(["Pitch Flight Data", "Bio Mech Data"])

with tab_flight:
    plot_heatmaps(heatmap_type)
    generate_plate_discipline_table()
    generate_pitch_traits_table()
    generate_batted_ball_table()
    plot_pitch_movement()
    # use the explicit rolling mode + optional pitch-by-pitch date
    generate_rolling_line_graphs(rolling_view_mode, pitch_by_pitch_date=pp_selected_date)
    plot_release_and_approach_angles()

with tab_biomech:
    st.subheader("Bio Mech Data")
    st.info("Coming soon: add biomechanics metrics, force plate summaries, motion-capture angles, and more.")

