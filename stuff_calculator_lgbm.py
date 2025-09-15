# stuff_calculator_lgbm.py
from __future__ import annotations
import os, json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
import matplotlib.pyplot as plt

# ── Your model feature sets ──────────────────────────────────────────────────────
BASE_STUFF = ["RelSpeed", "SpinRate", "spinDir", "RelHeight", "RelSide", "Extension",
              "InducedVertBreak", "HorzBreak"]
DELTA_FEATS = ["delta_velo", "delta_IVB", "delta_HB"]

# UI ranges (kept from your program)
FEATURE_RANGES = {
    "RelSpeed": {"min": 65.0, "max": 105.0, "default": 92.0, "step": 0.1, "unit": "mph", "description": "Release Speed"},
    "SpinRate": {"min": 500.0, "max": 3500.0, "default": 2300.0, "step": 50.0, "unit": "rpm", "description": "Spin Rate"},
    "spinDir": {"min": 0.0, "max": 360.0, "default": 225.0, "step": 5.0, "unit": "°", "description": "Spin Direction"},
    "RelHeight": {"min": 4.0, "max": 8.0, "default": 6.2, "step": 0.05, "unit": "ft", "description": "Release Height"},
    "RelSide": {"min": -3.5, "max": 3.5, "default": 1.8, "step": 0.05, "unit": "ft", "description": "Release Side"},
    "Extension": {"min": 4.0, "max": 8.0, "default": 6.1, "step": 0.05, "unit": "ft", "description": "Extension"},
    "InducedVertBreak": {"min": -5.0, "max": 30.0, "default": 16.5, "step": 0.1, "unit": "in", "description": "Induced Vertical Break"},
    "HorzBreak": {"min": -25.0, "max": 25.0, "default": 8.2, "step": 0.1, "unit": "in", "description": "Horizontal Break"},
    "delta_velo": {"min": -20.0, "max": 5.0, "default": -8.5, "step": 0.1, "unit": "mph", "description": "Velocity vs FB"},
    "delta_IVB": {"min": -40.0, "max": 15.0, "default": -12.0, "step": 0.1, "unit": "in", "description": "IVB vs FB"},
    "delta_HB": {"min": -40.0, "max": 40.0, "default": -5.5, "step": 0.1, "unit": "in", "description": "HB vs FB"}
}

# Where to find your artifacts in this repo/deployment.
# You can override with env var STUFF_ART_DIR.
ART_DIR = os.environ.get("STUFF_ART_DIR", "artifacts/stuffplus")

# ── Helpers ──────────────────────────────────────────────────────────────────────
def map_pitch_type_to_family(pitch_type: str) -> str:
    """Map your PitchType labels to Stuff+ families."""
    if not pitch_type:
        return "Fastball"
    pt = str(pitch_type).lower()
    if pt in {"four-seam", "fourseam", "ff", "cutter", "fc", "sinker", "si", "two-seam"}:
        return "Fastball"
    if pt in {"slider", "sl", "curveball", "cu", "sweeper", "kc"}:
        return "Breaking"
    if pt in {"changeup", "ch", "splitter", "fs", "forkball"}:
        return "Offspeed"
    return "Fastball"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load models, medians and population stats from ART_DIR."""
    try:
        with open(os.path.join(ART_DIR, "population_stats.json")) as f:
            pop_stats = json.load(f)
        with open(os.path.join(ART_DIR, "impute_medians.json")) as f:
            medians = json.load(f)

        boosters = {}
        for fam in ["Fastball", "Breaking", "Offspeed"]:
            p = os.path.join(ART_DIR, f"lgbm_{fam.lower()}_model.txt")
            if os.path.exists(p):
                boosters[fam] = lgb.Booster(model_file=p)
        return pop_stats, medians, boosters
    except Exception as e:
        st.error(f"Error loading artifacts from {ART_DIR}: {e}")
        return None, None, None

def apply_lefty_flip(features: Dict[str, float], is_lefty: bool) -> Dict[str, float]:
    if not is_lefty:
        return features
    f = features.copy()
    for k in ["HorzBreak", "RelSide"]:
        if k in f and pd.notna(f[k]):
            f[k] = -float(f[k])
    return f

def make_prediction(features: Dict[str, float], family: str, boosters: Dict, medians: Dict,
                    is_lefty: bool=False) -> Tuple[float, Dict[str, float]]:
    model_features = apply_lefty_flip(features, is_lefty)
    feat_list = BASE_STUFF[:] if family == "Fastball" else BASE_STUFF + DELTA_FEATS
    X = pd.DataFrame([model_features])
    # ensure expected order + impute missing with family medians
    fam_medians = medians.get(family, {})
    for col in feat_list:
        if col not in X.columns:
            X[col] = fam_medians.get(col, 0.0)
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(fam_medians.get(col, X[col].median()))
    booster = boosters.get(family)
    if booster is None:
        return 0.0, model_features
    pred = float(booster.predict(X.values)[0])
    return pred, model_features

def calculate_stuff_plus(prediction: float, family: str, pop_stats: Dict) -> Tuple[int, str, str]:
    run_suppress = -prediction
    fam_stats = pop_stats.get(family, {"mu": 0.0, "sd": 1.0})
    stuff_plus = 100 + 20 * ((run_suppress - float(fam_stats["mu"])) / float(fam_stats["sd"]))
    sp = int(round(stuff_plus))
    if sp >= 120:   return sp, "Elite (80)", "green"
    if sp >= 115:   return sp, "Plus-Plus (70)", "blue"
    if sp >= 110:   return sp, "Plus (60)", "blue"
    if sp >= 105:   return sp, "Above Average (55)", "orange"
    if sp >= 95:    return sp, "Average (50)", "gray"
    if sp >= 85:    return sp, "Below Average (45)", "orange"
    return sp, "Poor (40)", "red"

def _pitcher_pitchtype_avgs(df: pd.DataFrame, pitcher: str, pitch_type: str) -> Dict[str, float]:
    """Mean values for a pitcher/pitch_type. Also map Tilt->spinDir if needed."""
    one = df[(df.get("Pitcher") == pitcher) & (df.get("PitchType") == pitch_type)].copy()
    d: Dict[str, float] = {}
    if one.empty:
        return d
    # compute means for all possible fields
    for k in BASE_STUFF:
        if k in one.columns:
            d[k] = float(pd.to_numeric(one[k], errors="coerce").mean())
    # Tilt -> spinDir fallback
    if "spinDir" not in d:
        if "Tilt" in one.columns:
            d["spinDir"] = float(pd.to_numeric(one["Tilt"], errors="coerce").mean())
    return d

def _pitcher_fastball_avgs(df: pd.DataFrame, pitcher: str) -> Dict[str, float]:
    """Average FB metrics for deltas (use Four-Seam/Cutter/Sinker as FB family)."""
    fb_like = df[df.get("Pitcher").eq(pitcher) &
                 df.get("PitchType").isin(["Four-Seam", "Cutter", "Sinker", "Two-Seam"])].copy()
    d: Dict[str, float] = {}
    if fb_like.empty:
        return d
    for k in ["RelSpeed", "InducedVertBreak", "HorzBreak"]:
        if k in fb_like.columns:
            d[k] = float(pd.to_numeric(fb_like[k], errors="coerce").mean())
    return d

def _preset_from_pitcher(df: pd.DataFrame, pitcher: str, pitch_type: str,
                         current_family: str) -> Dict[str, float]:
    """Build a full preset dict (base + deltas if needed)."""
    base_vals = _pitcher_pitchtype_avgs(df, pitcher, pitch_type)
    # always include spinDir field (if missing after Tilt fallback)
    base_vals.setdefault("spinDir", FEATURE_RANGES["spinDir"]["default"])

    if current_family == "Fastball":
        return base_vals

    # For Breaking/Offspeed: compute deltas vs THIS pitcher’s fastball averages
    fb = _pitcher_fastball_avgs(df, pitcher)
    deltas = {}
    if fb:
        if "RelSpeed" in base_vals and "RelSpeed" in fb:
            deltas["delta_velo"] = base_vals["RelSpeed"] - fb["RelSpeed"]
        if "InducedVertBreak" in base_vals and "InducedVertBreak" in fb:
            deltas["delta_IVB"] = base_vals["InducedVertBreak"] - fb["InducedVertBreak"]
        if "HorzBreak" in base_vals and "HorzBreak" in fb:
            deltas["delta_HB"] = base_vals["HorzBreak"] - fb["HorzBreak"]
    # fill any missing deltas with defaults
    for k in DELTA_FEATS:
        deltas.setdefault(k, FEATURE_RANGES[k]["default"])
    return {**base_vals, **deltas}

# ── MAIN RENDERER (call this from your new tab) ───────────────────────────────────
def render_stuff_calculator_tab(season_df: pd.DataFrame, default_pitcher: str, default_pitch_type: str|None=None):
    st.subheader("Stuff+ Calculator (LightGBM)")
    pop_stats, medians, boosters = load_artifacts()
    if pop_stats is None:
        st.error("Could not load model artifacts.")
        st.write(f"Expected at: `{ART_DIR}`")
        return

    # Top controls
    l, r = st.columns(2)
    with l:
        pitchers = sorted(season_df.get("Pitcher", pd.Series(dtype=str)).dropna().unique().tolist())
        pitcher = st.selectbox("Pitcher (for presets)", pitchers,
                               index=pitchers.index(default_pitcher) if default_pitcher in pitchers else 0)
    with r:
        pitch_types = sorted(season_df.get("PitchType", pd.Series(dtype=str)).dropna().unique().tolist())
        if default_pitch_type and default_pitch_type in pitch_types:
            def_idx = pitch_types.index(default_pitch_type)
        else:
            def_idx = 0
        pitch_type = st.selectbox("Pitch Type (sets family + preset)", pitch_types, index=def_idx)

    # Derived family + handedness option
    family = map_pitch_type_to_family(pitch_type)
    is_lefty = st.checkbox("Left-handed pitcher", value=False)

    st.caption(f"Family detected for '{pitch_type}': **{family}**")

    # ── Inputs (sliders), with ability to preset from averages ──
    if st.button("Preset from this pitcher’s averages for selected pitch type", use_container_width=True):
        preset_vals = _preset_from_pitcher(season_df, pitcher, pitch_type, family)
        for k, v in preset_vals.items():
            st.session_state[f"sp_{k}"] = float(v)

    # Grid for sliders
    cols = st.columns(2)
    # Base features first
    base_features: Dict[str, float] = {}
    for i, k in enumerate(BASE_STUFF):
        rng = FEATURE_RANGES[k]
        with cols[i % 2]:
            base_features[k] = st.slider(
                f"{rng['description']} ({rng['unit']})",
                min_value=rng["min"], max_value=rng["max"],
                value=float(st.session_state.get(f"sp_{k}", rng["default"])),
                step=rng["step"], key=f"sl_{k}"
            )
    # Deltas only for non-fastballs
    delta_features: Dict[str, float] = {}
    if family != "Fastball":
        st.markdown("**Delta Features (vs Fastball)**")
        dcols = st.columns(2)
        for i, k in enumerate(DELTA_FEATS):
            rng = FEATURE_RANGES[k]
            with dcols[i % 2]:
                delta_features[k] = st.slider(
                    f"{rng['description']} ({rng['unit']})",
                    min_value=rng["min"], max_value=rng["max"],
                    value=float(st.session_state.get(f"sp_{k}", rng["default"])),
                    step=rng["step"], key=f"sl_{k}"
                )

    # Predict
    all_features = {**base_features, **delta_features}
    pred, used = make_prediction(all_features, family, boosters, medians, is_lefty)
    sp, grade, color = calculate_stuff_plus(pred, family, pop_stats)

    # KPIs
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Stuff+ Score", sp)
    with k2:
        st.metric("Family", family)
    with k3:
        st.metric("Raw Prediction", f"{pred:.4f}")

    # Grade message
    if color == "green":
        st.success(f"Grade: {grade}")
    elif color == "blue":
        st.info(f"Grade: {grade}")
    elif color == "orange":
        st.warning(f"Grade: {grade}")
    else:
        st.error(f"Grade: {grade}")

    # Technical
    st.markdown("##### Technical Details")
    fam_stats = pop_stats.get(family, {})
    st.write(f"Run Suppress: {-pred:.4f}  |  μ: {fam_stats.get('mu', 0):.4f}  |  σ: {fam_stats.get('sd', 1):.4f}  |  Lefty flip: {'Yes' if is_lefty else 'No'}")

    # Sensitivity mini-analysis (same idea as your file, narrower range to keep UI snappy)
    st.markdown("##### Sensitivity (±2 units around current value)")
    avail_features = BASE_STUFF if family == "Fastball" else BASE_STUFF + DELTA_FEATS
    sens_feature = st.selectbox("Feature to analyze:", avail_features)
    rng = FEATURE_RANGES[sens_feature]
    current_val = all_features[sens_feature]
    sens_min = max(rng["min"], current_val - 2.0)
    sens_max = min(rng["max"], current_val + 2.0)
    sens_step = rng["step"]
    sens_vals = np.arange(sens_min, sens_max + sens_step, sens_step)
    sens_stuff = []
    for v in sens_vals:
        tmp = dict(all_features)
        tmp[sens_feature] = float(v)
        p, _ = make_prediction(tmp, family, boosters, medians, is_lefty)
        s, _, _ = calculate_stuff_plus(p, family, pop_stats)
        sens_stuff.append(s)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sens_vals, sens_stuff, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel(f"{sens_feature} ({rng['unit']})")
    ax.set_ylabel("Stuff+")
    ax.grid(True, alpha=0.3)
    ax.axvline(current_val, color='red', linestyle='--', alpha=0.6)
    ax.axhline(sp, color='red', linestyle='--', alpha=0.6)
    st.pyplot(fig)
    plt.close()

    # Export current setup
    with st.expander("Export current settings"):
        export = {
            "pitcher": pitcher,
            "pitch_type": pitch_type,
            "family": family,
            "is_lefty": is_lefty,
            "features": all_features,
            "results": {"prediction": pred, "stuff_plus": sp, "grade": grade}
        }
        st.json(export)
