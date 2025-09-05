import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import math
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# === LOAD DATA ===
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'].notna()]
    return df

# File path for 2025 Season data
season_file_path = "Fall_2025_wRV_MASTER.csv"
season_df = load_data(season_file_path)
season_df = season_df[season_df['PitcherTeam'] == 'OLE_REB']


# Convert numeric columns
numeric_columns = ['RelSpeed', 'SpinRate', 'Tilt', 'RelHeight', 'RelSide', 
                   'Extension', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle', 'ExitSpeed']
for col in numeric_columns:
    season_df[col] = pd.to_numeric(season_df[col], errors='coerce')

# === LOAD ROLLING AND StuffPlus DATA ===
rolling_path = "Fall_2025_wRV_MASTER.csv"
class_plus_path = "Fall_2025_wRV_MASTER.csv"

rolling_df = load_data(rolling_path)
class_plus_df = pd.read_csv(class_plus_path)

class_plus_df["Season"] = "2025 Season"

# === STREAMLIT SETUP ===
st.title("Brewster Pitcher Reports (2025 CCBL Season)")
st.sidebar.header("Filters")

# Dropdowns
pitcher_name = st.sidebar.selectbox("Select Pitcher:", season_df['Pitcher'].unique())
heatmap_type = st.sidebar.selectbox("Select Heatmap Type:", ["Frequency", "Whiff", "Exit Velocity"])
batter_side = st.sidebar.selectbox("Select Batter Side:", ['Both', 'Right', 'Left'])
strikes = st.sidebar.selectbox("Select Strikes:", ['All', 0, 1, 2])
balls = st.sidebar.selectbox("Select Balls:", ['All', 0, 1, 2, 3])

# Date Filtering
st.sidebar.header("Date Filtering")
date_filter_option = st.sidebar.selectbox("Select Date Filter:", ["All", "Single Date", "Date Range"])
selected_date = None
start_date = None
end_date = None
if date_filter_option == "Single Date":
    selected_date = st.sidebar.date_input("Select a Date", value=datetime.today())
elif date_filter_option == "Date Range":
    date_range = st.sidebar.date_input("Select Date Range", value=[datetime.today(), datetime.today()])
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        st.sidebar.warning("Please select a valid date range.")

# === FILTER FUNCTION ===
def filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    df = season_df[season_df['Pitcher'] == pitcher_name]
    if batter_side != 'Both':
        df = df[df['BatterSide'] == batter_side]
    if strikes != 'All':
        df = df[df['Strikes'] == strikes]
    if balls != 'All':
        df = df[df['Balls'] == balls]
    if date_filter_option == "Single Date" and selected_date:
        df = df[df['Date'].dt.date == pd.to_datetime(selected_date).date()]
    elif date_filter_option == "Date Range" and start_date and end_date:
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    return df

# === FILTERED DATA FOR GLOBAL USE ===
filtered_df = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

# === EXTRACT RELEASE & APPROACH DATA FOR GLOBAL USE ===
release_data = filtered_df.dropna(subset=['HorzRelAngle', 'VertRelAngle'])
approach_data = filtered_df.dropna(subset=['HorzApprAngle', 'VertApprAngle'])









def plot_heatmaps(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date, map_type):
    try:
        # Filter data with date parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)
        
        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return
        
        # Remove rows where PlateLocSide or PlateLocHeight is NaN, for plotting purposes only
        plot_data = pitcher_data.dropna(subset=['PlateLocSide', 'PlateLocHeight'])
        
        if plot_data.empty:
            st.write("No data available to plot after filtering.")
            return
        
        # Get unique pitch types thrown by the selected pitcher
        unique_pitch_types = plot_data['TaggedPitchType'].unique()
        
        # Limit number of subplots per row (e.g., 3 per row)
        n_pitch_types = len(unique_pitch_types)
        plots_per_row = 3  # Set number of plots per row
        n_rows = math.ceil(n_pitch_types / plots_per_row)  # Calculate the number of rows needed
        
        # Adjust figure size dynamically
        fig_width = 12 * plots_per_row  # Set width based on number of plots per row
        fig_height = 16 * n_rows  # Set height to fit all rows

        # Create subplots with the appropriate number of rows and columns
        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(fig_width, fig_height))
        
        if n_pitch_types == 1:
            axes = [axes]  # Ensure axes is iterable
        else:
            axes = axes.flatten()  # Flatten axes array for easier access

        # Loop over each unique pitch type and create heatmaps
        for i, (ax, pitch_type) in enumerate(zip(axes, unique_pitch_types)):
            pitch_type_data = plot_data[plot_data['TaggedPitchType'] == pitch_type]
            
            if map_type == 'Frequency':
                # All pitches are used for frequency maps
                heatmap_data = pitch_type_data
            elif map_type == 'Whiff':
                # Only use pitches with 'StrikeSwinging' for heatmap
                heatmap_data = pitch_type_data[pitch_type_data['PitchCall'] == 'StrikeSwinging']
            elif map_type == 'Exit Velocity':
                # Use all pitches for Exit Velocity but map ExitSpeed
                heatmap_data = pitch_type_data

            # Scatter plot for all pitches
            ax.scatter(
                pitch_type_data['PlateLocSide'], 
                pitch_type_data['PlateLocHeight'], 
                color='black',  # Color for the dots
                edgecolor='white',  # Add a white border to make dots stand out
                s=300,  # Size of the dots
                alpha=0.7  # Transparency to allow overlap
            )
            
            # Check if enough data points exist for a heatmap
            if len(heatmap_data) >= 5:
                bw_adjust_value = 0.5 if len(heatmap_data) > 50 else 1  # Adjust bandwidth for small datasets
                sns.kdeplot(
                    x=heatmap_data['PlateLocSide'], 
                    y=heatmap_data['PlateLocHeight'], 
                    fill=True, 
                    cmap='Spectral_r' if map_type == 'Frequency' else 'coolwarm', 
                    levels=6, 
                    ax=ax,
                    bw_adjust=bw_adjust_value
                )

            # Add strike zone as a rectangle with black edgecolor
            strike_zone_width = 1.66166  # feet changed for widest raw strike (formerly 17/12)
            strike_zone_params = {
                'x_start': -strike_zone_width / 2,
                'y_start': 1.5,
                'width': strike_zone_width,
                'height': 3.3775 - 1.5
            }
            strike_zone = patches.Rectangle(
                (strike_zone_params['x_start'], strike_zone_params['y_start']),
                strike_zone_params['width'],
                strike_zone_params['height'],
                edgecolor='black',  # Black edge color for the strike zone
                facecolor='none',
                linewidth=2
            )
            ax.add_patch(strike_zone)
            
            # Set axis limits and remove ticks
            ax.set_xlim(-2, 2)
            ax.set_ylim(1, 4)
            ax.set_xticks([])  # Remove x-ticks
            ax.set_yticks([])  # Remove y-ticks
            
            # Remove axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Set pitch type as title
            ax.set_title(f"{pitch_type} ({pitcher_name})", fontsize=24)  # Increased font size

            # Equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
        
        # Remove any unused subplots
        for j in range(len(unique_pitch_types), len(axes)):
            fig.delaxes(axes[j])

        # Add a main title for all the heatmaps
        season = pitcher_data['Season'].iloc[0] if 'Season' in pitcher_data.columns else "Unknown"
        plt.suptitle(f"{pitcher_name} {map_type} Heatmap (2025 College Season)", 
                     fontsize=30, fontweight='bold')
        
        # Adjust the layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for suptitle
        
        # Show the updated figure
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error generating {map_type} heatmaps: {e}")












# Function to calculate InZone% and Chase%
def calculate_in_zone(df):
    # Strike zone boundaries
    in_zone = df[
        (df['PlateLocHeight'] >= 1.5) & 
        (df['PlateLocHeight'] <= 3.3775) & 
        (df['PlateLocSide'] >= -0.83083) & 
        (df['PlateLocSide'] <= 0.83083)
    ]
    return in_zone

# Function to manually format the dataframe before displaying
# Function to manually format the dataframe before displaying, with alternating row colors
# Function to manually format the dataframe before displaying, with rounding and alternating row colors
# Function to manually format the dataframe before displaying, with rounding and alternating row colors
# Function to manually format the dataframe before displaying (no alternating row colors)
def format_dataframe(df):
    df = df.copy()  # Create a copy to avoid warnings
    percent_columns = ['InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%']

    # Format percentages and floats
    for col in df.columns:
        if col in percent_columns:
            df[col] = df[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) and isinstance(x, (int, float)) else 'N/A')  # Add % symbol
        elif df[col].dtype.kind in 'f':  # if it's a float type column
            df[col] = df[col].apply(lambda x: round(x, 2) if pd.notna(x) else 'N/A')
        else:
            df[col] = df[col].fillna('N/A')  # Fill NaN with N/A for non-float columns

    return df

# Load StuffPlus CSV into a DataFrame


@st.cache_data
def load_class_plus_data(file_path):
    df = pd.read_csv(file_path)
    
    
    
    return df

class_plus_file_path = "Fall_2025_wRV_MASTER.csv"

class_plus_df = load_class_plus_data(class_plus_file_path)




season_class_plus_file_path = "Fall_2025_wRV_MASTER.csv"

#

# Load Spring StuffPlus CSV
@st.cache_data
def load_season_class_plus_data(file_path):
    df = pd.read_csv(file_path)
    df['Season'] = '2025 Season'  # Add season identifier
    # Rename pitch types to match other datasets
    
    return df

# Load the Spring StuffPlus dataset
season_class_plus_df = load_season_class_plus_data(season_class_plus_file_path)






def generate_pitch_traits_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)
        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Group by pitch type and calculate means
        grouped_data = pitcher_data.groupby('TaggedPitchType').agg(
            Count=('TaggedPitchType', 'size'),
            RelSpeed=('RelSpeed', 'mean'),
            InducedVertBreak=('InducedVertBreak', 'mean'),
            HorizontalBreak=('HorzBreak', 'mean'),
            SpinRate=('SpinRate', 'mean'),
            RelHeight=('RelHeight', 'mean'),
            RelSide=('RelSide', 'mean'),
            Extension=('Extension', 'mean'),
            VertApprAngle=('VertApprAngle', 'mean')
        ).reset_index()
        
        

        # Clean and round numeric columns
        rename_columns = {
            'TaggedPitchType': 'Pitch',
            'RelSpeed': 'Velo',
            'InducedVertBreak': 'iVB',
            'HorizontalBreak': 'HB',
            'SpinRate': 'Spin',
            'RelHeight': 'RelH',
            'RelSide': 'RelS',
            'Extension': 'Ext',
            'VertApprAngle': 'VAA'
        }
        grouped_data = grouped_data.rename(columns=rename_columns)
        # Merge with CLASS+ data (2025 only)
        filtered_class_plus = season_class_plus_df[season_class_plus_df["Pitcher"] == pitcher_name]
        grouped_data = pd.merge(
            grouped_data,
            filtered_class_plus[["TaggedPitchType", "StuffPlus"]], 
            how="left",
            left_on="Pitch",
            right_on="TaggedPitchType"
        )
        
        grouped_data["StuffPlus"] = pd.to_numeric(grouped_data["StuffPlus"], errors="coerce").fillna("N/A")

       
        
        numeric_columns = ['Velo', 'iVB', 'HB', 'Spin', 'RelH', 'RelS', 'Ext', 'VAA']
        for col in numeric_columns:
            grouped_data[col] = pd.to_numeric(grouped_data[col], errors='coerce')
        grouped_data[numeric_columns] = grouped_data[numeric_columns].round(1)

        # Sort and add 'All' row
        grouped_data = grouped_data.sort_values(by='Count', ascending=False)
        total_count = grouped_data["Count"].sum()
        weighted_averages = {
            col: np.average(grouped_data[col].dropna(), weights=grouped_data["Count"].loc[grouped_data[col].notna()])
            if grouped_data[col].notna().any() else "N/A"
            for col in numeric_columns
        }

        valid_class_plus = grouped_data.loc[grouped_data["StuffPlus"] != "N/A", "StuffPlus"].astype(float)
        valid_class_plus_weights = grouped_data.loc[grouped_data["StuffPlus"] != "N/A", "Count"]
        class_plus_weighted_avg = (
            np.average(valid_class_plus, weights=valid_class_plus_weights) if not valid_class_plus.empty else np.nan
        )

        all_row = {
            'Pitch': 'All',
            'Count': total_count,
            'Velo': round(weighted_averages['Velo'], 1) if pd.notna(weighted_averages['Velo']) else 'N/A',
            'iVB': round(weighted_averages['iVB'], 1) if pd.notna(weighted_averages['iVB']) else 'N/A',
            'HB': round(weighted_averages['HB'], 1) if pd.notna(weighted_averages['HB']) else 'N/A',
            'Spin': round(weighted_averages['Spin'], 1) if pd.notna(weighted_averages['Spin']) else 'N/A',
            'RelH': round(weighted_averages['RelH'], 1) if pd.notna(weighted_averages['RelH']) else 'N/A',
            'RelS': round(weighted_averages['RelS'], 1) if pd.notna(weighted_averages['RelS']) else 'N/A',
            'Ext': round(weighted_averages['Ext'], 1) if pd.notna(weighted_averages['Ext']) else 'N/A',
            'VAA': round(weighted_averages['VAA'], 1) if pd.notna(weighted_averages['VAA']) else 'N/A',
            'StuffPlus': round(class_plus_weighted_avg, 1) if pd.notna(class_plus_weighted_avg) else "N/A"


        }

        grouped_data = pd.concat([grouped_data, pd.DataFrame([all_row])], ignore_index=True)

        # Display table
        formatted_data = format_dataframe(grouped_data)
        st.subheader("Pitch Traits:")
        st.dataframe(formatted_data)

    except Exception as e:
        st.error(f"An error occurred while generating the pitch traits table: {e}")






def generate_plate_discipline_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on input parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        # Check if filtered data is empty
        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Total number of pitches for percentage calculations
        total_pitches = len(pitcher_data)

        # Function to calculate plate discipline metrics
        def calculate_metrics(df):
            # Determine pitches in the strike zone
            in_zone_pitches = calculate_in_zone(df)
            total_in_zone = len(in_zone_pitches)

            # Define swing-related conditions
            # First pitch logic
            fp_df = df[(df['Balls'] == 0) & (df['Strikes'] == 0)]
            fp_total = len(fp_df)
            fp_strikes = fp_df[
                ~fp_df['PitchCall'].isin(['HitByPitch', 'BallCalled', 'BallInDirt', 'BallinDirt'])
            ].shape[0]

            fp_strike_pct = (fp_strikes / fp_total) * 100 if fp_total > 0 else 0
            swing_conditions = ['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay']
            total_swings = df[df['PitchCall'].isin(swing_conditions)].shape[0]
            total_whiffs = df[df['PitchCall'] == 'StrikeSwinging'].shape[0]
            total_chase = df[
                (~df.index.isin(in_zone_pitches.index)) & 
                df['PitchCall'].isin(swing_conditions)
            ].shape[0]

            # Whiffs in the zone
            in_zone_whiffs = in_zone_pitches[in_zone_pitches['PitchCall'] == 'StrikeSwinging'].shape[0]

            # Define strike-related conditions
            strike_conditions = ['StrikeCalled', 'FoulBallFieldable', 'FoulBallNotFieldable', 'StrikeSwinging', 'InPlay']
            total_strikes = df[df['PitchCall'].isin(strike_conditions)].shape[0]

            # Calculate metrics
            metrics = {
                'InZone%': (total_in_zone / len(df)) * 100 if len(df) > 0 else 0,
                'Swing%': (total_swings / len(df)) * 100 if len(df) > 0 else 0,
                'Whiff%': (total_whiffs / total_swings) * 100 if total_swings > 0 else 0,
                'Chase%': (total_chase / total_swings) * 100 if total_swings > 0 else 0,
                'InZoneWhiff%': (in_zone_whiffs / total_in_zone) * 100 if total_in_zone > 0 else 0,
                'Strike%': (total_strikes / len(df)) * 100 if len(df) > 0 else 0,
                'FP Strike%': fp_strike_pct  # ✅ Add this line
            }
            return metrics

        # Group data by pitch type and calculate metrics
        plate_discipline_data = pitcher_data.groupby('TaggedPitchType').apply(calculate_metrics).apply(pd.Series).reset_index()

        # Calculate pitch percentage for each pitch type
        plate_discipline_data['Count'] = pitcher_data.groupby('TaggedPitchType')['TaggedPitchType'].count().values
        plate_discipline_data['Pitch%'] = (plate_discipline_data['Count'] / total_pitches) * 100

        # Reorder columns for display
        plate_discipline_data = plate_discipline_data[['TaggedPitchType', 'Count', 'Pitch%', 'Strike%', 'InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%', 'FP Strike%']]


        # Rename columns for readability
        rename_columns = {
            'TaggedPitchType': 'Pitch',
            'Count': 'Count',
            'Pitch%': 'Pitch%',
            'Strike%': 'Strike%',
            'InZone%': 'InZone%',
            'Swing%': 'Swing%',
            'Whiff%': 'Whiff%',
            'Chase%': 'Chase%',
            'InZoneWhiff%': 'InZoneWhiff%',
            'FP Strike%': 'FP Strike%'
        }
        plate_discipline_data = plate_discipline_data.rename(columns=rename_columns)

        # Calculate aggregate "All" row
        in_zone_pitches = calculate_in_zone(pitcher_data)
        total_swings = pitcher_data[pitcher_data['PitchCall'].isin(['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay'])].shape[0]
        total_whiffs = pitcher_data[pitcher_data['PitchCall'] == 'StrikeSwinging'].shape[0]
        total_chase = pitcher_data[
            (~pitcher_data.index.isin(in_zone_pitches.index)) & 
            pitcher_data['PitchCall'].isin(['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay'])
        ].shape[0]
        in_zone_whiffs = in_zone_pitches[in_zone_pitches['PitchCall'] == 'StrikeSwinging'].shape[0]
        total_strikes = pitcher_data[pitcher_data['PitchCall'].isin(['StrikeCalled', 'FoulBallFieldable', 'FoulBallNotFieldable', 'StrikeSwinging', 'InPlay'])].shape[0]
        fp_df = pitcher_data[(pitcher_data['Balls'] == 0) & (pitcher_data['Strikes'] == 0)]
        fp_total = len(fp_df)
        fp_strikes = fp_df[
            ~fp_df['PitchCall'].isin(['HitByPitch', 'BallCalled', 'BallInDirt', 'BallinDirt'])
        ].shape[0]
        fp_strike_pct = (fp_strikes / fp_total) * 100 if fp_total > 0 else 0

        all_row = {
            'Pitch': 'All',
            'Count': total_pitches,  # Total number of pitches
            'Pitch%': 100.0,  # Percentage is 100 for aggregate
            'Strike%': (total_strikes / total_pitches) * 100,
            'InZone%': (in_zone_pitches.shape[0] / total_pitches) * 100,
            'Swing%': (total_swings / total_pitches) * 100,
            'Whiff%': (total_whiffs / total_swings) * 100 if total_swings > 0 else 0,
            'Chase%': (total_chase / total_swings) * 100 if total_swings > 0 else 0,
            'InZoneWhiff%': (in_zone_whiffs / in_zone_pitches.shape[0]) * 100 if in_zone_pitches.shape[0] > 0 else 0,
            'FP Strike%': fp_strike_pct

        }

        # Append "All" row to the DataFrame
        all_row_df = pd.DataFrame([all_row])
        plate_discipline_data = pd.concat([plate_discipline_data, all_row_df], ignore_index=True)

        # Format the DataFrame for display
        formatted_data = format_dataframe(plate_discipline_data)

        # Display the results in Streamlit
        st.subheader("Plate Discipline:")
        st.dataframe(formatted_data)
    except Exception as e:
        st.error(f"An error occurred while generating the plate discipline table: {e}")


# Define a color dictionary for each pitch type
color_dict = {
    'Fastball': 'blue',
    'Four-Seam': 'blue',
    'Sinker': 'gold',
    'Slider': 'green',
    'Curveball': 'red',
    'Cutter': 'orange',
    'ChangeUp': 'purple',
    'Changeup': 'purple',
    'Splitter': 'teal',
    'Unknown': 'black',
    'Other': 'black'
}



import plotly.express as px
import plotly.graph_objects as go

import plotly.graph_objects as go

import plotly.graph_objects as go

def plot_pitch_movement(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on selected parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Drop NaN values for plotting
        movement_data = pitcher_data.dropna(subset=['InducedVertBreak', 'HorzBreak', 'RelSpeed', 'Date'])

        if movement_data.empty:
            st.write("No pitch movement data available for plotting.")
            return

        # Define Plotly color equivalents for pitch types
        plotly_color_dict = {
            'Fastball': 'royalblue',
            'Four-Seam': 'royalblue',
            'Sinker': 'goldenrod',
            'Slider': 'mediumseagreen',
            'Curveball': 'firebrick',
            'Cutter': 'darkorange',
            'ChangeUp': 'mediumpurple',
            'Changeup': 'mediumpurple',
            'Splitter': 'teal',
            'Unknown': 'black',
            'Other': 'black'
        }

        # Create Plotly figure
        fig = go.Figure()

        # Add **background reference lines** first so they appear **below** the scatter points
        fig.add_shape(
            type="line",
            x0=0, x1=0, y0=-25, y1=25,
            line=dict(color="black", width=2),
            layer="below"  # Keeps the line in the background
        )
        fig.add_shape(
            type="line",
            x0=-25, x1=25, y0=0, y1=0,
            line=dict(color="black", width=2),
            layer="below"  # Keeps the line in the background
        )

        # Ensure the black x-axis line stays visible
        fig.update_xaxes(
            zeroline=True, zerolinewidth=2, zerolinecolor='black'  # Forces the x-axis black line
        )
        fig.update_yaxes(
            zeroline=True, zerolinewidth=2, zerolinecolor='black'  # Forces the y-axis black line
        )

        # Get unique pitch types
        unique_pitch_types = movement_data['TaggedPitchType'].unique()

        for pitch_type in unique_pitch_types:
            pitch_data = movement_data[movement_data['TaggedPitchType'] == pitch_type]

            # Round numeric values for hover info
            pitch_data['RelSpeed'] = pitch_data['RelSpeed'].round(1)
            pitch_data['InducedVertBreak'] = pitch_data['InducedVertBreak'].round(1)
            pitch_data['HorzBreak'] = pitch_data['HorzBreak'].round(1)


            # Add scatter points **AFTER** the border lines to keep them on top
            fig.add_trace(go.Scatter(
                x=pitch_data['HorzBreak'],
                y=pitch_data['InducedVertBreak'],
                mode='markers',
                name=pitch_type,
                marker=dict(
                    size=9,  # Slightly larger for better visibility
                    color=plotly_color_dict.get(pitch_type, 'black'),
                    opacity=0.8,
                    line=dict(width=1, color="white")  # White edge for better contrast
                ),
                text=pitch_data.apply(
                    lambda row: f"Date: {row['Date']}<br>RelSpeed: {row['RelSpeed']}<br>iVB: {row['InducedVertBreak']}<br>HB: {row['HorzBreak']}<br>Spin: {row['SpinRate']}<br>Pitch#: {row['PitchNo']}",
                    axis=1
                ),
                hoverinfo='text'
            ))

        # Customize layout
        fig.update_layout(
            title=f"Pitch Movement for {pitcher_name}",
            xaxis=dict(title="Horizontal Break (inches)", range=[-30, 30]),
            yaxis=dict(title="Induced Vertical Break (inches)", range=[-30, 30]),
            template="plotly_white",
            showlegend=True,
            width=800,
            height=700
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while generating the pitch movement graph: {e}")






# Function to generate the Batted Ball table
def generate_batted_ball_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on the provided filters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Categorize batted balls into types based on angle
        def categorize_batted_type(angle):
            if angle < 10:
                return "GroundBall"
            elif 10 <= angle < 25:
                return "LineDrive"
            elif 25 <= angle < 50:
                return "FlyBall"
            else:
                return "PopUp"

        # Create 'BattedType' column
        pitcher_data['BattedType'] = pitcher_data['Angle'].apply(categorize_batted_type)

        # Filter rows where PitchCall is 'InPlay' for BIP calculations
        batted_data = pitcher_data[pitcher_data['PitchCall'] == 'InPlay']

        # Group by pitch type and calculate metrics
        batted_ball_summary = batted_data.groupby('TaggedPitchType').agg(
            BIP=('PitchCall', 'size'),
            GB=('BattedType', lambda x: (x == "GroundBall").sum()),
            FB=('BattedType', lambda x: (x == "FlyBall").sum()),
            EV=('ExitSpeed', 'mean'),
            Hard=('ExitSpeed', lambda x: (x >= 95).sum()),
            Soft=('ExitSpeed', lambda x: (x < 95).sum())
        ).reset_index()

        # Ensure all pitch types are included
        unique_pitch_types = pitcher_data['TaggedPitchType'].unique()
        full_summary = pd.DataFrame({'TaggedPitchType': unique_pitch_types})
        batted_ball_summary = pd.merge(full_summary, batted_ball_summary, on='TaggedPitchType', how='left')

        # Fill missing values with defaults
        batted_ball_summary[['BIP', 'GB', 'FB', 'EV', 'Hard', 'Soft']] = batted_ball_summary[
            ['BIP', 'GB', 'FB', 'EV', 'Hard', 'Soft']
        ].fillna(0)

        # Add total pitch counts for each type
        pitch_counts = pitcher_data.groupby('TaggedPitchType')['PitchCall'].count().reset_index(name='Count')
        batted_ball_summary = pd.merge(batted_ball_summary, pitch_counts, on='TaggedPitchType', how='left')

        # Calculate percentages
        batted_ball_summary['GB%'] = ((batted_ball_summary['GB'] / batted_ball_summary['BIP']) * 100).fillna(0).round(1).astype(str) + '%'
        batted_ball_summary['FB%'] = ((batted_ball_summary['FB'] / batted_ball_summary['BIP']) * 100).fillna(0).round(1).astype(str) + '%'
        batted_ball_summary['Hard%'] = ((batted_ball_summary['Hard'] / batted_ball_summary['BIP']) * 100).fillna(0).round(1).astype(str) + '%'
        batted_ball_summary['Soft%'] = ((batted_ball_summary['Soft'] / batted_ball_summary['BIP']) * 100).fillna(0).round(1).astype(str) + '%'

        # Calculate Contact%
        def calculate_contact(df):
            swings = df[df['PitchCall'].isin(['StrikeSwinging', 'InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])].shape[0]
            contact = df[df['PitchCall'].isin(['InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])].shape[0]
            return (contact / swings * 100) if swings > 0 else 0

        contact_values = []
        for pitch_type in batted_ball_summary['TaggedPitchType']:
            contact_values.append(
                calculate_contact(pitcher_data[pitcher_data['TaggedPitchType'] == pitch_type])
            )
        batted_ball_summary['Contact%'] = [f"{round(val, 1)}%" for val in contact_values]

        # Drop intermediate columns
        batted_ball_summary = batted_ball_summary.drop(columns=['GB', 'FB', 'Hard', 'Soft'])

        # Rename columns for display
        rename_columns = {
            'TaggedPitchType': 'Pitch',
            'Count': 'Count',
            'BIP': 'BIP',
            'EV': 'EV',
            'GB%': 'GB%',
            'FB%': 'FB%',
            'Hard%': 'Hard%',
            'Soft%': 'Soft%',
            'Contact%': 'Contact%'
        }
        batted_ball_summary = batted_ball_summary.rename(columns=rename_columns)

        # Calculate "All" row
        all_row = {
            'Pitch': 'All',
            'Count': pitcher_data.shape[0],
            'BIP': batted_data.shape[0],
            'EV': batted_data['ExitSpeed'].mean() if batted_data.shape[0] > 0 else 0,
            'GB%': f"{round((batted_data['BattedType'] == 'GroundBall').sum() / batted_data.shape[0] * 100, 1) if batted_data.shape[0] > 0 else 0}%",
            'FB%': f"{round((batted_data['BattedType'] == 'FlyBall').sum() / batted_data.shape[0] * 100, 1) if batted_data.shape[0] > 0 else 0}%",
            'Hard%': f"{round((batted_data['ExitSpeed'] >= 95).sum() / batted_data.shape[0] * 100, 1) if batted_data.shape[0] > 0 else 0}%",
            'Soft%': f"{round((batted_data['ExitSpeed'] < 95).sum() / batted_data.shape[0] * 100, 1) if batted_data.shape[0] > 0 else 0}%",
            'Contact%': f"{round(calculate_contact(pitcher_data), 1)}%"
        }
        all_row_df = pd.DataFrame([all_row])
        batted_ball_summary = pd.concat([batted_ball_summary, all_row_df], ignore_index=True)

        # Format the data for display
        formatted_data = format_dataframe(batted_ball_summary)

        # Display the table in Streamlit
        st.subheader("Batted Ball Summary")
        st.dataframe(formatted_data)

    except Exception as e:
        st.error(f"Error generating batted ball table: {e}")



import plotly.express as px



def generate_rolling_line_graphs(
    rolling_df, pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date
):
    try:
        # Sidebar toggle to choose between Full Rolling or Pitch-by-Pitch
        if date_filter_option == "Single Date":
            view_option = st.sidebar.radio(
                "Select Rolling View:",
                options=["Full Dataset Rolling Averages", "Pitch-by-Pitch (Single Date)"],
                index=0  # Default to full dataset
            )
        else:
            view_option = "Full Dataset Rolling Averages"  # Default for other date selections

        # Filter only by pitcher name (keep full data for rolling averages)
        full_filtered_data = rolling_df[rolling_df['Pitcher'] == pitcher_name]

        if full_filtered_data.empty:
            st.write("No data available for the selected pitcher.")
            return

        # Ensure numeric conversion for selected metrics
        numeric_columns = {
            'RelSpeed': 'Velocity',
            'InducedVertBreak': 'iVB',
            'HorzBreak': 'HB',
            'SpinRate': 'Spin',
            'Extension': 'Extension',
            'StuffPlus': 'StuffPlus'
        }
        for col in numeric_columns.keys():
            full_filtered_data[col] = pd.to_numeric(full_filtered_data[col], errors='coerce')

        # Convert Date column to datetime and drop NaN dates
        full_filtered_data['Date'] = pd.to_datetime(full_filtered_data['Date'], errors='coerce')
        full_filtered_data = full_filtered_data.dropna(subset=['Date'])

       

        # Get unique pitch types
        unique_pitch_types = full_filtered_data['TaggedPitchType'].unique()

        # Define color mapping
        color_dict = {
            'Fastball': 'blue',
            'Sinker': 'gold',
            'Slider': 'green',
            'Curveball': 'red',
            'Cutter': 'orange',
            'ChangeUp': 'purple',
            'Changeup': 'purple',
            'Splitter': 'teal',
            'Unknown': 'black',
            'Other': 'black'
        }

        ### **Option 1: Full Dataset Rolling Averages**
        if view_option == "Full Dataset Rolling Averages":
            # Sort data by date for proper rolling trend
            rolling_data = (
                full_filtered_data.groupby(['Date', 'TaggedPitchType'])
                .agg({col: 'mean' for col in numeric_columns.keys()})
                .reset_index()
                .sort_values(by="Date")
            )

            st.subheader("Rolling Averages Across Full Database")

            for metric, metric_label in numeric_columns.items():
                fig = px.line(
                    rolling_data,
                    x="Date",
                    y=metric,
                    color="TaggedPitchType",
                    title=f"{metric_label} Rolling Averages by Pitch Type (Full Dataset)",
                    labels={"Date": "Date", metric: metric_label, "TaggedPitchType": "Pitch Type"},
                    color_discrete_map=color_dict,
                    hover_data={"Date": "|%B %d, %Y", metric: ":.2f"},
                )

                # Scatter points for each date
                for pitch_type in unique_pitch_types:
                    pitch_subset = rolling_data[rolling_data['TaggedPitchType'] == pitch_type]
                    fig.add_scatter(
                        x=pitch_subset['Date'],
                        y=pitch_subset[metric],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=color_dict.get(pitch_type, 'black'),
                            opacity=0.7
                        ),
                        name=f"{pitch_type} Dots",
                        showlegend=False
                    )

                # Highlight selected date(s)
                if date_filter_option == "Single Date" and selected_date:
                    selected_datetime = pd.to_datetime(selected_date)
                    fig.add_vrect(x0=selected_datetime, x1=selected_datetime, fillcolor="gray", opacity=0.3, line_width=0)
                elif date_filter_option == "Date Range" and start_date and end_date:
                    start_datetime, end_datetime = pd.to_datetime(start_date), pd.to_datetime(end_date)
                    fig.add_vrect(x0=start_datetime, x1=end_datetime, fillcolor="gray", opacity=0.3, line_width=0)

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=metric_label,
                    legend_title="Pitch Type",
                    template="plotly_white",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

        ### **Option 2: Pitch-by-Pitch View (Only for Single Date)**
        elif view_option == "Pitch-by-Pitch (Single Date)":
            # Filter data by selected date
            selected_datetime = pd.to_datetime(selected_date)
            pitch_data = full_filtered_data[full_filtered_data['Date'].dt.date == selected_datetime.date()]

            if pitch_data.empty:
                st.write("No data available for the selected date.")
                return

            # Ensure PitchNo is numeric and sort
            pitch_data['PitchNo'] = pd.to_numeric(pitch_data['PitchNo'], errors='coerce')
            pitch_data = pitch_data.dropna(subset=['PitchNo']).sort_values(by="PitchNo")

            st.subheader(f"Pitch-by-Pitch View for {selected_date.strftime('%B %d, %Y')}")

            for metric, metric_label in numeric_columns.items():
                fig = px.line(
                    pitch_data,
                    x="PitchNo",
                    y=metric,
                    color="TaggedPitchType",
                    title=f"{metric_label} Pitch-by-Pitch",
                    labels={"PitchNo": "Pitch Number", metric: metric_label, "TaggedPitchType": "Pitch Type"},
                    color_discrete_map=color_dict,
                    hover_data={"PitchNo": ":.0f", metric: ":.2f"},
                )

                # Scatter points for each pitch
                for pitch_type in unique_pitch_types:
                    pitch_subset = pitch_data[pitch_data['TaggedPitchType'] == pitch_type]
                    fig.add_scatter(
                        x=pitch_subset['PitchNo'],
                        y=pitch_subset[metric],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_dict.get(pitch_type, 'black')
                        ),
                        name=f"{pitch_type} Dots",
                        showlegend=False
                    )

                # Set x-axis to match smallest to largest pitch number for the day
                fig.update_xaxes(range=[pitch_data['PitchNo'].min() - 1, pitch_data['PitchNo'].max() + 1])

                fig.update_layout(
                    xaxis_title="Pitch Number",
                    yaxis_title=metric_label,
                    legend_title="Pitch Type",
                    template="plotly_white",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while generating rolling line graphs: {e}")


plotly_color_dict = {
            'Fastball': 'royalblue',
            'Four-Seam': 'blue',
            'Sinker': 'goldenrod',
            'Slider': 'mediumseagreen',
            'Curveball': 'firebrick',
            'Cutter': 'darkorange',
            'ChangeUp': 'mediumpurple',
            'Changeup': 'mediumpurple',
            'Splitter': 'teal',
            'Unknown': 'black',
            'Other': 'black'
        }

import plotly.graph_objects as go

def plot_release_and_approach_angles(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on selected parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Drop NaN values for plotting
        release_data = pitcher_data.dropna(subset=['HorzRelAngle', 'VertRelAngle'])
        approach_data = pitcher_data.dropna(subset=['HorzApprAngle', 'VertApprAngle'])

        if release_data.empty and approach_data.empty:
            st.write("No angle data available for plotting.")
            return

        # Define Plotly color equivalents for pitch types
        plotly_color_dict = {
            'Fastball': 'royalblue',
            'Four-Seam': 'blue',
            'Sinker': 'goldenrod',
            'Slider': 'mediumseagreen',
            'Curveball': 'firebrick',
            'Cutter': 'darkorange',
            'ChangeUp': 'mediumpurple',
            'Changeup': 'mediumpurple',
            'Splitter': 'teal',
            'Unknown': 'black',
            'Other': 'black'
        }

        # Function to create a scatter plot with bounding circles and average values
        def create_scatter_plot(data, x_col, y_col, title, x_lim, y_lim):
            fig = go.Figure()

            # Get unique pitch types
            unique_pitch_types = data['TaggedPitchType'].unique()

            for pitch_type in unique_pitch_types:
                pitch_type_data = data[data['TaggedPitchType'] == pitch_type]

                # Calculate mean and standard deviation for bounding circle
                mean_x = pitch_type_data[x_col].mean()
                mean_y = pitch_type_data[y_col].mean()
                std_dev_x = pitch_type_data[x_col].std()
                std_dev_y = pitch_type_data[y_col].std()

                # Format average values for legend
                avg_label = f"{pitch_type} ({mean_x:.1f}, {mean_y:.1f})"

                # Plot scatter points
                fig.add_trace(go.Scatter(
                    x=pitch_type_data[x_col],
                    y=pitch_type_data[y_col],
                    mode='markers',
                    name=avg_label,  # Use formatted label
                    marker=dict(
                        size=8,
                        color=plotly_color_dict.get(pitch_type, 'black'),
                        opacity=0.7
                    )
                ))

                # Draw bounding circle if data exists
                if not (pd.isna(mean_x) or pd.isna(mean_y) or pd.isna(std_dev_x) or pd.isna(std_dev_y)):
                    radius = max(std_dev_x, std_dev_y)  # Use the largest std dev
                    fig.add_shape(
                        type="circle",
                        xref="x", yref="y",
                        x0=mean_x - radius, y0=mean_y - radius,
                        x1=mean_x + radius, y1=mean_y + radius,
                        line=dict(color=plotly_color_dict.get(pitch_type, 'black'), width=2),
                        opacity=0.3
                    )

            # Customize layout with limits
            fig.update_layout(
                title=title,
                xaxis=dict(title=x_col, range=x_lim),
                yaxis=dict(title=y_col, range=y_lim),
                template="plotly_white",
                showlegend=True,
                width=800,
                height=700
            )

            return fig

        # Create and display the release angle plot
        if not release_data.empty:
            release_fig = create_scatter_plot(
                release_data, 
                'HorzRelAngle', 'VertRelAngle', 
                "Release Angles by Pitch Type", 
                x_lim=[-7.5, 7.5], 
                y_lim=[-5, 3]
            )
            st.plotly_chart(release_fig, use_container_width=True)

        # Create and display the approach angle plot
        if not approach_data.empty:
            approach_fig = create_scatter_plot(
                approach_data, 
                'HorzApprAngle', 'VertApprAngle', 
                "Approach Angles by Pitch Type", 
                x_lim=[-6, 6], 
                y_lim=[-12, 0]
            )
            st.plotly_chart(approach_fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while generating the angle plots: {e}")



# Define Plotly color equivalents for pitch types (make available globally)
plotly_color_dict = {
    'Fastball': 'royalblue',
    'Four-Seam': 'blue',
    'Sinker': 'goldenrod',
    'Slider': 'mediumseagreen',
    'Curveball': 'firebrick',
    'Cutter': 'darkorange',
    'ChangeUp': 'mediumpurple',
    'Changeup': 'mediumpurple',
    'Splitter': 'teal',
    'Unknown': 'black',
    'Other': 'black'
}


# Function to create a scatter plot with bounding circles and average values
def create_scatter_plot(data, x_col, y_col, title, x_lim, y_lim):
    fig = go.Figure()

    # Get unique pitch types
    unique_pitch_types = data['TaggedPitchType'].unique()

    for pitch_type in unique_pitch_types:
        pitch_type_data = data[data['TaggedPitchType'] == pitch_type]

        # Calculate mean and standard deviation for bounding circle
        mean_x = pitch_type_data[x_col].mean()
        mean_y = pitch_type_data[y_col].mean()
        std_dev_x = pitch_type_data[x_col].std()
        std_dev_y = pitch_type_data[y_col].std()

        # Format average values for legend
        avg_label = f"{pitch_type} ({mean_x:.1f}, {mean_y:.1f})"

        # Plot scatter points
        fig.add_trace(go.Scatter(
            x=pitch_type_data[x_col],
            y=pitch_type_data[y_col],
            mode='markers',
            name=avg_label,
            marker=dict(
                size=8,
                color=plotly_color_dict.get(pitch_type, 'black'),
                opacity=0.7
            )
        ))

        # Draw bounding circle if data exists
        if not (pd.isna(mean_x) or pd.isna(mean_y) or pd.isna(std_dev_x) or pd.isna(std_dev_y)):
            radius = max(std_dev_x, std_dev_y)
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=mean_x - radius, y0=mean_y - radius,
                x1=mean_x + radius, y1=mean_y + radius,
                line=dict(color=plotly_color_dict.get(pitch_type, 'black'), width=2),
                opacity=0.3
            )

    # Customize layout with limits
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_col, range=x_lim),
        yaxis=dict(title=y_col, range=y_lim),
        template="plotly_white",
        showlegend=True,
        width=800,
        height=700
    )

    return fig  # <-- This was over-indented




# Generate heatmaps based on selections
# Generate heatmaps based on selections
plot_heatmaps(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date,
    heatmap_type  # Pass the selected heatmap type
)


# Generate and display the pitch traits and plate discipline tables
generate_plate_discipline_table(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date
)

generate_pitch_traits_table(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date
)

generate_batted_ball_table(
    pitcher_name,
    batter_side,
    strikes,
    balls,
    date_filter_option,
    selected_date,
    start_date,
    end_date
)


# Call the function in your Streamlit app
plot_pitch_movement(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date
)


# Generate rolling line graphs based on selected metrics and pitch types
generate_rolling_line_graphs(
    rolling_df,
    pitcher_name,
    batter_side,
    strikes,
    balls,
    date_filter_option,
    selected_date,
    start_date,
    end_date
)


# Call the function at the bottom of your Streamlit app
plot_release_and_approach_angles(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date
)
