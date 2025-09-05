import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from datetime import datetime
import math
from typing import Dict, List, Optional, Tuple


# === CONFIGURATION ===
class Config:
    """Centralized configuration for the application"""
    
    # File paths
    DATA_FILE = "Fall_2025_wRV_MASTER.csv"
    TEAM_FILTER = 'OLE_REB'
    
    # Pitch type color mapping
    PITCH_COLORS = {
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
    
    # Numeric columns for processing
    NUMERIC_COLUMNS = [
        'RelSpeed', 'SpinRate', 'Tilt', 'RelHeight', 'RelSide', 
        'Extension', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle', 'ExitSpeed'
    ]
    
    # Strike zone parameters
    STRIKE_ZONE = {
        'width': 1.66166,
        'x_start': -1.66166 / 2,
        'y_start': 1.5,
        'height': 3.3775 - 1.5
    }
    
    # Swing-related pitch calls
    SWING_CALLS = ['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay']
    
    # Strike-related pitch calls  
    STRIKE_CALLS = ['StrikeCalled', 'FoulBallFieldable', 'FoulBallNotFieldable', 'StrikeSwinging', 'InPlay']


# === DATA MANAGER ===
class DataManager:
    """Handles all data loading and caching operations"""
    
    @staticmethod
    @st.cache_data
    def load_data(file_path: str) -> pd.DataFrame:
        """Load and preprocess the main dataset"""
        df = pd.read_csv(file_path)
        
        # Process date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[df['Date'].notna()]
        
        # Filter by team
        if 'PitcherTeam' in df.columns:
            df = df[df['PitcherTeam'] == Config.TEAM_FILTER]
        
        # Convert numeric columns
        for col in Config.NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add season identifier
        df['Season'] = '2025 Season'
        
        return df
    
    @staticmethod
    def filter_data(
        df: pd.DataFrame,
        pitcher_name: str,
        batter_side: str = 'Both',
        strikes: str = 'All',
        balls: str = 'All',
        date_filter_option: str = "All",
        selected_date: Optional[datetime] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Apply filters to the dataset"""
        filtered_df = df[df['Pitcher'] == pitcher_name].copy()
        
        # Batter side filter
        if batter_side != 'Both':
            filtered_df = filtered_df[filtered_df['BatterSide'] == batter_side]
        
        # Count filters
        if strikes != 'All':
            filtered_df = filtered_df[filtered_df['Strikes'] == strikes]
        if balls != 'All':
            filtered_df = filtered_df[filtered_df['Balls'] == balls]
        
        # Date filters
        if date_filter_option == "Single Date" and selected_date:
            filtered_df = filtered_df[filtered_df['Date'].dt.date == pd.to_datetime(selected_date).date()]
        elif date_filter_option == "Date Range" and start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['Date'] >= pd.to_datetime(start_date)) & 
                (filtered_df['Date'] <= pd.to_datetime(end_date))
            ]
        
        return filtered_df


# === UTILITY FUNCTIONS ===
class Utils:
    """Utility functions for calculations and formatting"""
    
    @staticmethod
    def calculate_in_zone(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pitches in the strike zone"""
        sz = Config.STRIKE_ZONE
        return df[
            (df['PlateLocHeight'] >= sz['y_start']) & 
            (df['PlateLocHeight'] <= sz['y_start'] + sz['height']) & 
            (df['PlateLocSide'] >= sz['x_start']) & 
            (df['PlateLocSide'] <= -sz['x_start'])
        ]
    
    @staticmethod
    def categorize_batted_type(angle: float) -> str:
        """Categorize batted ball type based on launch angle"""
        if pd.isna(angle):
            return "Unknown"
        elif angle < 10:
            return "GroundBall"
        elif 10 <= angle < 25:
            return "LineDrive"
        elif 25 <= angle < 50:
            return "FlyBall"
        else:
            return "PopUp"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Format a decimal as a percentage string"""
        if pd.isna(value):
            return "N/A"
        return f"{round(value, 1)}%"
    
    @staticmethod
    def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Format dataframe for display"""
        df = df.copy()
        percent_columns = ['InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%', 'Strike%', 'FP Strike%', 'Pitch%', 'GB%', 'FB%', 'Hard%', 'Soft%', 'Contact%']
        
        for col in df.columns:
            if col in percent_columns:
                df[col] = df[col].apply(lambda x: f"{round(x, 1)}%" if pd.notna(x) and isinstance(x, (int, float)) else 'N/A')
            elif df[col].dtype.kind in 'f':
                df[col] = df[col].apply(lambda x: round(x, 1) if pd.notna(x) else 'N/A')
            else:
                df[col] = df[col].fillna('N/A')
        
        return df


# === ANALYSIS COMPONENTS ===
class PitchAnalysis:
    """Components for pitch analysis and visualization"""
    
    @staticmethod
    def generate_pitch_traits_table(filtered_df: pd.DataFrame) -> None:
        """Generate pitch traits summary table"""
        try:
            if filtered_df.empty:
                st.write("No data available for the selected parameters.")
                return
            
            # Group by pitch type and calculate means
            grouped_data = filtered_df.groupby('TaggedPitchType').agg(
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
            
            # Rename columns
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
            
            # Add StuffPlus data if available
            if 'StuffPlus' in filtered_df.columns:
                stuff_plus_data = filtered_df.groupby('TaggedPitchType')['StuffPlus'].mean().reset_index()
                grouped_data = pd.merge(grouped_data, stuff_plus_data, left_on='Pitch', right_on='TaggedPitchType', how='left')
                grouped_data['StuffPlus'] = pd.to_numeric(grouped_data['StuffPlus'], errors='coerce')
                grouped_data = grouped_data.drop(columns=['TaggedPitchType'])
            
            # Round numeric columns
            numeric_columns = ['Velo', 'iVB', 'HB', 'Spin', 'RelH', 'RelS', 'Ext', 'VAA']
            for col in numeric_columns:
                grouped_data[col] = pd.to_numeric(grouped_data[col], errors='coerce')
            grouped_data[numeric_columns] = grouped_data[numeric_columns].round(1)
            
            # Sort by count and add 'All' row
            grouped_data = grouped_data.sort_values(by='Count', ascending=False)
            
            # Calculate weighted averages for 'All' row
            total_count = grouped_data["Count"].sum()
            weighted_averages = {}
            for col in numeric_columns:
                valid_data = grouped_data[col].dropna()
                valid_weights = grouped_data.loc[grouped_data[col].notna(), "Count"]
                if len(valid_data) > 0:
                    weighted_averages[col] = np.average(valid_data, weights=valid_weights)
                else:
                    weighted_averages[col] = np.nan
            
            # StuffPlus weighted average
            stuff_plus_weighted_avg = np.nan
            if 'StuffPlus' in grouped_data.columns:
                valid_stuff_plus = grouped_data.loc[grouped_data["StuffPlus"].notna(), "StuffPlus"]
                valid_stuff_plus_weights = grouped_data.loc[grouped_data["StuffPlus"].notna(), "Count"]
                if len(valid_stuff_plus) > 0:
                    stuff_plus_weighted_avg = np.average(valid_stuff_plus, weights=valid_stuff_plus_weights)
            
            # Create 'All' row
            all_row = {
                'Pitch': 'All',
                'Count': total_count,
                **{col: round(weighted_averages[col], 1) if pd.notna(weighted_averages[col]) else 'N/A' for col in numeric_columns}
            }
            
            if 'StuffPlus' in grouped_data.columns:
                all_row['StuffPlus'] = round(stuff_plus_weighted_avg, 1) if pd.notna(stuff_plus_weighted_avg) else "N/A"
            
            grouped_data = pd.concat([grouped_data, pd.DataFrame([all_row])], ignore_index=True)
            
            # Display table
            formatted_data = Utils.format_dataframe(grouped_data)
            st.subheader("Pitch Traits:")
            st.dataframe(formatted_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating pitch traits table: {e}")
    
    @staticmethod
    def generate_plate_discipline_table(filtered_df: pd.DataFrame) -> None:
        """Generate plate discipline metrics table"""
        try:
            if filtered_df.empty:
                st.write("No data available for the selected parameters.")
                return
            
            def calculate_metrics(df):
                """Calculate plate discipline metrics for a subset of data"""
                in_zone_pitches = Utils.calculate_in_zone(df)
                total_in_zone = len(in_zone_pitches)
                
                # First pitch logic
                fp_df = df[(df['Balls'] == 0) & (df['Strikes'] == 0)]
                fp_total = len(fp_df)
                fp_strikes = fp_df[~fp_df['PitchCall'].isin(['HitByPitch', 'BallCalled', 'BallInDirt', 'BallinDirt'])].shape[0]
                fp_strike_pct = (fp_strikes / fp_total) * 100 if fp_total > 0 else 0
                
                # Swing and strike calculations
                total_swings = df[df['PitchCall'].isin(Config.SWING_CALLS)].shape[0]
                total_whiffs = df[df['PitchCall'] == 'StrikeSwinging'].shape[0]
                total_chase = df[(~df.index.isin(in_zone_pitches.index)) & df['PitchCall'].isin(Config.SWING_CALLS)].shape[0]
                in_zone_whiffs = in_zone_pitches[in_zone_pitches['PitchCall'] == 'StrikeSwinging'].shape[0]
                total_strikes = df[df['PitchCall'].isin(Config.STRIKE_CALLS)].shape[0]
                
                return {
                    'InZone%': (total_in_zone / len(df)) * 100 if len(df) > 0 else 0,
                    'Swing%': (total_swings / len(df)) * 100 if len(df) > 0 else 0,
                    'Whiff%': (total_whiffs / total_swings) * 100 if total_swings > 0 else 0,
                    'Chase%': (total_chase / total_swings) * 100 if total_swings > 0 else 0,
                    'InZoneWhiff%': (in_zone_whiffs / total_in_zone) * 100 if total_in_zone > 0 else 0,
                    'Strike%': (total_strikes / len(df)) * 100 if len(df) > 0 else 0,
                    'FP Strike%': fp_strike_pct
                }
            
            # Calculate by pitch type
            plate_discipline_data = filtered_df.groupby('TaggedPitchType').apply(lambda x: pd.Series(calculate_metrics(x))).reset_index()
            
            # Add counts and pitch percentages
            pitch_counts = filtered_df.groupby('TaggedPitchType').size().reset_index(name='Count')
            plate_discipline_data = pd.merge(plate_discipline_data, pitch_counts, on='TaggedPitchType', how='left')
            total_pitches = len(filtered_df)
            plate_discipline_data['Pitch%'] = (plate_discipline_data['Count'] / total_pitches) * 100
            
            # Reorder and rename columns
            plate_discipline_data = plate_discipline_data.rename(columns={'TaggedPitchType': 'Pitch'})
            column_order = ['Pitch', 'Count', 'Pitch%', 'Strike%', 'InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%', 'FP Strike%']
            plate_discipline_data = plate_discipline_data[column_order]
            
            # Add 'All' row
            all_metrics = calculate_metrics(filtered_df)
            all_row = {
                'Pitch': 'All',
                'Count': total_pitches,
                'Pitch%': 100.0,
                **all_metrics
            }
            plate_discipline_data = pd.concat([plate_discipline_data, pd.DataFrame([all_row])], ignore_index=True)
            
            # Display table
            formatted_data = Utils.format_dataframe(plate_discipline_data)
            st.subheader("Plate Discipline:")
            st.dataframe(formatted_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating plate discipline table: {e}")
    
    @staticmethod
    def generate_batted_ball_table(filtered_df: pd.DataFrame) -> None:
        """Generate batted ball summary table"""
        try:
            if filtered_df.empty:
                st.write("No data available for the selected parameters.")
                return
            
            # Add batted ball categorization
            filtered_df['BattedType'] = filtered_df['Angle'].apply(Utils.categorize_batted_type)
            
            # Filter for balls in play
            batted_data = filtered_df[filtered_df['PitchCall'] == 'InPlay']
            
            # Calculate contact percentage by pitch type
            def calculate_contact(df):
                swings = df[df['PitchCall'].isin(Config.SWING_CALLS)].shape[0]
                contact = df[df['PitchCall'].isin(['InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])].shape[0]
                return (contact / swings * 100) if swings > 0 else 0
            
            # Group by pitch type for batted ball metrics
            batted_ball_summary = batted_data.groupby('TaggedPitchType').agg(
                BIP=('PitchCall', 'size'),
                GB=('BattedType', lambda x: (x == "GroundBall").sum()),
                FB=('BattedType', lambda x: (x == "FlyBall").sum()),
                EV=('ExitSpeed', 'mean'),
                Hard=('ExitSpeed', lambda x: (x >= 95).sum()),
                Soft=('ExitSpeed', lambda x: (x < 95).sum())
            ).reset_index()
            
            # Ensure all pitch types are included
            unique_pitch_types = filtered_df['TaggedPitchType'].unique()
            full_summary = pd.DataFrame({'TaggedPitchType': unique_pitch_types})
            batted_ball_summary = pd.merge(full_summary, batted_ball_summary, on='TaggedPitchType', how='left')
            batted_ball_summary = batted_ball_summary.fillna(0)
            
            # Add total pitch counts
            pitch_counts = filtered_df.groupby('TaggedPitchType').size().reset_index(name='Count')
            batted_ball_summary = pd.merge(batted_ball_summary, pitch_counts, on='TaggedPitchType', how='left')
            
            # Calculate percentages
            batted_ball_summary['GB%'] = ((batted_ball_summary['GB'] / batted_ball_summary['BIP']) * 100).fillna(0)
            batted_ball_summary['FB%'] = ((batted_ball_summary['FB'] / batted_ball_summary['BIP']) * 100).fillna(0)
            batted_ball_summary['Hard%'] = ((batted_ball_summary['Hard'] / batted_ball_summary['BIP']) * 100).fillna(0)
            batted_ball_summary['Soft%'] = ((batted_ball_summary['Soft'] / batted_ball_summary['BIP']) * 100).fillna(0)
            
            # Calculate contact percentages
            contact_values = []
            for pitch_type in batted_ball_summary['TaggedPitchType']:
                contact_values.append(calculate_contact(filtered_df[filtered_df['TaggedPitchType'] == pitch_type]))
            batted_ball_summary['Contact%'] = contact_values
            
            # Clean up and rename columns
            batted_ball_summary = batted_ball_summary.drop(columns=['GB', 'FB', 'Hard', 'Soft'])
            batted_ball_summary = batted_ball_summary.rename(columns={'TaggedPitchType': 'Pitch'})
            
            # Add 'All' row
            all_row = {
                'Pitch': 'All',
                'Count': len(filtered_df),
                'BIP': len(batted_data),
                'EV': batted_data['ExitSpeed'].mean() if len(batted_data) > 0 else 0,
                'GB%': ((batted_data['BattedType'] == 'GroundBall').sum() / len(batted_data) * 100) if len(batted_data) > 0 else 0,
                'FB%': ((batted_data['BattedType'] == 'FlyBall').sum() / len(batted_data) * 100) if len(batted_data) > 0 else 0,
                'Hard%': ((batted_data['ExitSpeed'] >= 95).sum() / len(batted_data) * 100) if len(batted_data) > 0 else 0,
                'Soft%': ((batted_data['ExitSpeed'] < 95).sum() / len(batted_data) * 100) if len(batted_data) > 0 else 0,
                'Contact%': calculate_contact(filtered_df)
            }
            batted_ball_summary = pd.concat([batted_ball_summary, pd.DataFrame([all_row])], ignore_index=True)
            
            # Display table
            formatted_data = Utils.format_dataframe(batted_ball_summary)
            st.subheader("Batted Ball Summary:")
            st.dataframe(formatted_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating batted ball table: {e}")


# === VISUALIZATION COMPONENTS ===
class Visualizations:
    """Components for creating visualizations"""
    
    @staticmethod
    def plot_heatmaps(filtered_df: pd.DataFrame, map_type: str, pitcher_name: str) -> None:
        """Create location heatmaps"""
        try:
            plot_data = filtered_df.dropna(subset=['PlateLocSide', 'PlateLocHeight'])
            
            if plot_data.empty:
                st.write("No location data available for plotting.")
                return
            
            unique_pitch_types = plot_data['TaggedPitchType'].unique()
            n_pitch_types = len(unique_pitch_types)
            plots_per_row = 3
            n_rows = math.ceil(n_pitch_types / plots_per_row)
            
            fig_width = 12 * plots_per_row
            fig_height = 16 * n_rows
            
            fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(fig_width, fig_height))
            if n_pitch_types == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (ax, pitch_type) in enumerate(zip(axes, unique_pitch_types)):
                pitch_type_data = plot_data[plot_data['TaggedPitchType'] == pitch_type]
                
                # Determine heatmap data based on type
                if map_type == 'Frequency':
                    heatmap_data = pitch_type_data
                elif map_type == 'Whiff':
                    heatmap_data = pitch_type_data[pitch_type_data['PitchCall'] == 'StrikeSwinging']
                elif map_type == 'Exit Velocity':
                    heatmap_data = pitch_type_data
                else:
                    heatmap_data = pitch_type_data
                
                # Scatter plot for all pitches
                ax.scatter(
                    pitch_type_data['PlateLocSide'], 
                    pitch_type_data['PlateLocHeight'], 
                    color='black',
                    edgecolor='white',
                    s=300,
                    alpha=0.7
                )
                
                # Add heatmap if enough data
                if len(heatmap_data) >= 5:
                    bw_adjust_value = 0.5 if len(heatmap_data) > 50 else 1
                    sns.kdeplot(
                        x=heatmap_data['PlateLocSide'], 
                        y=heatmap_data['PlateLocHeight'], 
                        fill=True, 
                        cmap='Spectral_r' if map_type == 'Frequency' else 'coolwarm', 
                        levels=6, 
                        ax=ax,
                        bw_adjust=bw_adjust_value
                    )
                
                # Add strike zone
                sz = Config.STRIKE_ZONE
                strike_zone = patches.Rectangle(
                    (sz['x_start'], sz['y_start']),
                    sz['width'],
                    sz['height'],
                    edgecolor='black',
                    facecolor='none',
                    linewidth=2
                )
                ax.add_patch(strike_zone)
                
                # Set limits and formatting
                ax.set_xlim(-2, 2)
                ax.set_ylim(1, 4)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title(f"{pitch_type} ({pitcher_name})", fontsize=24)
                ax.set_aspect('equal', adjustable='box')
            
            # Remove unused subplots
            for j in range(len(unique_pitch_types), len(axes)):
                fig.delaxes(axes[j])
            
            plt.suptitle(f"{pitcher_name} {map_type} Heatmap (2025 College Season)", 
                        fontsize=30, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating {map_type} heatmaps: {e}")
    
    @staticmethod
    def plot_pitch_movement(filtered_df: pd.DataFrame, pitcher_name: str) -> None:
        """Create pitch movement chart"""
        try:
            movement_data = filtered_df.dropna(subset=['InducedVertBreak', 'HorzBreak', 'RelSpeed', 'Date'])
            
            if movement_data.empty:
                st.write("No pitch movement data available for plotting.")
                return
            
            fig = go.Figure()
            
            # Add reference lines
            fig.add_hline(y=0, line=dict(color="black", width=2))
            fig.add_vline(x=0, line=dict(color="black", width=2))
            
            unique_pitch_types = movement_data['TaggedPitchType'].unique()
            
            for pitch_type in unique_pitch_types:
                pitch_data = movement_data[movement_data['TaggedPitchType'] == pitch_type]
                
                # Round numeric values for hover
                pitch_data = pitch_data.copy()
                pitch_data['RelSpeed'] = pitch_data['RelSpeed'].round(1)
                pitch_data['InducedVertBreak'] = pitch_data['InducedVertBreak'].round(1)
                pitch_data['HorzBreak'] = pitch_data['HorzBreak'].round(1)
                
                fig.add_trace(go.Scatter(
                    x=pitch_data['HorzBreak'],
                    y=pitch_data['InducedVertBreak'],
                    mode='markers',
                    name=pitch_type,
                    marker=dict(
                        size=9,
                        color=Config.PITCH_COLORS.get(pitch_type, 'black'),
                        opacity=0.8,
                        line=dict(width=1, color="white")
                    ),
                    text=pitch_data.apply(
                        lambda row: f"Date: {row['Date']}<br>Velocity: {row['RelSpeed']}<br>iVB: {row['InducedVertBreak']}<br>HB: {row['HorzBreak']}<br>Spin: {row['SpinRate']}<br>Pitch#: {row['PitchNo']}",
                        axis=1
                    ),
                    hoverinfo='text'
                ))
            
            fig.update_layout(
                title=f"Pitch Movement for {pitcher_name}",
                xaxis=dict(title="Horizontal Break (inches)", range=[-30, 30]),
                yaxis=dict(title="Induced Vertical Break (inches)", range=[-30, 30]),
                template="plotly_white",
                showlegend=True,
                width=800,
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating pitch movement chart: {e}")
    
    @staticmethod
    def plot_release_and_approach_angles(filtered_df: pd.DataFrame, pitcher_name: str) -> None:
        """Create release and approach angle plots"""
        try:
            release_data = filtered_df.dropna(subset=['HorzRelAngle', 'VertRelAngle'])
            approach_data = filtered_df.dropna(subset=['HorzApprAngle', 'VertApprAngle'])
            
            def create_angle_plot(data, x_col, y_col, title, x_lim, y_lim):
                fig = go.Figure()
                unique_pitch_types = data['TaggedPitchType'].unique()
                
                for pitch_type in unique_pitch_types:
                    pitch_type_data = data[data['TaggedPitchType'] == pitch_type]
                    
                    mean_x = pitch_type_data[x_col].mean()
                    mean_y = pitch_type_data[y_col].mean()
                    std_dev_x = pitch_type_data[x_col].std()
                    std_dev_y = pitch_type_data[y_col].std()
                    
                    avg_label = f"{pitch_type} ({mean_x:.1f}, {mean_y:.1f})"
                    
                    fig.add_trace(go.Scatter(
                        x=pitch_type_data[x_col],
                        y=pitch_type_data[y_col],
                        mode='markers',
                        name=avg_label,
                        marker=dict(
                            size=8,
                            color=Config.PITCH_COLORS.get(pitch_type, 'black'),
                            opacity=0.7
                        )
                    ))
                    
                    # Add bounding circle
                    if not any(pd.isna([mean_x, mean_y, std_dev_x, std_dev_y])):
                        radius = max(std_dev_x, std_dev_y)
                        fig.add_shape(
                            type="circle",
                            xref="x", yref="y",
                            x0=mean_x - radius, y0=mean_y - radius,
                            x1=mean_x + radius, y1=mean_y + radius,
                            line=dict(color=Config.PITCH_COLORS.get(pitch_type, 'black'), width=2),
                            opacity=0.3
                        )
                
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
            
            if not release_data.empty:
                release_fig = create_angle_plot(
                    release_data, 'HorzRelAngle', 'VertRelAngle', 
                    f"Release Angles by Pitch Type - {pitcher_name}", 
                    [-7.5, 7.5], [-5, 3]
                )
                st.plotly_chart(release_fig, use_container_width=True)
            
            if not approach_data.empty:
                approach_fig = create_angle_plot(
                    approach_data, 'HorzApprAngle', 'VertApprAngle', 
                    f"Approach Angles by Pitch Type - {pitcher_name}", 
                    [-6, 6], [-12, 0]
                )
                st.plotly_chart(approach_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating angle plots: {e}")


# === MAIN APPLICATION ===
def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title="Brewster Pitcher Reports",
        page_icon="⚾",
        layout="wide"
    )
    
    # Load data
    try:
        season_df = DataManager.load_data(Config.DATA_FILE)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # App title and sidebar
    st.title("Brewster Pitcher Reports (2025 CCBL Season)")
    st.sidebar.header("Filters")
    
    # Sidebar controls
    pitcher_name = st.sidebar.selectbox("Select Pitcher:", season_df['Pitcher'].unique())
    heatmap_type = st.sidebar.selectbox("Select Heatmap Type:", ["Frequency", "Whiff", "Exit Velocity"])
    batter_side = st.sidebar.selectbox("Select Batter Side:", ['Both', 'Right', 'Left'])
    strikes = st.sidebar.selectbox("Select Strikes:", ['All', 0, 1, 2])
    balls = st.sidebar.selectbox("Select Balls:", ['All', 0, 1, 2, 3])
    
    # Date filtering
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
    
    # Filter data
    filtered_df = DataManager.filter_data(
        season_df, pitcher_name, batter_side, strikes, balls,
        date_filter_option, selected_date, start_date, end_date
    )
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Display components
    try:
        # Heatmaps
        Visualizations.plot_heatmaps(filtered_df, heatmap_type, pitcher_name)
        
        # Analysis tables
        col1, col2 = st.columns(2)
        
        with col1:
            PitchAnalysis.generate_plate_discipline_table(filtered_df)
        
        with col2:
            PitchAnalysis.generate_pitch_traits_table(filtered_df)
        
        PitchAnalysis.generate_batted_ball_table(filtered_df)
        
        # Visualizations
        Visualizations.plot_pitch_movement(filtered_df, pitcher_name)
        Visualizations.plot_release_and_approach_angles(filtered_df, pitcher_name)
        
    except Exception as e:
        st.error(f"Error in main application: {e}")


if __name__ == "__main__":
    main()
