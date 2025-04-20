import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from utils.constants import TEAM_COLORS, ROOKIES
from base import ensure_visualization_folder, save_figure, get_team_color, adjust_color
from utils.helpers import ensure_directory
from utils.constants import DRIVERS

def create_driver_comparison(predictions, driver1, driver2, historical_data=None, save_path=None):
    """
    Create a comprehensive comparison visualization between two drivers.
    
    Args:
        predictions (DataFrame): Predicted race results
        driver1 (str): First driver name
        driver2 (str): Second driver name
        historical_data (DataFrame, optional): Historical race data for these drivers
        save_path (str, optional): Path to save the visualization
        
    Returns:
        None
    """
    # Check if drivers exist in predictions
    if driver1 not in predictions['Driver'].values:
        raise ValueError(f"Driver '{driver1}' not found in predictions")
    if driver2 not in predictions['Driver'].values:
        raise ValueError(f"Driver '{driver2}' not found in predictions")
    
    # Extract driver data
    driver1_data = predictions[predictions['Driver'] == driver1].iloc[0]
    driver2_data = predictions[predictions['Driver'] == driver2].iloc[0]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot 1: Basic comparison (position, grid, interval)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create comparison table
    comparison_data = {
        'Metric': ['Grid Position', 'Predicted Position', 'Position Change', 'Interval'],
        driver1: [
            f"P{int(driver1_data['Grid'])}", 
            f"P{int(driver1_data['Position'])}", 
            int(driver1_data['Grid'] - driver1_data['Position']), 
            driver1_data['Interval']
        ],
        driver2: [
            f"P{int(driver2_data['Grid'])}", 
            f"P{int(driver2_data['Position'])}", 
            int(driver2_data['Grid'] - driver2_data['Position']), 
            driver2_data['Interval']
        ]
    }
    
    # Format as table
    ax1.axis('tight')
    ax1.axis('off')
    ax1.table(
        cellText=[
            [comparison_data['Metric'][i], comparison_data[driver1][i], comparison_data[driver2][i]] 
            for i in range(len(comparison_data['Metric']))
        ],
        colLabels=['Metric', driver1, driver2],
        cellLoc='center',
        loc='center'
    )
    
    ax1.set_title('Basic Comparison', fontsize=14)
    
    # Plot 2: Position Change Visualization
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create arrows showing position change
    teams = [
        driver1_data['Team'],
        driver2_data['Team']
    ]
    
    colors = [
        TEAM_COLORS.get(driver1_data['Team'], 'skyblue'),
        TEAM_COLORS.get(driver2_data['Team'], 'orange')
    ]
    
    for i, driver in enumerate([driver1, driver2]):
        driver_data = predictions[predictions['Driver'] == driver].iloc[0]
        grid_pos = driver_data['Grid']
        pred_pos = driver_data['Position']
        
        ax2.plot([0, 1], [grid_pos, pred_pos], '-o', color=colors[i], linewidth=2, label=driver)
        ax2.text(0, grid_pos, f"P{int(grid_pos)}", ha='right', va='center')
        ax2.text(1, pred_pos, f"P{int(pred_pos)}", ha='left', va='center')
    
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Grid', 'Finish'])
    ax2.set_ylabel('Position')
    ax2.set_title('Position Change', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.invert_yaxis()  # Invert so P1 is at the top
    
    # Plot 3: Interval comparison
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Extract intervals
    intervals = [
        driver1_data['Interval (s)'] if driver1_data['Interval (s)'] > 0 else 0,
        driver2_data['Interval (s)'] if driver2_data['Interval (s)'] > 0 else 0
    ]
    
    # Create bar chart
    bars = ax3.bar([driver1, driver2], intervals, color=colors)
    
    # Add interval labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        interval_text = "WINNER" if height == 0 else f"+{height:.3f}s"
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1, interval_text, ha='center', va='bottom')
    
    ax3.set_ylabel('Interval to Winner (seconds)')
    ax3.set_title('Predicted Intervals', fontsize=14)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: Historical head-to-head (if data available)
    ax4 = fig.add_subplot(gs[1, :])
    
    if historical_data is not None and len(historical_data) > 0:
        # Process historical data
        h2h_summary = process_historical_head_to_head(historical_data, driver1, driver2)
        
        # Plot historical performance
        races = list(h2h_summary.keys())
        driver1_results = [h2h_summary[race][driver1] for race in races]
        driver2_results = [h2h_summary[race][driver2] for race in races]
        
        x = np.arange(len(races))
        width = 0.35
        
        ax4.bar(x - width/2, driver1_results, width, label=driver1, color=colors[0])
        ax4.bar(x + width/2, driver2_results, width, label=driver2, color=colors[1])
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(races, rotation=45, ha='right')
        ax4.set_ylabel('Position')
        ax4.set_title('Historical Head-to-Head Comparison', fontsize=14)
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.invert_yaxis()  # Invert y-axis so position 1 is at the top
    else:
        # No historical data
        ax4.text(0.5, 0.5, "No historical data available for comparison", 
                ha='center', va='center', fontsize=12)
        ax4.axis('off')
    
    # Add overall title
    plt.suptitle(f'Driver Comparison: {driver1} vs {driver2}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Driver comparison visualization saved to {save_path}")
    else:
        # Use default path
        output_dir = ensure_directory('visualizations')
        filename = f"{driver1.lower()}_{driver2.lower()}_comparison.png"
        plt.savefig(os.path.join(output_dir, filename))
        print(f"Driver comparison visualization saved to 'visualizations/{filename}'")
    
    plt.close()
    
def process_historical_head_to_head(historical_data, driver1, driver2):
    """
    Process historical data to extract head-to-head comparison.
    
    Args:
        historical_data (DataFrame): Historical race data
        driver1 (str): First driver name
        driver2 (str): Second driver name
        
    Returns:
        dict: Head-to-head summary by race
    """
    # Group data by race
    races = historical_data['RaceName'].unique()
    
    h2h_summary = {}
    
    for race in races:
        race_data = historical_data[historical_data['RaceName'] == race]
        
        # Get driver results
        driver1_result = race_data[race_data['Driver'] == driver1]['RacePosition'].values
        driver2_result = race_data[race_data['Driver'] == driver2]['RacePosition'].values
        
        # Store results if both drivers participated
        if len(driver1_result) > 0 and len(driver2_result) > 0:
            h2h_summary[race] = {
                driver1: driver1_result[0],
                driver2: driver2_result[0]
            }
    
    return h2h_summary
def plot_qualifying_deltas(predictions, folder_path=None):
    """
    Create a visualization of qualifying time deltas.
    
    Args:
        predictions (DataFrame): Predicted race results
        folder_path (str, optional): Folder to save the visualization in
        
    Returns:
        str: Path to the saved visualization
    """
    # Create visualization folder
    folder_path = ensure_visualization_folder(folder_path)
    
    # Sort by grid position (qualifying result)
    quali_results = predictions.sort_values('Grid').copy()
    
    # Calculate gap to pole (approximate based on QualifyingGapToPole if available)
    if 'QualifyingGapToPole' in quali_results.columns:
        pole_gap = quali_results['QualifyingGapToPole']
    else:
        # Approximate gaps based on grid position (about 0.1s per position)
        pole_gap = (quali_results['Grid'] - 1) * 0.1
    
    quali_results['QualiDelta'] = pole_gap
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bars
    team_colors = [get_team_color(team) for team in quali_results['Team']]
    bars = ax.barh(quali_results['Driver'], quali_results['QualiDelta'], 
                   color=team_colors, height=0.7, edgecolor='white')
    
    # Add custom grid
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Formatting
    ax.set_title('Qualifying Time Deltas', fontsize=14)
    ax.set_xlabel('Gap to Pole (seconds)', fontsize=12)
    
    # Add gap values as text
    for i, gap in enumerate(quali_results['QualiDelta']):
        # Only add text for non-zero gaps
        if gap > 0:
            ax.text(gap + 0.02, i, f"+{gap:.3f}s", va='center')
    
    # Invert y-axis to show pole at the top
    ax.invert_yaxis()
    
    # Save figure
    filepath = save_figure(fig, 'qualifying_deltas', folder_path)
    
    return filepath

# ----- Driver Analysis Visualizations -----

def plot_rookie_comparison(predictions, folder_path=None):
    """
    Create a visualization that highlights rookie performance compared to expectations.
    
    Args:
        predictions (DataFrame): Predicted race results
        folder_path (str, optional): Folder to save the visualization in
        
    Returns:
        str: Path to the saved visualization
    """
    # Create visualization folder
    folder_path = ensure_visualization_folder(folder_path)
    
    # Filter for rookies
    rookie_data = predictions[predictions['Driver'].isin(ROOKIES)].copy()
    
    if len(rookie_data) == 0:
        print("No rookies found in the predictions. Skipping rookie comparison plot.")
        return None
    
    # Calculate grid vs finish position delta
    rookie_data['PositionDelta'] = rookie_data['Grid'] - rookie_data['Position']
    
    # Determine color based on whether they gained or lost positions
    colors = []
    for delta in rookie_data['PositionDelta']:
        if delta > 0:
            colors.append('green')  # Gained positions
        elif delta < 0:
            colors.append('red')    # Lost positions
        else:
            colors.append('gray')   # Maintained position
def plot_sector_performance(predictions, sector_data=None):
    """
    Create visualization of sector performance for top drivers.
    
    Args:
        predictions (DataFrame): Predicted race results
        sector_data (DataFrame, optional): Sector time data
        
    Returns:
        None: Saves visualization to file
    """
    # Get visualization folder
    viz_folder = ensure_visualization_folder()
    
    # Get top 5 drivers
    top_drivers = predictions.head(5)
    
    # Set up figure
    plt.figure(figsize=(14, 10))
    
    if sector_data is not None and not sector_data.empty:
        # Using provided sector data
        drivers = top_drivers['Driver'].tolist()
        teams = top_drivers['Team'].tolist()
        
        # Calculate relative performance per sector
        sectors = ['Sector1Time', 'Sector2Time', 'Sector3Time']
        
        # Normalize sector times (lower is better)
        sector_times = []
        
        for sector in sectors:
            # Get sector data for the drivers
            driver_sectors = sector_data[sector_data['Driver'].isin(drivers)]
            
            # Get best time per sector
            best_time = driver_sectors[sector].min()
            
            # Calculate relative performance (% gap to best)
            driver_sectors[f'{sector}_rel'] = (driver_sectors[sector] / best_time - 1) * 100
            
            # Store relative performance
            for driver in drivers:
                driver_data = driver_sectors[driver_sectors['Driver'] == driver]
                if not driver_data.empty:
                    rel_time = driver_data[f'{sector}_rel'].iloc[0]
                    sector_times.append({
                        'Driver': driver,
                        'Sector': sector.replace('Time', ''),
                        'RelativeTime': rel_time
                    })
                else:
                    sector_times.append({
                        'Driver': driver,
                        'Sector': sector.replace('Time', ''),
                        'RelativeTime': np.nan
                    })
        
        # Convert to DataFrame
        sector_df = pd.DataFrame(sector_times)
    else:
        # Create synthetic sector data
        drivers = top_drivers['Driver'].tolist()
        teams = top_drivers['Team'].tolist()
        
        # Create sector times with different driver strengths
        sector_times = []
        
        for driver, team in DRIVERS.items():
            
            # Base disadvantage based on position (leader is reference)
            base_disadvantage = i * 0.15
            
            # Sector-specific performance
            sector1 = base_disadvantage
            sector2 = base_disadvantage
            sector3 = base_disadvantage
            
            # Add driver-specific characteristics
            if i == 1:  # Second driver
                sector1 = max(0, sector1 - 0.2)  # Better in S1
                sector3 = sector3 + 0.1  # Weaker in S3
            elif i == 2:  # Third driver
                sector2 = max(0, sector2 - 0.3)  # Better in S2
            elif i == 3:  # Fourth driver
                sector3 = max(0, sector3 - 0.25)  # Better in S3
                sector1 = sector1 + 0.1  # Weaker in S1
            
            # Add some randomness
            sector1 += np.random.uniform(-0.05, 0.05)
            sector2 += np.random.uniform(-0.05, 0.05)
            sector3 += np.random.uniform(-0.05, 0.05)
            
            # Add to dataframe
            sector_times.extend([
                {'Driver': driver, 'Sector': 'Sector1', 'RelativeTime': sector1},
                {'Driver': driver, 'Sector': 'Sector2', 'RelativeTime': sector2},
                {'Driver': driver, 'Sector': 'Sector3', 'RelativeTime': sector3}
            ])
        
        # Convert to DataFrame
        sector_df = pd.DataFrame(sector_times)
    
    # Create heatmap data
    heatmap_data = sector_df.pivot(index='Driver', columns='Sector', values='RelativeTime')
    
    # Calculate average performance across sectors for sorting
    driver_avg = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.loc[driver_avg.sort_values().index]
    
    # Create a custom color palette based on team colors
    driver_colors = []
    for driver in heatmap_data.index:
        team = top_drivers[top_drivers['Driver'] == driver]['Team'].iloc[0]
        color = TEAM_COLORS.get(team, 'gray')
        driver_colors.append(color)
    
    # Create the heatmap
    ax = plt.subplot(111)
    
    # Custom diverging colormap (green=good, red=bad)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap=cmap, center=0, fmt='.2f',
                linewidths=0.5, cbar_kws={'label': '% Gap to Best'})
    
    # Add custom colored boxes for each driver
    for i, driver in enumerate(heatmap_data.index):
        team = top_drivers[top_drivers['Driver'] == driver]['Team'].iloc[0]
        color = TEAM_COLORS.get(team, 'gray')
        
        # Add colored box at the left of the row
        plt.plot([0], [i + 0.5], marker='s', markersize=15, color=color)
    
    plt.title('Sector Performance Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, 'sector_performance.png'))
    plt.close()

def plot_race_pace_evolution(predictions, lap_time_data=None):
    """
    Visualize how race pace evolves throughout the race for top drivers.
    Shows fuel effect, tire degradation, and strategic pace management.
    
    Args:
        predictions (DataFrame): Predicted race results
        lap_time_data (DataFrame, optional): Historical lap time data
        
    Returns:
        None: Saves visualization to file
    """
    # Get visualization folder
    viz_folder = ensure_visualization_folder()
    
    # Get top 6 drivers for visualization
    top_drivers = predictions.head(6)
    
    plt.figure(figsize=(14, 8))
    
    if lap_time_data is not None and not lap_time_data.empty:
        # Use real lap time data
        drivers = top_drivers['Driver'].tolist()
        
        # Filter and plot lap times for each driver
        for driver in drivers:
            driver_data = lap_time_data[lap_time_data['Driver'] == driver]
            
            if not driver_data.empty:
                # Sort by lap number
                driver_data = driver_data.sort_values('LapNumber')
                
                # Get team color
                team = top_drivers[top_drivers['Driver'] == driver]['Team'].iloc[0]
                color = TEAM_COLORS.get(team, 'gray')
                
                # Plot lap time evolution
                plt.plot(driver_data['LapNumber'], driver_data['LapTime_sec'], 
                         label=driver, color=color)
                
                # Add pit stop markers if available
                if 'PitIn' in driver_data.columns:
                    pit_in_laps = driver_data[driver_data['PitIn'] == True]['LapNumber']
                    for lap in pit_in_laps:
                        plt.axvline(x=lap, color=color, linestyle=':', alpha=0.5)
    else:
        # Create synthetic lap time data
        drivers = top_drivers['Driver'].tolist()
        teams = top_drivers['Team'].tolist()
        colors = [TEAM_COLORS.get(team, 'gray') for team in teams]
        
        # Create lap numbers for a typical race
        lap_numbers = np.arange(1, 61)
        
        # Base lap time (around 90 seconds)
        base_lap_time = 90.0
        
        # Create synthetic lap time evolution for each driver
        for i, (driver, color) in enumerate(zip(drivers, colors)):
            # Driver-specific base pace (position-based)
            driver_pace = base_lap_time * (1 - 0.005 * (len(drivers) - i))
            
            # Fuel effect: cars get faster as fuel burns off (about 0.1s per lap)
            fuel_effect = -0.1 * lap_numbers
            
            # Tire degradation: different for each stint
            # Define stint lengths (generate different strategies)
            if i % 3 == 0:  # 2-stop strategy
                stint1_end = 18
                stint2_end = 38
            elif i % 3 == 1:  # 1-stop conservative
                stint1_end = 30
                stint2_end = 60
            else:  # 1-stop aggressive
                stint1_end = 24
                stint2_end = 60
            
            tire_deg = np.zeros_like(lap_numbers, dtype=float)
            
            # Stint 1: Fresh tires to first pit stop
            stint1_mask = lap_numbers <= stint1_end
            stint1_age = lap_numbers[stint1_mask] - 1
            tire_deg[stint1_mask] = 0.02 * stint1_age**1.5
            
            # Stint 2: First pit stop to second pit stop
            stint2_mask = (lap_numbers > stint1_end) & (lap_numbers <= stint2_end)
            stint2_age = lap_numbers[stint2_mask] - stint1_end - 1
            tire_deg[stint2_mask] = 0.02 * stint2_age**1.5
            
            # Stint 3: Second pit stop to end (if applicable)
            if stint2_end < 60:
                stint3_mask = lap_numbers > stint2_end
                stint3_age = lap_numbers[stint3_mask] - stint2_end - 1
                tire_deg[stint3_mask] = 0.02 * stint3_age**1.5
            
            # Create race pace model
            race_pace = driver_pace + fuel_effect + tire_deg
            
            # Add randomness and anomalies
            noise = np.random.normal(0, 0.2, size=race_pace.shape)
            # Add occasional outlier laps (traffic, small mistakes)
            outliers = np.random.choice([0, 1], size=race_pace.shape, p=[0.95, 0.05])
            outlier_effect = outliers * np.random.uniform(0.5, 1.5, size=race_pace.shape)
            
            race_pace = race_pace + noise + outlier_effect
            
            # Add push laps at the beginning of a stint
            push_laps = [1, stint1_end + 1]
            if stint2_end < 60:
                push_laps.append(stint2_end + 1)
            
            for lap in push_laps:
                if lap <= len(race_pace):
                    lap_idx = lap - 1
                    race_pace[lap_idx] = race_pace[lap_idx] - 0.5  # Push lap is faster
            
            # Plot the data
            plt.plot(lap_numbers, race_pace, label=driver, color=color)
            
            # Add pit stop markers
            if stint1_end < 60:
                plt.axvline(x=stint1_end, color=color, linestyle=':', alpha=0.5)
            if stint2_end < 60:
                plt.axvline(x=stint2_end, color=color, linestyle=':', alpha=0.5)
    
    # Customize plot
    plt.title('Race Pace Evolution', fontsize=16)
    plt.xlabel('Lap Number', fontsize=12)
    plt.ylabel('Lap Time (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add race phases
    phases = [
        {'name': 'Phase 1: Start', 'start': 1, 'end': 5, 'description': 'Initial positioning'},
        {'name': 'Phase 2: Early Race', 'start': 6, 'end': 20, 'description': 'Settling into pace'},
        {'name': 'Phase 3: Mid Race', 'start': 21, 'end': 40, 'description': 'Strategy execution'},
        {'name': 'Phase 4: End Game', 'start': 41, 'end': 60, 'description': 'Push to finish'}
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    
    for phase, color in zip(phases, colors):
        plt.axvspan(phase['start'], phase['end'], alpha=0.2, color=color)
        plt.text((phase['start'] + phase['end'])/2, plt.ylim()[0] + 0.4, 
                 phase['name'], ha='center', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.8))
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, 'race_pace_evolution.png'), bbox_inches='tight')
    plt.close()

def create_driver_comparison_dashboard(predictions, quali_data=None, race_data=None):
    """
    Create a comprehensive dashboard comparing two drivers with multiple metrics.
    
    Args:
        predictions (DataFrame): Predicted race results
        quali_data (DataFrame, optional): Qualifying data
        race_data (DataFrame, optional): Race data
        
    Returns:
        None: Saves visualization to file
    """
    # Get visualization folder
    viz_folder = ensure_visualization_folder()
    
    # Select top two drivers to compare
    driver1 = predictions.iloc[0]['Driver']
    driver2 = predictions.iloc[1]['Driver']
    
    team1 = predictions.iloc[0]['Team']
    team2 = predictions.iloc[1]['Team']
    
    color1 = TEAM_COLORS.get(team1, 'blue')
    color2 = TEAM_COLORS.get(team2, 'red')
    
    is_rookie1 = driver1 in ROOKIES
    is_rookie2 = driver2 in ROOKIES
    
    # Set up a complex figure
    plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=plt.gcf())
    
    # 1. Header with driver info
    ax_header = plt.subplot(gs[0, :])
    ax_header.axis('off')
    
    # Create header text with driver comparison
    driver1_text = f"{driver1}\n{team1}" + (" (ROOKIE)" if is_rookie1 else "")
    driver2_text = f"{driver2}\n{team2}" + (" (ROOKIE)" if is_rookie2 else "")
    
    header = f"Driver Comparison Dashboard\n{driver1_text} vs {driver2_text}"
    ax_header.text(0.5, 0.5, header, fontsize=18, ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.5", fc='white', ec="gray"))
    
    # Create rest of the sections (qualifying performance, race simulation, etc.)
    # These would be filled with real data when available
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, 'driver_comparison_dashboard.png'), bbox_inches='tight')
    plt.close()
    
def plot_speed_comparison(predictions, telemetry_data=None):
    """
    Create speed comparison chart between top two drivers.
    
    Args:
        predictions (DataFrame): Predicted race results
        telemetry_data (DataFrame, optional): Speed telemetry data
        
    Returns:
        None: Saves visualization to file
    """
    # Get visualization folder
    viz_folder = ensure_visualization_folder()
    
    # Get top two drivers
    top_drivers = predictions.head(2)
    
    plt.figure(figsize=(14, 8))
    
    if telemetry_data is not None and not telemetry_data.empty:
        # Filter telemetry for top two drivers
        driver1 = top_drivers.iloc[0]['Driver']
        driver2 = top_drivers.iloc[1]['Driver']
        
        driver1_data = telemetry_data[telemetry_data['Driver'] == driver1]
        driver2_data = telemetry_data[telemetry_data['Driver'] == driver2]
        
        # Plot speed over distance
        team1 = top_drivers.iloc[0]['Team']
        team2 = top_drivers.iloc[1]['Team']
        
        color1 = TEAM_COLORS.get(team1, 'blue')
        color2 = TEAM_COLORS.get(team2, 'red')
        
        plt.plot(driver1_data['Distance'], driver1_data['Speed'], label=driver1, color=color1)
        plt.plot(driver2_data['Distance'], driver2_data['Speed'], label=driver2, color=color2)
        
        # Highlight speed differences
        plt.fill_between(driver1_data['Distance'], 
                         driver1_data['Speed'], driver2_data['Speed'],
                         where=(driver1_data['Speed'] > driver2_data['Speed']),
                         alpha=0.3, color=color1)
        
        plt.fill_between(driver1_data['Distance'], 
                         driver1_data['Speed'], driver2_data['Speed'],
                         where=(driver1_data['Speed'] < driver2_data['Speed']),
                         alpha=0.3, color=color2)
    else:
        # Create synthetic speed data for comparison
        driver1 = top_drivers.iloc[0]['Driver']
        driver2 = top_drivers.iloc[1]['Driver']
        
        team1 = top_drivers.iloc[0]['Team']
        team2 = top_drivers.iloc[1]['Team']
        
        color1 = TEAM_COLORS.get(team1, 'blue')
        color2 = TEAM_COLORS.get(team2, 'red')
        
        # Create a simulated lap with distance points (5.5 km circuit)
        distance = np.linspace(0, 5500, 1000)
        
        # Define track sections
        straights = [(0, 500), (1200, 2000), (3000, 3800), (4500, 5000)]
        corners = [(500, 1200), (2000, 3000), (3800, 4500), (5000, 5500)]
        
        # Create base speed profile
        speed_profile = np.zeros_like(distance)
        
        # Set speeds for different sections
        for start, end in straights:
            mask = (distance >= start) & (distance <= end)
            # Accelerate at the start, constant speed in the middle, decelerate at the end
            section_distance = distance[mask]
            # Normalize to 0-1 range within this section
            norm_distance = (section_distance - start) / (end - start)
            
            # Create speed profile for straight (accelerate, maintain, decelerate)
            straight_speed = 300 + norm_distance * 20  # Accelerate to top speed
            # Apply plateau in the middle
            straight_speed[norm_distance > 0.3] = 320
            # Apply deceleration near the end
            straight_speed[norm_distance > 0.8] = 320 - (norm_distance[norm_distance > 0.8] - 0.8) * 100
            
            speed_profile[mask] = straight_speed
        
        for start, end in corners:
            mask = (distance >= start) & (distance <= end)
            section_distance = distance[mask]
            # Normalize to 0-1 range within this section
            norm_distance = (section_distance - start) / (end - start)
            
            # Create a "V" shaped speed profile for the corner
            # Starting from braking point, reaching minimum at corner apex, then accelerating out
            if end - start < 500:  # Sharper corner
                min_corner_speed = 80  # Slower for sharper corners
            else:
                min_corner_speed = 120  # Faster for wider corners
                
            # Speed at corner entry (comes from previous section)
            entry_speed = speed_profile[mask][0] if len(speed_profile[mask]) > 0 else 150
            
            # Corner braking and acceleration profile
            corner_speed = entry_speed - norm_distance * (entry_speed - min_corner_speed) * 2
            corner_speed[norm_distance > 0.5] = min_corner_speed + (norm_distance[norm_distance > 0.5] - 0.5) * (320 - min_corner_speed) * 2
            
            speed_profile[mask] = corner_speed
            
        # Create driver-specific speed profiles with small differences
        # Driver 1 - better in corners
        speed1 = speed_profile.copy()
        for start, end in corners:
            mask = (distance >= start) & (distance <= end)
            speed1[mask] = speed1[mask] * 1.05  # 5% faster in corners
            
        # Driver 2 - better on straights
        speed2 = speed_profile.copy()
        for start, end in straights:
            mask = (distance >= start) & (distance <= end)
            speed2[mask] = speed2[mask] * 1.03  # 3% faster on straights
            
        # Add some noise
        speed1 += np.random.normal(0, 2, size=speed1.shape)
        speed2 += np.random.normal(0, 2, size=speed2.shape)
        
        # Plot
        plt.plot(distance, speed1, label=driver1, color=color1)
        plt.plot(distance, speed2, label=driver2, color=color2)
        
        # Highlight speed differences
        plt.fill_between(distance, speed1, speed2, 
                         where=(speed1 > speed2), 
                         alpha=0.3, color=color1)
        
        plt.fill_between(distance, speed1, speed2, 
                         where=(speed1 < speed2), 
                         alpha=0.3, color=color2)
        
        # Add corner labels
        for i, (start, end) in enumerate(corners):
            mid_point = (start + end) / 2
            plt.text(mid_point, 50, f"C{i+1}", fontsize=12, ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.8))
    
    # Customize plot
    plt.title(f'Speed Comparison: {driver1} vs {driver2}', fontsize=16)
    plt.xlabel('Distance (m)', fontsize=12)
    plt.ylabel('Speed (km/h)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, 'speed_comparison.png'))
    plt.close()

