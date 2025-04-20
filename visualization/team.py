# Create a new file: visualization/team.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualization.base import ensure_visualization_folder, get_team_color, save_figure

def plot_team_performance(predictions, save_path=None):
    """
    Create a visualization of team performance.
    
    Args:
        predictions (DataFrame): Predicted race results
        save_path (str, optional): Path to save the visualization
        
    Returns:
        str: Path to the saved visualization
    """
    # Group data by team
    team_performance = predictions.groupby('Team')['Position'].agg(['min', 'max', 'mean']).reset_index()
    
    # Sort by best position (min)
    team_performance = team_performance.sort_values('min')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot range for each team
    for i, row in team_performance.iterrows():
        team = row['Team']
        pos_min = row['min']
        pos_max = row['max']
        pos_avg = row['mean']
        
        # Get team color
        color = get_team_color(team)
        
        # Plot range
        ax.plot([team, team], [pos_min, pos_max], 'o-', color=color, markersize=10, label=team)
        
        # Add average marker
        ax.plot(team, pos_avg, 'D', color='black', markersize=6)
        
        # Add text annotations
        ax.text(i, pos_min - 0.4, f"P{int(pos_min)}", ha='center')
        ax.text(i, pos_max + 0.4, f"P{int(pos_max)}", ha='center')
    
    # Create custom legend for average marker
    ax.plot([], [], 'D', color='black', markersize=6, label='Team Average')
    
    # Customize plot
    ax.set_title('Predicted Team Performance', fontsize=16)
    ax.set_xlabel('Team', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_yticks(range(1, 21))
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(team_performance)))
    ax.set_xticklabels(team_performance['Team'], rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.invert_yaxis()  # Invert y-axis so position 1 is at the top
    
    plt.tight_layout()
    
    # Save visualization
    if save_path:
        plt.savefig(save_path)
        saved_path = save_path
    else:
        # Use default path
        output_dir = ensure_visualization_folder()
        saved_path = os.path.join(output_dir, 'team_performance.png')
        plt.savefig(saved_path)
    
    plt.close(fig)
    
    return saved_path

def plot_team_pace_comparison(predictions, folder_path=None):
    """
    Create a visualization to compare team pace rankings.
    Inspired by FastF1's team_pace_ranking plot.
    
    Args:
        predictions (DataFrame): Predicted race results
        folder_path (str, optional): Folder to save the visualization in
        
    Returns:
        str: Path to the saved visualization
    """
    # Create visualization folder
    folder_path = ensure_visualization_folder(folder_path)
    
    # Group by team and calculate average interval
    team_data = []
    
    for team in predictions['Team'].unique():
        team_rows = predictions[predictions['Team'] == team]
        
        # Calculate pace metrics (excluding the winner's 0 interval)
        intervals = team_rows['Interval (s)'].values if 'Interval (s)' in team_rows.columns else []
        
        # If no 'Interval (s)', try to convert from 'Interval'
        if len(intervals) == 0 and 'Interval' in team_rows.columns:
            intervals = [0 if x == 'WINNER' else float(x.replace('+', '').replace('s', '')) for x in team_rows['Interval']]
        
        intervals = [i for i in intervals if i > 0]
        
        if len(intervals) > 0:
            avg_interval = np.mean(intervals)
            median_interval = np.median(intervals)
            min_interval = np.min(intervals) if len(intervals) > 0 else 0
            max_interval = np.max(intervals) if len(intervals) > 0 else 0
        else:
            # If this team has the winner, use a small value
            avg_interval = 0.1
            median_interval = 0.1
            min_interval = 0
            max_interval = 0.2
        
        # Get driver names
        drivers = ', '.join(team_rows['Driver'].values)
        
        team_data.append({
            'Team': team,
            'AvgInterval': avg_interval,
            'MedianInterval': median_interval,
            'MinInterval': min_interval,
            'MaxInterval': max_interval,
            'Drivers': drivers
        })
    
    # Convert to DataFrame and sort by average interval
    team_df = pd.DataFrame(team_data)
    team_df = team_df.sort_values('AvgInterval')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create custom "box plot" style bars
    for i, (_, row) in enumerate(team_df.iterrows()):
        team = row['Team']
        color = get_team_color(team)
        
        # Plot interval range
        ax.plot([row['MinInterval'], row['MaxInterval']], [i, i], 
                color=color, linewidth=2.5)
        
        # Plot median point
        ax.scatter(row['MedianInterval'], i, color=color, s=80, zorder=3)
        
        # Add team name and drivers
        ax.text(row['MaxInterval'] + 0.5, i, f"{team} ({row['Drivers']})", 
                va='center', ha='left')
    
    # Customize plot
    ax.set_title('Team Pace Comparison', fontsize=14)
    ax.set_xlabel('Interval to Winner (seconds)', fontsize=12)
    ax.set_yticks([])  # Hide y-axis ticks since we've added team names as text
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save figure
    filepath = save_figure(fig, 'team_pace_comparison', folder_path)
    
    return filepath