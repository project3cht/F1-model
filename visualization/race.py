# Create a new file: visualization/race.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from visualization.base import ensure_visualization_folder, save_figure, get_team_color

# Set up logger
logger = logging.getLogger('f1_prediction.visualization.race')

def plot_grid_vs_finish(predictions, title=None, figsize=(10, 6), save_path=None):
    """
    Create a visualization comparing grid positions to finish positions.
    
    Args:
        predictions (DataFrame): Race predictions with Grid and Position columns
        title (str, optional): Plot title
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the visualization
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    grid_positions = predictions['Grid'].values
    finish_positions = predictions['Position'].values
    drivers = predictions['Driver'].values
    
    # Plot grid vs finish positions
    scatter = ax.scatter(grid_positions, finish_positions, s=100, alpha=0.7)
    
    # Add team colors if teams column exists
    if 'Team' in predictions.columns:
        teams = predictions['Team'].values
        for i, team in enumerate(teams):
            color = get_team_color(team)
            scatter.get_paths()[i].set_facecolor(color)
    
    # Add diagonal line (grid = finish)
    max_pos = max(grid_positions.max(), finish_positions.max())
    ax.plot([1, max_pos], [1, max_pos], 'k--', alpha=0.3)
    
    # Add driver labels
    for i, driver in enumerate(drivers):
        ax.annotate(driver, (grid_positions[i], finish_positions[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Set plot limits and labels
    ax.set_xlim(0.5, max_pos + 0.5)
    ax.set_ylim(0.5, max_pos + 0.5)
    ax.set_xlabel('Grid Position')
    ax.set_ylabel('Finish Position')
    ax.invert_yaxis()  # Invert y-axis so that 1st is at the top
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Save plot if requested
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving visualization to {save_path}: {e}")
    
    return fig

def plot_position_changes(predictions, folder_path=None):
    """
    Create a visualization that simulates position changes during a race.
    
    Args:
        predictions (DataFrame): Predicted race results
        folder_path (str, optional): Folder to save the visualization in
        
    Returns:
        str: Path to the saved visualization
    """
    # Create visualization folder
    folder_path = ensure_visualization_folder(folder_path)
    
    # Extract data dimensions
    num_drivers = len(predictions)
    num_laps = 50  # Simulate a 50-lap race
    
    # Extract grid and predicted positions
    grid_positions = predictions['Grid'].values
    final_positions = predictions['Position'].values
    drivers = predictions['Driver'].values
    teams = predictions['Team'].values
    
    # Create a matrix of positions for each lap
    position_matrix = np.zeros((num_drivers, num_laps))
    
    # Fill the position matrix with simulated lap positions
    for i in range(num_drivers):
        start_pos = grid_positions[i]
        end_pos = final_positions[i]
        
        # Determine when position changes occur
        # More changes tend to happen at the start and fewer toward the end
        change_points = sorted(np.random.choice(
            range(1, num_laps), 
            size=min(10, abs(int(end_pos - start_pos))), 
            replace=False
        ))
        
        # Calculate intermediate positions
        positions = [start_pos]
        current_pos = start_pos
        
        for lap in range(1, num_laps):
            if lap in change_points:
                # Move toward the final position
                if current_pos < end_pos:
                    current_pos += 1
                elif current_pos > end_pos:
                    current_pos -= 1
            positions.append(current_pos)
        
        position_matrix[i] = positions
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set of rookies - provide an empty set as default since we don't have this information
    rookies = set()
    
    for i in range(num_drivers):
        # Get team color
        color = get_team_color(teams[i])
        
        # Add line style for rookies
        linestyle = '--' if drivers[i] in rookies else '-'
        linewidth = 2 if drivers[i] in rookies else 1.5
        
        # Plot position changes
        ax.plot(range(1, num_laps + 1), position_matrix[i], 
                color=color, linestyle=linestyle, linewidth=linewidth,
                label=drivers[i])
    
    # Customize the plot
    ax.invert_yaxis()  # Invert y-axis so position 1 is at the top
    ax.set_title('Simulated Race Position Changes', fontsize=14)
    ax.set_xlabel('Lap Number', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(num_drivers + 0.5, 0.5)  # Set y-axis limits
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure
    filepath = save_figure(fig, 'position_changes', folder_path)
    
    return filepath

def plot_top10_intervals(predictions, folder_path=None):
    """
    Create a horizontal bar chart showing top 10 finishers with time intervals.
    
    Args:
        predictions (DataFrame): Predicted race results
        folder_path (str, optional): Folder to save the visualization in
        
    Returns:
        str: Path to the saved visualization
    """
    # Filter to top 10 finishers
    top_10 = predictions.head(10).copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # If 'Interval (s)' not in columns, try to extract it
    if 'Interval (s)' not in top_10.columns:
        # Try to convert from string format if available
        if 'Interval' in top_10.columns:
            top_10['Interval (s)'] = top_10['Interval'].apply(
                lambda x: 0 if x == 'WINNER' else float(x.replace('+', '').replace('s', ''))
            )
        else:
            # Create synthetic intervals
            top_10['Interval (s)'] = [0 if i == 0 else i * 1.5 for i in range(len(top_10))]
    
    # Create horizontal bars
    bars = ax.barh(top_10['Driver'], top_10['Interval (s)'], color='skyblue')
    
    # Color bars by team
    for i, bar in enumerate(bars):
        team = top_10.iloc[i]['Team']
        color = get_team_color(team)
        bar.set_color(color)
    
    # Customize plot
    ax.set_xlabel('Time Gap to Winner (seconds)', fontsize=12)
    ax.set_title('Predicted Top 10 Finishers with Time Intervals', fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add interval values as text
    for i, row in enumerate(top_10.iterrows()):
        _, data = row
        if data.get('Interval') == "WINNER" or data['Interval (s)'] == 0:
            ax.text(0.1, i, "WINNER", va='center', weight='bold')
        else:
            ax.text(data['Interval (s)'] + 0.1, i, f"+{data['Interval (s)']:.3f}s", va='center')
    
    # Save figure
    filepath = save_figure(fig, 'top10_intervals', folder_path)
    
    return filepath