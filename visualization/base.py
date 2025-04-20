# visualization/base.py (fixed)
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

# Set up logger
logger = logging.getLogger('f1_prediction.visualization')

def ensure_visualization_folder(folder_path=None):
    """Create visualization folder if it doesn't exist."""
    if folder_path is None:
        folder_path = os.path.join(os.getcwd(), 'visualizations')
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def get_team_color(team):
    """Get team color from constants or return default color."""
    # Default team colors
    team_colors = {
        'Red Bull Racing': '#0600EF',
        'Ferrari': '#DC0000',
        'Mercedes': '#00D2BE',
        'McLaren': '#FF8700',
        'Aston Martin': '#006F62',
        'Alpine': '#0090FF',
        'Williams': '#005AFF',
        'Racing Bulls': '#1E41FF',
        'Kick Sauber': '#900000',
        'Haas F1 Team': '#FFFFFF'
    }
    return team_colors.get(team, 'skyblue')

def adjust_color(color, amount=0.5):
    """Adjust the brightness of a color."""
    try:
        # Convert hex to RGB
        color = color.lstrip('#')
        lv = len(color)
        r, g, b = tuple(int(color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        
        # Adjust brightness
        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'
    except:
        return '#FFFFFF'  # Return white if something goes wrong

def save_figure(fig, filename, folder_path=None, dpi=300):
    """Save figure to file with error handling."""
    folder_path = ensure_visualization_folder(folder_path)
    
    # Ensure filename has extension
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    filepath = os.path.join(folder_path, filename)
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving figure to {filepath}: {e}")
        return None

def plot_grid_vs_finish(predictions, title=None, figsize=(10, 6), save_dir=None, filename=None, save_path=None):
    """
    Create a visualization comparing grid positions to finish positions.
    
    Args:
        predictions (DataFrame): Race predictions with Grid and Position columns
        title (str, optional): Plot title
        figsize (tuple): Figure size
        save_dir (str, optional): Directory to save the plot
        filename (str, optional): Filename to save the plot
        save_path (str, optional): Full path to save the plot (alternative to save_dir/filename)
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        grid_positions = predictions['Grid'].values if 'Grid' in predictions.columns else predictions['GridPosition'].values
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
            # Direct save to full path
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving visualization to {save_path}: {e}")
        elif save_dir and filename:
            # Save to directory with filename
            try:
                os.makedirs(save_dir, exist_ok=True)
                full_path = os.path.join(save_dir, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {full_path}")
            except Exception as e:
                logger.error(f"Error saving visualization to {save_dir}/{filename}: {e}")
        
        return fig
    except Exception as e:
        logger.error(f"Error in plot_grid_vs_finish: {e}")
        # Create a minimal fallback plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error creating grid vs finish plot: {e}", 
                ha='center', va='center', fontsize=12)
        plt.tight_layout()
        
        # Try to save even the error plot
        if save_path:
            try:
                plt.savefig(save_path)
            except:
                pass
        elif save_dir and filename:
            try:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, filename))
            except:
                pass
        
        return fig

def plot_race_results(results_df, title=None, figsize=(10, 8)):
    """
    Plot race results in a horizontal bar chart.
    
    Args:
        results_df (DataFrame): Race results with Position and Interval columns
        title (str, optional): Plot title
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by position if not already sorted
        if 'Position' in results_df.columns:
            sorted_results = results_df.sort_values('Position').copy()
        else:
            sorted_results = results_df.copy()
        
        # Extract data
        drivers = sorted_results['Driver'].values
        
        # Create interval data for plotting
        intervals = []
        for _, row in sorted_results.iterrows():
            if 'IntervalSeconds' in row and pd.notna(row['IntervalSeconds']):
                intervals.append(row['IntervalSeconds'])
            elif 'Interval' in row and row['Interval'] != "WINNER":
                try:
                    # Try to extract the interval value from string format (e.g., "+12.345s")
                    interval_str = str(row['Interval']).replace('+', '').replace('s', '')
                    intervals.append(float(interval_str))
                except:
                    intervals.append(0)
            else:
                intervals.append(0)
        
        # Create horizontal bars for intervals
        bars = ax.barh(drivers, intervals, height=0.7)
        
        # Add team colors if available
        if 'Team' in sorted_results.columns:
            teams = sorted_results['Team'].values
            for i, team in enumerate(teams):
                color = get_team_color(team)
                bars[i].set_color(color)
        
        # Add interval values as text
        for i, interval in enumerate(intervals):
            if interval > 0:
                ax.text(interval + 0.1, i, f"+{interval:.3f}s", va='center')
            else:
                ax.text(0.1, i, "WINNER", va='center', fontweight='bold')
        
        # Set plot limits and labels
        ax.set_xlabel('Interval (seconds)')
        ax.set_title(title or 'Race Results')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        return fig
    except Exception as e:
        logger.error(f"Error in plot_race_results: {e}")
        # Create a minimal fallback plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error creating race results plot: {e}", 
                ha='center', va='center', fontsize=12)
        plt.tight_layout()
        return fig