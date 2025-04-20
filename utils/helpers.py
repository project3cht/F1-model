# utils/helpers.py
"""
Helper functions for F1 race predictions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.constants import TEAM_COLORS, DRIVERS

def ensure_directory(directory_path):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to directory
        
    Returns:
        str: Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    return directory_path

def get_team_color(team, default='gray'):
    """
    Get color for a team.
    
    Args:
        team (str): Team name
        default (str): Default color
        
    Returns:
        str: Hex color code
    """
    return TEAM_COLORS.get(team, default)

def get_team_for_driver(driver):
    """
    Get the team for a driver.
    
    Args:
        driver (str): Driver name
        
    Returns:
        str: Team name or None if driver not found
    """
    return DRIVERS.get(driver)

def save_figure(fig, filename, directory='results'):
    """
    Save a matplotlib figure to disk.
    
    Args:
        fig (Figure): Matplotlib figure
        filename (str): Filename
        directory (str): Directory to save to
        
    Returns:
        str: Path to saved figure
    """
    # Ensure directory exists
    ensure_directory(directory)
    
    # Add timestamp to filename to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = f"{timestamp}_{filename}"
    
    # Create full path
    filepath = os.path.join(directory, filename_with_timestamp)
    
    # Save figure
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Figure saved to {filepath}")
    
    return filepath