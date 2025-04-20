"""
Helper functions and constants for F1 race predictions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Drivers and their teams
DRIVERS = {
    'Max Verstappen': 'Red Bull Racing', 
    'Yuki Tsunoda': 'Red Bull Racing', 
    'Charles Leclerc': 'Ferrari', 
    'Lewis Hamilton': 'Ferrari',
    'George Russell': 'Mercedes', 
    'Kimi Antonelli': 'Mercedes', 
    'Lando Norris': 'McLaren', 
    'Oscar Piastri': 'McLaren',
    'Fernando Alonso': 'Aston Martin', 
    'Lance Stroll': 'Aston Martin', 
    'Liam Lawson': 'Racing Bulls', 
    'Isack Hadjar': 'Racing Bulls',
    'Alexander Albon': 'Williams', 
    'Carlos Sainz': 'Williams', 
    'Gabriel Bortoleto': 'Kick Sauber', 
    'Nico Hulkenberg': 'Kick Sauber',
    'Oliver Bearman': 'Haas F1 Team', 
    'Esteban Ocon': 'Haas F1 Team', 
    'Jack Doohan': 'Alpine', 
    'Pierre Gasly': 'Alpine'
}

# Team colors for visualization
TEAM_COLORS = {
    'Red Bull Racing': '#0600EF',
    'Ferrari': '#DC0000',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'Racing Bulls': '#1E41FF',
    'Haas F1 Team': '#FFFFFF',
    'Kick Sauber': '#900000'
}

# Rookie drivers for 2025
ROOKIES = [
    'Kimi Antonelli', 'Oliver Bearman', 'Gabriel Bortoleto', 
    'Jack Doohan', 'Isack Hadjar', 'Liam Lawson'
]

# Performance factors for teams and drivers (lower = better)
TEAM_PERFORMANCE = {
    'Red Bull Racing': 0.98,
    'Ferrari': 0.99,
    'McLaren': 0.98,
    'Mercedes': 0.99,
    'Aston Martin': 1.01,
    'Racing Bulls': 1.01,
    'Williams': 1.03,
    'Haas F1 Team': 1.02,
    'Kick Sauber': 1.03,
    'Alpine': 1.03
}

DRIVER_PERFORMANCE = {
    'Max Verstappen': 0.97,
    'Charles Leclerc': 0.98,
    'Lando Norris': 0.98,
    'Lewis Hamilton': 0.99,
    'Carlos Sainz': 0.99,
    'George Russell': 0.99,
    'Oscar Piastri': 0.99,
    'Fernando Alonso': 1.00,
    'Kimi Antonelli': 1.00,
    'Liam Lawson': 1.00,
    'Alexander Albon': 1.01,
    'Yuki Tsunoda': 1.01,
    'Nico Hulkenberg': 1.01,
    'Oliver Bearman': 1.01,
    'Gabriel Bortoleto': 1.01,
    'Jack Doohan': 1.01,
    'Isack Hadjar': 1.01,
    'Lance Stroll': 1.02,
    'Esteban Ocon': 1.02,
    'Pierre Gasly': 1.02
}

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