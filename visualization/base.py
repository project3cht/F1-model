# Create a new file: visualization/base.py

import os
import matplotlib.pyplot as plt
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