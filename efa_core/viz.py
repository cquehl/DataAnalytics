"""
Visualization Utilities Module
==============================

Minimal style setup and helper functions.
Each test script handles its own plotting for maximum flexibility.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def setup_style() -> None:
    """
    Configure matplotlib and seaborn style settings.

    Sets:
    - Agg backend for headless environments
    - Seaborn whitegrid style
    - Consistent font sizes
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
    })


def get_colors() -> dict:
    """
    Return consistent color palette for plots.

    Returns:
        Dictionary with named colors
    """
    return {
        'primary': '#3498db',      # Blue
        'secondary': '#2ecc71',    # Green
        'accent': '#e74c3c',       # Red
        'neutral': '#95a5a6',      # Gray
        'highlight': '#f39c12',    # Orange
        'significant': '#2ecc71',  # Green (for p < 0.05)
        'not_significant': '#95a5a6',  # Gray
    }


def save_figure(fig: plt.Figure, path: str, dpi: int = None) -> None:
    """
    Save figure with consistent settings.

    Parameters:
        fig: Matplotlib figure
        path: Output file path
        dpi: Resolution. Defaults to config.DEFAULT_DPI
    """
    if dpi is None:
        dpi = config.DEFAULT_DPI

    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")


def get_cmap(style: str = 'diverging') -> str:
    """
    Return appropriate colormap name.

    Parameters:
        style: 'diverging' for correlation/loadings, 'sequential' for counts

    Returns:
        Colormap name string
    """
    if style == 'diverging':
        return 'RdBu_r'
    elif style == 'sequential':
        return 'Blues'
    else:
        return 'viridis'


def create_figure(nrows: int = 1, ncols: int = 1, figsize: tuple = None) -> tuple:
    """
    Create figure with subplots using consistent settings.

    Parameters:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Optional figure size (width, height)

    Returns:
        Tuple of (fig, axes)
    """
    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.set_facecolor('white')
    return fig, axes
