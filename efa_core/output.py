"""
Output Naming and Saving Module
===============================

Functions for creating dated output directories and saving results
with consistent naming conventions.

Naming Pattern: {DATE}-{TEST}-{SUFFIX}.{EXT}
Example: 2024-02-09-efa-loadings.csv
"""

import os
from datetime import date
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_output_dir(test_name: str, base: str = None) -> Path:
    """
    Create and return dated output directory.

    Creates directory: {base}/{DATE}-{test_name}/
    Example: outputs/2024-02-09-efa/

    Parameters:
        test_name: Name of the test/analysis (lowercase-hyphen)
        base: Base output directory. Defaults to config.DEFAULT_OUTPUT_BASE

    Returns:
        Path to created output directory
    """
    if base is None:
        base = config.DEFAULT_OUTPUT_BASE

    today = date.today().isoformat()
    output_dir = Path(base) / f"{today}-{test_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    return output_dir


def _build_filename(output_dir: Path, test_name: str, suffix: str, ext: str) -> Path:
    """Build dated filename with pattern: {DATE}-{TEST}-{SUFFIX}.{EXT}"""
    today = date.today().isoformat()
    filename = f"{today}-{test_name}-{suffix}.{ext}"
    return output_dir / filename


def save_csv(
    df: pd.DataFrame,
    output_dir: Path,
    test_name: str,
    suffix: str,
    index: bool = False
) -> Path:
    """
    Save DataFrame to CSV with dated filename.

    Parameters:
        df: DataFrame to save
        output_dir: Output directory path
        test_name: Test name for filename
        suffix: Descriptive suffix (e.g., 'results', 'loadings')
        index: Whether to include index in output

    Returns:
        Path to saved file
    """
    filepath = _build_filename(output_dir, test_name, suffix, 'csv')
    df.to_csv(filepath, index=index)
    print(f"Saved: {filepath}")
    return filepath


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    test_name: str,
    suffix: str,
    dpi: int = None
) -> Path:
    """
    Save matplotlib figure with dated filename.

    Parameters:
        fig: Matplotlib figure to save
        output_dir: Output directory path
        test_name: Test name for filename
        suffix: Descriptive suffix (e.g., 'scree', 'loadings')
        dpi: Resolution. Defaults to config.DEFAULT_DPI

    Returns:
        Path to saved file
    """
    if dpi is None:
        dpi = config.DEFAULT_DPI

    filepath = _build_filename(output_dir, test_name, suffix, 'png')
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def save_report(
    text: str,
    output_dir: Path,
    test_name: str,
    suffix: str = 'report'
) -> Path:
    """
    Save text report with dated filename.

    Parameters:
        text: Text content to save
        output_dir: Output directory path
        test_name: Test name for filename
        suffix: Descriptive suffix

    Returns:
        Path to saved file
    """
    filepath = _build_filename(output_dir, test_name, suffix, 'txt')
    with open(filepath, 'w') as f:
        f.write(text)
    print(f"Saved: {filepath}")
    return filepath


def list_outputs(output_dir: Path) -> list[str]:
    """List all files in the output directory."""
    if output_dir.exists():
        files = sorted(os.listdir(output_dir))
        return files
    return []


def print_summary(output_dir: Path) -> None:
    """Print summary of all output files."""
    files = list_outputs(output_dir)
    if files:
        print(f"\nFiles generated in {output_dir}:")
        for f in files:
            print(f"  - {f}")
    else:
        print(f"\nNo files generated in {output_dir}")
