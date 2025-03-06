"""
Visualization and logging manager for the XRAI package.

This module handles all visualization, printing, and logging functionality
for the live evolution process.
"""

from .setup import setup_visualization
from .plotting import update_plots, create_figure_if_closed
from .saving import save_results, save_checkpoint
from .console import print_start_info, print_generation_stats, print_checkpoint_info
from .utils import cleanup_visualization

# Export all the main functions
__all__ = [
    'setup_visualization',
    'update_plots',
    'create_figure_if_closed',
    'save_results',
    'save_checkpoint',
    'print_start_info',
    'print_generation_stats',
    'print_checkpoint_info',
    'cleanup_visualization'
] 
