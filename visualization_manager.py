#!/usr/bin/env python3
"""
Visualization and logging manager for the XRAI package.

This module handles all visualization, printing, and logging functionality
for the live evolution process.

Note: This file now serves as a compatibility layer for the refactored visualization package.
"""

# Import all functions from the refactored modules
from visualization import (
    setup_visualization,
    update_plots,
    create_figure_if_closed,
    save_results,
    save_checkpoint,
    print_start_info,
    print_generation_stats,
    print_checkpoint_info,
    cleanup_visualization
)

# Re-export all functions to maintain backward compatibility
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
