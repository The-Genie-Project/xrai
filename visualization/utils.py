"""
Utility functions for visualization.
"""
import matplotlib.pyplot as plt
import numpy as np

def cleanup_visualization():
    """Close all matplotlib figures."""
    plt.close('all') 

def downsample_fitness_history(fitness_history, generations):
    # Downsample long histories for plotting
    if len(fitness_history) > 10000:
        # Only plot every Nth point
        plot_indices = np.linspace(0, len(fitness_history)-1, 5000, dtype=int)
        plot_generations = generations[plot_indices]
        plot_fitness = np.array(fitness_history)[plot_indices]
    else:
        plot_generations = generations
        plot_fitness = fitness_history
    return plot_generations, plot_fitness 
