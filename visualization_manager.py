#!/usr/bin/env python3
"""
Visualization and logging manager for the XRAI package.

This module handles all visualization, printing, and logging functionality
for the live evolution process.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

def create_figure_if_closed(fig, figsize=(10, 5)):
    """Create a new figure if the previous one was closed."""
    if not plt.fignum_exists(fig.number):
        new_fig, new_ax = plt.subplots(figsize=figsize)
        return new_fig, new_ax
    return fig, fig.axes[0]

def setup_visualization():
    """Set up and return the visualization figures and axes."""
    # Enable interactive mode for matplotlib
    plt.ion()
    
    # Create figures for live updates
    fig1, ax1 = plt.subplots(figsize=(10, 5))  # Fitness history
    fig2, ax2 = plt.subplots(figsize=(10, 5))  # Prediction vs actual
    fig3, ax3 = plt.subplots(figsize=(10, 5))  # Weight evolution
    fig4, ax4 = plt.subplots(figsize=(10, 5))  # Fitness deviation
    fig5, ax5 = plt.subplots(figsize=(10, 5))  # Error comparison
    
    # Set window close events to handle gracefully
    for fig in [fig1, fig2, fig3, fig4, fig5]:
        fig.canvas.mpl_connect('close_event', lambda event: None)  # Do nothing when window is closed
    
    return (fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4), (fig5, ax5)

def print_checkpoint_info(checkpoint, start_generation, population_size, mutation_rate, r, global_best_predictor, global_best_fitness, global_best_generation):
    """Print information about a loaded checkpoint."""
    print(f"Successfully loaded checkpoint from generation {start_generation}")
    print(f"Continuing with parameters:")
    print(f"  - Population size: {population_size}")
    print(f"  - Mutation rate: {mutation_rate}")
    print(f"  - Chaos parameter (r): {r}")
    
    if global_best_predictor:
        print(f"Global best predictor (from generation {global_best_generation}):")
        print(f"  - Fitness: {global_best_fitness:.6f}")
        print(f"  - Weights: a={global_best_predictor.weights[0]:.4f}, b={global_best_predictor.weights[1]:.4f}, c={global_best_predictor.weights[2]:.4f}")

def print_start_info(population_size, mutation_rate, r, update_interval):
    """Print information about the starting parameters."""
    print(f"Starting live evolution with parameters:")
    print(f"  - Population size: {population_size}")
    print(f"  - Mutation rate: {mutation_rate}")
    print(f"  - Chaos parameter (r): {r}")
    print(f"  - Update interval: {update_interval}")
    print(f"Press Ctrl+C to stop the evolution")

def print_generation_stats(generation, predictor_fitness, meta_fitness, current_best_fitness, current_best_predictor, global_best_fitness, global_best_predictor, global_best_generation):
    """Print statistics for the current generation."""
    print(f"\nGeneration {generation}:")
    print(f"  - Current predictor fitness: {predictor_fitness:.4f}")
    print(f"  - Current meta-predictor fitness: {meta_fitness:.4f}")
    
    # Print information about current best and global best
    print(f"  - Current best predictor fitness: {current_best_fitness:.4f}")
    print(f"  - Current best weights: a={current_best_predictor.weights[0]:.4f}, b={current_best_predictor.weights[1]:.4f}, c={current_best_predictor.weights[2]:.4f}")
    print(f"  - Current equation: f(x) = {current_best_predictor.weights[0]:.4f}x² + {current_best_predictor.weights[1]:.4f}x + {current_best_predictor.weights[2]:.4f}")
    
    print(f"  - GLOBAL best predictor fitness: {global_best_fitness:.4f} (from generation {global_best_generation})")
    print(f"  - GLOBAL best weights: a={global_best_predictor.weights[0]:.4f}, b={global_best_predictor.weights[1]:.4f}, c={global_best_predictor.weights[2]:.4f}")
    print(f"  - GLOBAL best equation: f(x) = {global_best_predictor.weights[0]:.4f}x² + {global_best_predictor.weights[1]:.4f}x + {global_best_predictor.weights[2]:.4f}")

def update_plots(figures_axes, generation, fitness_history, weight_history, current_best_predictor, global_best_predictor, global_best_fitness, global_best_generation, r):
    """Update all visualization plots."""
    (fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4), (fig5, ax5) = figures_axes
    
    # Check if figures are still open and recreate them if needed
    fig1, ax1 = create_figure_if_closed(fig1)
    fig2, ax2 = create_figure_if_closed(fig2)
    fig3, ax3 = create_figure_if_closed(fig3)
    fig4, ax4 = create_figure_if_closed(fig4)
    fig5, ax5 = create_figure_if_closed(fig5)
    
    # Unpack fitness history
    predictor_fitness_values, meta_fitness_values = zip(*fitness_history) if fitness_history else ([], [])
    generations = np.arange(len(fitness_history))
    
    # Update fitness plot if figure is still open
    if plt.fignum_exists(fig1.number):
        ax1.clear()
        ax1.plot(generations, predictor_fitness_values, label="Predictor Fitness")
        ax1.plot(generations, meta_fitness_values, label="Meta-Predictor Fitness")
        ax1.axhline(y=global_best_fitness, color='r', linestyle='--', label=f"Global Best ({global_best_fitness:.4f})")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title(f"Evolution Progress (Generation {generation})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.canvas.draw_idle()
    
    # Update prediction vs actual plot if figure is still open
    if plt.fignum_exists(fig2.number):
        ax2.clear()
        x_values = np.linspace(0, 1, 100)
        
        # Import here to avoid circular imports
        from core import chaotic_function
        
        actual_values = np.array([chaotic_function(x, r) for x in x_values])
        current_predicted_values = np.array([current_best_predictor.predict(x) for x in x_values])
        global_predicted_values = np.array([global_best_predictor.predict(x) for x in x_values])
        
        # Plot the target function
        ax2.plot(x_values, actual_values, 'r-', label="Target (Chaotic Function)", alpha=0.7)
        
        # Plot the current best prediction
        ax2.plot(x_values, current_predicted_values, 'b-', label=f"Current Best (Gen {generation})", alpha=0.7)
        
        # Plot the global best prediction
        ax2.plot(x_values, global_predicted_values, 'g-', label=f"Global Best (Gen {global_best_generation})", alpha=0.7)
        
        ax2.set_xlabel("Input (x)")
        ax2.set_ylabel("Output")
        ax2.set_title(f"Predictor Performance: Current vs Global Best")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.canvas.draw_idle()
    
    # Update weights plot if figure is still open
    if plt.fignum_exists(fig3.number) and len(weight_history) > 0:
        ax3.clear()
        weight_history_array = np.array(weight_history)
        ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 0], label="Weight a (x²)")
        ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 1], label="Weight b (x)")
        ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 2], label="Weight c (constant)")
        
        # Add horizontal lines for global best weights
        ax3.axhline(y=global_best_predictor.weights[0], color='r', linestyle='--', alpha=0.5, 
                   label=f"Global Best a={global_best_predictor.weights[0]:.4f}")
        ax3.axhline(y=global_best_predictor.weights[1], color='g', linestyle='--', alpha=0.5,
                   label=f"Global Best b={global_best_predictor.weights[1]:.4f}")
        ax3.axhline(y=global_best_predictor.weights[2], color='b', linestyle='--', alpha=0.5,
                   label=f"Global Best c={global_best_predictor.weights[2]:.4f}")
        
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Weight Value")
        ax3.set_title(f"Evolution of Best Predictor Weights (Generation {generation})")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.canvas.draw_idle()
    
    # Update fitness deviation plot if figure is still open
    if plt.fignum_exists(fig4.number) and len(fitness_history) > 0:
        ax4.clear()
        # Calculate deviation between predictor and meta-predictor fitness
        fitness_deviation = np.array([abs(p - m) for p, m in fitness_history])
        
        # Plot the deviation
        ax4.plot(generations, fitness_deviation, 'g-', label="Fitness Deviation")
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Absolute Deviation")
        ax4.set_title(f"Predictor vs Meta-Predictor Fitness Deviation (Generation {generation})")
        
        # Add a moving average for trend visualization
        if len(fitness_deviation) > 10:
            window_size = min(50, len(fitness_deviation) // 5)
            moving_avg = np.convolve(fitness_deviation, np.ones(window_size)/window_size, mode='valid')
            ax4.plot(generations[window_size-1:], moving_avg, 'r-', 
                     label=f"Moving Average (window={window_size})")
        
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.canvas.draw_idle()
    
    # Update comparison between current best and global best if figure is still open
    if plt.fignum_exists(fig5.number):
        ax5.clear()
        x_values = np.linspace(0, 1, 100)
        
        # Import here to avoid circular imports
        from core import chaotic_function
        
        # Calculate error for current best and global best
        current_error = np.array([abs(chaotic_function(x, r) - current_best_predictor.predict(x)) for x in x_values])
        global_error = np.array([abs(chaotic_function(x, r) - global_best_predictor.predict(x)) for x in x_values])
        
        # Plot the errors
        ax5.plot(x_values, current_error, 'b-', label=f"Current Best Error (Gen {generation})", alpha=0.7)
        ax5.plot(x_values, global_error, 'g-', label=f"Global Best Error (Gen {global_best_generation})", alpha=0.7)
        
        # Add mean error lines
        current_mean_error = np.mean(current_error)
        global_mean_error = np.mean(global_error)
        ax5.axhline(y=current_mean_error, color='b', linestyle='--', alpha=0.5,
                   label=f"Current Mean Error: {current_mean_error:.4f}")
        ax5.axhline(y=global_mean_error, color='g', linestyle='--', alpha=0.5,
                   label=f"Global Mean Error: {global_mean_error:.4f}")
        
        ax5.set_xlabel("Input (x)")
        ax5.set_ylabel("Absolute Error")
        ax5.set_title(f"Error Comparison: Current vs Global Best")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        fig5.tight_layout()
        fig5.canvas.draw_idle()
    
    # Return updated figures and axes
    return [(fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4), (fig5, ax5)]

def save_results(results_dir, generation, figures_axes, population_size, mutation_rate, r, 
                current_best_fitness, current_best_predictor, global_best_fitness, 
                global_best_predictor, global_best_generation, is_final=False):
    """Save results, figures, and checkpoint."""
    suffix = "final" if is_final else f"gen{generation}"
    (fig1, _), (fig2, _), (fig3, _), (fig4, _), (fig5, _) = figures_axes
    
    # Save figures that are still open
    if plt.fignum_exists(fig1.number):
        fig1.savefig(os.path.join(results_dir, f"fitness_history_{suffix}.png"))
    if plt.fignum_exists(fig2.number):
        fig2.savefig(os.path.join(results_dir, f"prediction_vs_actual_{suffix}.png"))
    if plt.fignum_exists(fig3.number):
        fig3.savefig(os.path.join(results_dir, f"weight_evolution_{suffix}.png"))
    if plt.fignum_exists(fig4.number):
        fig4.savefig(os.path.join(results_dir, f"fitness_deviation_{suffix}.png"))
    if plt.fignum_exists(fig5.number):
        fig5.savefig(os.path.join(results_dir, f"error_comparison_{suffix}.png"))
    
    # Save the best weights and parameters to a text file
    with open(os.path.join(results_dir, f"best_weights_{suffix}.txt"), 'w') as f:
        f.write(f"{'Final ' if is_final else ''}Generation: {generation}\n")
        f.write(f"Parameters:\n")
        f.write(f"  - Population size: {population_size}\n")
        f.write(f"  - Mutation rate: {mutation_rate}\n")
        f.write(f"  - Chaos parameter (r): {r}\n\n")
        
        f.write(f"Current best predictor (generation {generation}):\n")
        f.write(f"  - Fitness: {current_best_fitness:.6f}\n")
        f.write(f"  - Weights:\n")
        f.write(f"    - a (x²): {current_best_predictor.weights[0]:.6f}\n")
        f.write(f"    - b (x): {current_best_predictor.weights[1]:.6f}\n")
        f.write(f"    - c (constant): {current_best_predictor.weights[2]:.6f}\n")
        f.write(f"  - Equation: f(x) = {current_best_predictor.weights[0]:.6f}x² + {current_best_predictor.weights[1]:.6f}x + {current_best_predictor.weights[2]:.6f}\n\n")
        
        f.write(f"GLOBAL best predictor (from generation {global_best_generation}):\n")
        f.write(f"  - Fitness: {global_best_fitness:.6f}\n")
        f.write(f"  - Weights:\n")
        f.write(f"    - a (x²): {global_best_predictor.weights[0]:.6f}\n")
        f.write(f"    - b (x): {global_best_predictor.weights[1]:.6f}\n")
        f.write(f"    - c (constant): {global_best_predictor.weights[2]:.6f}\n")
        f.write(f"  - Equation: f(x) = {global_best_predictor.weights[0]:.6f}x² + {global_best_predictor.weights[1]:.6f}x + {global_best_predictor.weights[2]:.6f}\n")

def save_checkpoint(results_dir, generation, fitness_history, weight_history, predictors, meta_predictors,
                   global_best_predictor, global_best_fitness, global_best_generation,
                   population_size, mutation_rate, r, is_final=False):
    """Save checkpoint for continuing the run later."""
    suffix = "final" if is_final else f"gen{generation}"
    
    checkpoint = {
        'generation': generation,
        'fitness_history': fitness_history,
        'weight_history': weight_history,
        'predictors': predictors,
        'meta_predictors': meta_predictors,
        'results_dir': results_dir,
        'global_best_predictor': global_best_predictor,
        'global_best_fitness': global_best_fitness,
        'global_best_generation': global_best_generation,
        'parameters': {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'r': r
        }
    }
    
    checkpoint_path = os.path.join(results_dir, f"checkpoint_{suffix}.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Create a symlink to the latest checkpoint for easy access
    latest_checkpoint_path = os.path.join(results_dir, "latest_checkpoint.pkl")
    if os.path.exists(latest_checkpoint_path):
        try:
            os.remove(latest_checkpoint_path)
        except:
            pass
    
    try:
        # Use relative path for the symlink to make it more portable
        os.symlink(f"checkpoint_{suffix}.pkl", latest_checkpoint_path)
    except:
        # If symlink fails, just copy the file
        import shutil
        shutil.copy2(checkpoint_path, latest_checkpoint_path)
    
    if is_final:
        print(f"\nFinal results saved to: {results_dir}")
        print(f"To continue this run later, use: --continue-from {os.path.join(results_dir, 'latest_checkpoint.pkl')}")
    else:
        print(f"Saved results and checkpoint at generation {generation}")

def cleanup_visualization():
    """Close all matplotlib figures."""
    plt.close('all') 
