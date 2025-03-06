"""
Functions for saving results and checkpoints.
"""
import os
import pickle
import matplotlib.pyplot as plt

def save_results(results_dir, generation, figures_axes, population_size, mutation_rate, r, 
                current_best_fitness, current_best_predictor, global_best_fitness, 
                global_best_predictor, global_best_generation, is_final=False,
                hierarchy_level=None, total_levels=None):
    """Save results, figures, and checkpoint."""
    suffix = "final" if is_final else f"gen{generation}"
    
    # Add hierarchy level to suffix if provided
    if hierarchy_level is not None and total_levels is not None:
        suffix += f"_level{hierarchy_level}of{total_levels}"
    
    # Check if visualization is disabled
    if figures_axes is not None:
        try:
            # Unpack the figures and axes
            main_fig, main_axes, *individual_figs_axes = figures_axes
            
            # Save the main dashboard figure
            main_fig.savefig(os.path.join(results_dir, f"dashboard_{suffix}.png"), dpi=150)
            
            # Save individual figures for detailed viewing
            for i, (fig, _) in enumerate(individual_figs_axes):
                if i == 0:
                    fig.savefig(os.path.join(results_dir, f"fitness_history_{suffix}.png"))
                elif i == 1:
                    fig.savefig(os.path.join(results_dir, f"prediction_vs_actual_{suffix}.png"))
                elif i == 2:
                    fig.savefig(os.path.join(results_dir, f"weight_evolution_{suffix}.png"))
                elif i == 3:
                    fig.savefig(os.path.join(results_dir, f"fitness_deviation_{suffix}.png"))
                elif i == 4:
                    fig.savefig(os.path.join(results_dir, f"error_comparison_{suffix}.png"))
                elif i == 5:
                    fig.savefig(os.path.join(results_dir, f"mutation_rate_evolution_{suffix}.png"))
                elif i == 6:
                    fig.savefig(os.path.join(results_dir, f"chaos_phase_space_{suffix}.png"))
                elif i == 7:
                    fig.savefig(os.path.join(results_dir, f"chaos_cobweb_{suffix}.png"))
        except Exception as e:
            print(f"Error saving figures: {e}")
            print("Continuing without saving figures...")
    
    # Save the best weights and parameters to a text file
    with open(os.path.join(results_dir, f"best_weights_{suffix}.txt"), 'w') as f:
        f.write(f"{'Final ' if is_final else ''}Generation: {generation}\n")
        
        # Add hierarchy information if provided
        if hierarchy_level is not None and total_levels is not None:
            f.write(f"Hierarchy Level: {hierarchy_level}/{total_levels}\n")
        
        f.write(f"Parameters:\n")
        f.write(f"  - Population size: {population_size}\n")
        f.write(f"  - Initial mutation rate: {mutation_rate}\n")
        f.write(f"  - Chaos parameter (r): {r}\n\n")
        
        # Add mutation rate information if available
        if hasattr(current_best_predictor, 'mutation_rate'):
            f.write(f"  - Current best mutation rate: {current_best_predictor.mutation_rate:.6f}\n")
        if hasattr(global_best_predictor, 'mutation_rate'):
            f.write(f"  - Global best mutation rate: {global_best_predictor.mutation_rate:.6f}\n\n")
        
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
                   population_size, mutation_rate, r, is_final=False,
                   current_hierarchy_level=None, hierarchy_generation_counts=None,
                   chaos_values_history=None):
    """Save checkpoint for continuing the run later."""
    suffix = "final" if is_final else f"gen{generation}"
    
    # Add hierarchy level to suffix if provided
    if current_hierarchy_level is not None and hierarchy_generation_counts is not None:
        suffix += f"_level{current_hierarchy_level+1}of{len(hierarchy_generation_counts)}"
    
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
    
    # Add chaos values history if provided
    if chaos_values_history is not None:
        checkpoint['chaos_values_history'] = chaos_values_history
    
    # Add hierarchy information if provided
    if current_hierarchy_level is not None:
        checkpoint['current_hierarchy_level'] = current_hierarchy_level
    if hierarchy_generation_counts is not None:
        checkpoint['hierarchy_generation_counts'] = hierarchy_generation_counts
    
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
        hierarchy_info = ""
        if current_hierarchy_level is not None and hierarchy_generation_counts is not None:
            hierarchy_info = f" (Level {current_hierarchy_level+1}/{len(hierarchy_generation_counts)})"
        print(f"Saved results and checkpoint at generation {generation}{hierarchy_info}") 
