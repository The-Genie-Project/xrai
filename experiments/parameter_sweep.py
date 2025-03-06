import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

from ..core import evolve
from ..visualization import plot_fitness_history
from ..utils.metrics import calculate_fitness_stats
from ..utils.io import save_experiment_results

def run_parameter_sweep(
    generations_values=[50, 100, 200],
    population_values=[10, 20, 50],
    mutation_values=[0.05, 0.1, 0.2],
    r_values=[3.6, 3.8, 4.0],
    save_results=True
):
    """
    Run a parameter sweep experiment with the evolutionary algorithm.
    
    Args:
        generations_values (list): Values for number of generations
        population_values (list): Values for population size
        mutation_values (list): Values for mutation rate
        r_values (list): Values for chaotic function parameter
        save_results (bool): Whether to save the results
    
    Returns:
        dict: Results of the parameter sweep
    """
    print("Running parameter sweep experiment")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"parameter_sweep_{timestamp}")
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
    
    # Store results
    all_results = {}
    
    # Generate parameter combinations
    param_combinations = list(itertools.product(
        generations_values, 
        population_values, 
        mutation_values, 
        r_values
    ))
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    
    # Run experiments for each parameter combination
    for i, (num_generations, population_size, mutation_rate, r) in enumerate(param_combinations):
        print(f"\nRunning experiment {i+1}/{len(param_combinations)}:")
        print(f"  - Generations: {num_generations}")
        print(f"  - Population size: {population_size}")
        print(f"  - Mutation rate: {mutation_rate}")
        print(f"  - Chaos parameter (r): {r}")
        
        # Run the evolutionary algorithm
        fitness_history, predictors, meta_predictors = evolve(
            num_generations=num_generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            r=r
        )
        
        # Calculate statistics
        stats = calculate_fitness_stats(fitness_history)
        
        # Store results
        key = (num_generations, population_size, mutation_rate, r)
        all_results[key] = {
            'fitness_history': fitness_history,
            'stats': stats,
            'best_predictor': predictors[np.argmax([f[0] for f in fitness_history]) % len(predictors)],
            'best_meta_predictor': meta_predictors[np.argmax([f[1] for f in fitness_history]) % len(meta_predictors)]
        }
        
        # Print summary
        print(f"  Results:")
        print(f"    - Final predictor fitness: {stats['predictor_final']:.4f}")
        print(f"    - Final meta-predictor fitness: {stats['meta_final']:.4f}")
        
        # Save individual experiment results
        if save_results:
            # Create experiment directory
            exp_dir = os.path.join(
                results_dir, 
                f"gen{num_generations}_pop{population_size}_mut{mutation_rate:.2f}_r{r:.1f}"
            )
            os.makedirs(exp_dir, exist_ok=True)
            
            # Plot and save fitness history
            fig = plot_fitness_history(fitness_history)
            fig.savefig(os.path.join(exp_dir, "fitness_history.png"))
            plt.close(fig)
    
    # Find best parameter combination
    best_key = max(all_results.keys(), key=lambda k: all_results[k]['stats']['predictor_final'])
    best_result = all_results[best_key]
    
    print("\nParameter Sweep Complete")
    print(f"Best parameter combination:")
    print(f"  - Generations: {best_key[0]}")
    print(f"  - Population size: {best_key[1]}")
    print(f"  - Mutation rate: {best_key[2]:.2f}")
    print(f"  - Chaos parameter (r): {best_key[3]:.1f}")
    print(f"  - Final predictor fitness: {best_result['stats']['predictor_final']:.4f}")
    print(f"  - Final meta-predictor fitness: {best_result['stats']['meta_final']:.4f}")
    
    # Save overall results
    if save_results:
        # Create summary plots
        create_summary_plots(all_results, results_dir)
        
        print(f"\nResults saved to: {results_dir}")
    
    return all_results

def create_summary_plots(all_results, results_dir):
    """
    Create summary plots for parameter sweep results.
    
    Args:
        all_results (dict): Results from parameter sweep
        results_dir (str): Directory to save plots
    """
    # Extract parameters and results
    params = list(all_results.keys())
    predictor_fitness = [all_results[p]['stats']['predictor_final'] for p in params]
    meta_fitness = [all_results[p]['stats']['meta_final'] for p in params]
    
    # Create parameter-specific plots
    param_names = ['Generations', 'Population Size', 'Mutation Rate', 'Chaos Parameter (r)']
    
    for param_idx in range(4):
        # Get unique values for this parameter
        unique_values = sorted(set(p[param_idx] for p in params))
        
        # Group results by this parameter
        grouped_results = {}
        for p in params:
            param_value = p[param_idx]
            if param_value not in grouped_results:
                grouped_results[param_value] = []
            grouped_results[param_value].append(all_results[p]['stats']['predictor_final'])
        
        # Calculate mean fitness for each parameter value
        mean_fitness = [np.mean(grouped_results[v]) for v in unique_values]
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(unique_values)), mean_fitness)
        plt.xlabel(param_names[param_idx])
        plt.ylabel('Mean Predictor Fitness')
        plt.title(f'Effect of {param_names[param_idx]} on Predictor Fitness')
        plt.xticks(range(len(unique_values)), [str(v) for v in unique_values])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f"param_effect_{param_idx}.png"))
        plt.close()

if __name__ == "__main__":
    # Run with smaller parameter set for quicker execution
    run_parameter_sweep(
        generations_values=[50, 100],
        population_values=[10, 20],
        mutation_values=[0.05, 0.1],
        r_values=[3.6, 3.8]
    ) 
