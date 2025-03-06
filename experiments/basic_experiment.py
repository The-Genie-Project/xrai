import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from xrai.core import evolve
from xrai.visualization import plot_fitness_history, plot_prediction_vs_actual
from xrai.utils.metrics import calculate_fitness_stats
from xrai.utils.io import save_experiment_results

def run_basic_experiment(num_generations=100, population_size=20, mutation_rate=0.1, r=3.8, save_results=True):
    """
    Run a basic experiment with the evolutionary algorithm.
    
    Args:
        num_generations (int): Number of generations to evolve
        population_size (int): Size of the population
        mutation_rate (float): Rate of mutation for offspring
        r (float): Parameter for the chaotic function
        save_results (bool): Whether to save the results
    
    Returns:
        tuple: (fitness_history, best_predictor, best_meta_predictor)
    """
    print(f"Running basic experiment with parameters:")
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
    
    # Get the best predictor and meta-predictor
    predictor_fitness, meta_fitness = zip(*fitness_history)
    best_predictor_idx = np.argmax(predictor_fitness)
    best_meta_idx = np.argmax(meta_fitness)
    
    best_predictor = predictors[best_predictor_idx % len(predictors)]
    best_meta_predictor = meta_predictors[best_meta_idx % len(meta_predictors)]
    
    # Calculate and print statistics
    stats = calculate_fitness_stats(fitness_history)
    print("\nExperiment Results:")
    print(f"  - Final predictor fitness: {stats['predictor_final']:.4f}")
    print(f"  - Final meta-predictor fitness: {stats['meta_final']:.4f}")
    print(f"  - Max predictor fitness: {stats['predictor_max']:.4f}")
    print(f"  - Max meta-predictor fitness: {stats['meta_max']:.4f}")
    print(f"  - Predictor improvement: {stats['predictor_improvement']:.4f}")
    print(f"  - Meta-predictor improvement: {stats['meta_improvement']:.4f}")
    
    # Plot results
    fig1 = plot_fitness_history(fitness_history)
    fig2 = plot_prediction_vs_actual(best_predictor, r=r)
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"basic_experiment_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save figures
        fig1.savefig(os.path.join(results_dir, "fitness_history.png"))
        fig2.savefig(os.path.join(results_dir, "prediction_vs_actual.png"))
        
        # Save experiment data
        save_experiment_results(
            fitness_history, 
            best_predictor, 
            best_meta_predictor, 
            os.path.join(results_dir, "experiment_results.pkl")
        )
        
        print(f"\nResults saved to: {results_dir}")
    
    plt.show()
    
    return fitness_history, best_predictor, best_meta_predictor

if __name__ == "__main__":
    run_basic_experiment() 
