#!/usr/bin/env python3
"""
Live streaming evolution script for the XRAI package.

This script runs the evolutionary algorithm in a continuous loop,
displaying results in real-time without blocking.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from core import evolve, chaotic_function, Predictor, MetaPredictor
from visualization import plot_fitness_history, plot_prediction_vs_actual
from utils.metrics import calculate_fitness_stats

# Enable interactive mode for matplotlib
plt.ion()

def live_evolve(population_size=20, mutation_rate=0.1, r=3.8, 
                update_interval=10, max_generations=10000, 
                save_interval=100):
    """
    Run the evolutionary algorithm with live updates.
    
    Args:
        population_size (int): Size of the population
        mutation_rate (float): Rate of mutation for offspring
        r (float): Parameter for the chaotic function
        update_interval (int): Number of generations between visual updates
        max_generations (int): Maximum number of generations to run
        save_interval (int): Number of generations between saving results
    """
    print(f"Starting live evolution with parameters:")
    print(f"  - Population size: {population_size}")
    print(f"  - Mutation rate: {mutation_rate}")
    print(f"  - Chaos parameter (r): {r}")
    print(f"  - Update interval: {update_interval}")
    print(f"Press Ctrl+C to stop the evolution")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"live_evolution_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize population of predictors and meta-predictors
    predictors = [Predictor() for _ in range(population_size)]
    meta_predictors = [MetaPredictor() for _ in range(population_size)]
    
    # Initialize fitness history
    fitness_history = []
    
    # Create figures for live updates
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    # Generation counter
    generation = 0
    
    try:
        while generation < max_generations:
            # Run a batch of generations
            for _ in range(update_interval):
                x = np.random.uniform(0, 1)  # Random initial state
                true_value = chaotic_function(x, r)  # The actual outcome
                
                # Evaluate predictor fitness
                predictor_fitness = np.array([1 - abs(p.predict(x) - true_value) for p in predictors])
                
                # Evaluate meta-predictor fitness
                meta_fitness = np.array([1 - abs(m.predict_fitness(p, x) - f) 
                                        for m, p, f in zip(meta_predictors, predictors, predictor_fitness)])
                
                # Selection: Keep the top 50% of predictors and meta-predictors
                top_indices = np.argsort(predictor_fitness)[-population_size//2:]
                predictors = [predictors[i] for i in top_indices]
                meta_predictors = [meta_predictors[i] for i in top_indices]
                
                # Reproduce: Generate new mutated offspring
                predictors += [p.mutate(mutation_rate) for p in predictors]
                meta_predictors += [m.mutate(mutation_rate) for m in meta_predictors]
                
                # Store best fitness
                fitness_history.append((np.max(predictor_fitness), np.max(meta_fitness)))
                
                generation += 1
            
            # Get the best predictor
            predictor_fitness_values, meta_fitness_values = zip(*fitness_history)
            best_predictor_idx = np.argmax(predictor_fitness_values)
            best_meta_idx = np.argmax(meta_fitness_values)
            
            best_predictor = predictors[best_predictor_idx % len(predictors)]
            best_meta_predictor = meta_predictors[best_meta_idx % len(meta_predictors)]
            
            # Calculate and print statistics
            stats = calculate_fitness_stats(fitness_history)
            print(f"\nGeneration {generation}:")
            print(f"  - Current predictor fitness: {predictor_fitness_values[-1]:.4f}")
            print(f"  - Current meta-predictor fitness: {meta_fitness_values[-1]:.4f}")
            print(f"  - Max predictor fitness: {stats['predictor_max']:.4f}")
            print(f"  - Max meta-predictor fitness: {stats['meta_max']:.4f}")
            
            # Update plots
            ax1.clear()
            generations = np.arange(len(fitness_history))
            ax1.plot(generations, predictor_fitness_values, label="Predictor Fitness")
            ax1.plot(generations, meta_fitness_values, label="Meta-Predictor Fitness")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness")
            ax1.set_title(f"Evolution Progress (Generation {generation})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            fig1.tight_layout()
            
            ax2.clear()
            x_values = np.linspace(0, 1, 100)
            actual_values = np.array([chaotic_function(x, r) for x in x_values])
            predicted_values = np.array([best_predictor.predict(x) for x in x_values])
            ax2.scatter(x_values, actual_values, label="Actual", alpha=0.7)
            ax2.scatter(x_values, predicted_values, label="Predicted", alpha=0.7)
            ax2.set_xlabel("Input (x)")
            ax2.set_ylabel("Output")
            ax2.set_title(f"Best Predictor Performance (Generation {generation})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            
            # Draw and pause to update the figures
            fig1.canvas.draw_idle()
            fig2.canvas.draw_idle()
            plt.pause(0.01)
            
            # Save results periodically
            if generation % save_interval == 0:
                fig1.savefig(os.path.join(results_dir, f"fitness_history_gen{generation}.png"))
                fig2.savefig(os.path.join(results_dir, f"prediction_vs_actual_gen{generation}.png"))
                print(f"Saved results at generation {generation}")
    
    except KeyboardInterrupt:
        print("\nEvolution stopped by user")
    
    # Final save
    fig1.savefig(os.path.join(results_dir, f"fitness_history_final.png"))
    fig2.savefig(os.path.join(results_dir, f"prediction_vs_actual_final.png"))
    print(f"\nFinal results saved to: {results_dir}")
    
    # Keep plots open until user closes them
    plt.ioff()
    plt.show()
    
    return fitness_history, predictors, meta_predictors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Evolution of Predictors and Meta-Predictors")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    parser.add_argument("--mutation", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--r", type=float, default=3.8, help="Chaos parameter")
    parser.add_argument("--update", type=int, default=10, help="Update interval (generations)")
    parser.add_argument("--max-gen", type=int, default=10000, help="Maximum generations")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval (generations)")
    
    args = parser.parse_args()
    
    live_evolve(
        population_size=args.population,
        mutation_rate=args.mutation,
        r=args.r,
        update_interval=args.update,
        max_generations=args.max_gen,
        save_interval=args.save_interval
    )
