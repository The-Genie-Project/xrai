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
import pickle
from datetime import datetime

from core import evolve, chaotic_function, Predictor, MetaPredictor
from visualization import plot_fitness_history, plot_prediction_vs_actual
from utils.metrics import calculate_fitness_stats

# Enable interactive mode for matplotlib
plt.ion()

def live_evolve(population_size=20, mutation_rate=0.1, r=3.8, 
                update_interval=10, max_generations=10000, 
                save_interval=100, continue_from=None):
    """
    Run the evolutionary algorithm with live updates.
    
    Args:
        population_size (int): Size of the population
        mutation_rate (float): Rate of mutation for offspring
        r (float): Parameter for the chaotic function
        update_interval (int): Number of generations between visual updates
        max_generations (int): Maximum number of generations to run
        save_interval (int): Number of generations between saving results
        continue_from (str): Path to a previous run's checkpoint to continue from
    """
    # Initialize variables for continuing from a previous run
    start_generation = 0
    fitness_history = []
    weight_history = []
    predictors = None
    meta_predictors = None
    results_dir = None
    
    # Check if continuing from a previous run
    if continue_from:
        if os.path.exists(continue_from):
            print(f"Loading previous run from: {continue_from}")
            try:
                with open(continue_from, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                # Extract data from checkpoint
                start_generation = checkpoint.get('generation', 0)
                fitness_history = checkpoint.get('fitness_history', [])
                weight_history = checkpoint.get('weight_history', [])
                predictors = checkpoint.get('predictors', None)
                meta_predictors = checkpoint.get('meta_predictors', None)
                results_dir = checkpoint.get('results_dir', None)
                
                # Override parameters if they were saved
                saved_params = checkpoint.get('parameters', {})
                population_size = saved_params.get('population_size', population_size)
                mutation_rate = saved_params.get('mutation_rate', mutation_rate)
                r = saved_params.get('r', r)
                
                print(f"Successfully loaded checkpoint from generation {start_generation}")
                print(f"Continuing with parameters:")
                print(f"  - Population size: {population_size}")
                print(f"  - Mutation rate: {mutation_rate}")
                print(f"  - Chaos parameter (r): {r}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting a new run instead.")
                continue_from = None
        else:
            print(f"Checkpoint file not found: {continue_from}")
            print("Starting a new run instead.")
            continue_from = None
    
    # If not continuing or failed to load, print starting parameters
    if not continue_from:
        print(f"Starting live evolution with parameters:")
        print(f"  - Population size: {population_size}")
        print(f"  - Mutation rate: {mutation_rate}")
        print(f"  - Chaos parameter (r): {r}")
        print(f"  - Update interval: {update_interval}")
    
    print(f"Press Ctrl+C to stop the evolution")
    
    # Create results directory if not continuing or if continuing but no results_dir
    if not results_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"live_evolution_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize population if not continuing or if predictors/meta_predictors are None
    if predictors is None or meta_predictors is None:
        predictors = [Predictor() for _ in range(population_size)]
        meta_predictors = [MetaPredictor() for _ in range(population_size)]
    
    # Create figures for live updates
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig4, ax4 = plt.subplots(figsize=(10, 5))  # New figure for fitness deviation
    
    # Generation counter
    generation = start_generation
    
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
            
            # Store best weights
            weight_history.append(best_predictor.weights.copy())
            
            # Calculate and print statistics
            stats = calculate_fitness_stats(fitness_history)
            print(f"\nGeneration {generation}:")
            print(f"  - Current predictor fitness: {predictor_fitness_values[-1]:.4f}")
            print(f"  - Current meta-predictor fitness: {meta_fitness_values[-1]:.4f}")
            print(f"  - Max predictor fitness: {stats['predictor_max']:.4f}")
            print(f"  - Max meta-predictor fitness: {stats['meta_max']:.4f}")
            print(f"  - Best predictor weights: a={best_predictor.weights[0]:.4f}, b={best_predictor.weights[1]:.4f}, c={best_predictor.weights[2]:.4f}")
            print(f"  - Equation: f(x) = {best_predictor.weights[0]:.4f}x² + {best_predictor.weights[1]:.4f}x + {best_predictor.weights[2]:.4f}")
            
            # Update fitness plot
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
            
            # Update prediction vs actual plot
            ax2.clear()
            x_values = np.linspace(0, 1, 100)
            actual_values = np.array([chaotic_function(x, r) for x in x_values])
            predicted_values = np.array([best_predictor.predict(x) for x in x_values])
            
            # Plot the target function
            ax2.plot(x_values, actual_values, 'r-', label="Target (Chaotic Function)", alpha=0.7)
            
            # Plot the prediction
            ax2.plot(x_values, predicted_values, 'b-', label="Prediction", alpha=0.7)
            
            # Add scatter points for better visibility
            ax2.scatter(x_values[::10], actual_values[::10], color='red', alpha=0.7)
            ax2.scatter(x_values[::10], predicted_values[::10], color='blue', alpha=0.7)
            
            ax2.set_xlabel("Input (x)")
            ax2.set_ylabel("Output")
            ax2.set_title(f"Best Predictor Performance (Generation {generation})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            
            # Update weights plot
            ax3.clear()
            if len(weight_history) > 0:
                weight_history_array = np.array(weight_history)
                ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 0], label="Weight a (x²)")
                ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 1], label="Weight b (x)")
                ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 2], label="Weight c (constant)")
                ax3.set_xlabel("Generation")
                ax3.set_ylabel("Weight Value")
                ax3.set_title(f"Evolution of Best Predictor Weights (Generation {generation})")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                fig3.tight_layout()
            
            # Update fitness deviation plot
            ax4.clear()
            if len(fitness_history) > 0:
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
            
            # Draw and pause to update the figures
            fig1.canvas.draw_idle()
            fig2.canvas.draw_idle()
            fig3.canvas.draw_idle()
            fig4.canvas.draw_idle()
            plt.pause(0.01)
            
            # Save results periodically
            if generation % save_interval == 0:
                # Save figures
                fig1.savefig(os.path.join(results_dir, f"fitness_history_gen{generation}.png"))
                fig2.savefig(os.path.join(results_dir, f"prediction_vs_actual_gen{generation}.png"))
                fig3.savefig(os.path.join(results_dir, f"weight_evolution_gen{generation}.png"))
                fig4.savefig(os.path.join(results_dir, f"fitness_deviation_gen{generation}.png"))
                
                # Save the best weights and parameters to a text file
                with open(os.path.join(results_dir, f"best_weights_gen{generation}.txt"), 'w') as f:
                    f.write(f"Generation: {generation}\n")
                    f.write(f"Parameters:\n")
                    f.write(f"  - Population size: {population_size}\n")
                    f.write(f"  - Mutation rate: {mutation_rate}\n")
                    f.write(f"  - Chaos parameter (r): {r}\n\n")
                    f.write(f"Best predictor fitness: {stats['predictor_max']:.6f}\n")
                    f.write(f"Best predictor weights:\n")
                    f.write(f"  - a (x²): {best_predictor.weights[0]:.6f}\n")
                    f.write(f"  - b (x): {best_predictor.weights[1]:.6f}\n")
                    f.write(f"  - c (constant): {best_predictor.weights[2]:.6f}\n")
                    f.write(f"Equation: f(x) = {best_predictor.weights[0]:.6f}x² + {best_predictor.weights[1]:.6f}x + {best_predictor.weights[2]:.6f}\n")
                
                # Save checkpoint for continuing the run later
                checkpoint = {
                    'generation': generation,
                    'fitness_history': fitness_history,
                    'weight_history': weight_history,
                    'predictors': predictors,
                    'meta_predictors': meta_predictors,
                    'results_dir': results_dir,
                    'parameters': {
                        'population_size': population_size,
                        'mutation_rate': mutation_rate,
                        'r': r
                    }
                }
                
                checkpoint_path = os.path.join(results_dir, f"checkpoint_gen{generation}.pkl")
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                
                # Create a symlink to the latest checkpoint for easy access
                latest_checkpoint_path = os.path.join(results_dir, "latest_checkpoint.pkl")
                if os.path.exists(latest_checkpoint_path):
                    os.remove(latest_checkpoint_path)
                
                # Use relative path for the symlink to make it more portable
                os.symlink(f"checkpoint_gen{generation}.pkl", latest_checkpoint_path)
                
                print(f"Saved results and checkpoint at generation {generation}")
    
    except KeyboardInterrupt:
        print("\nEvolution stopped by user")
    
    # Final save
    fig1.savefig(os.path.join(results_dir, f"fitness_history_final.png"))
    fig2.savefig(os.path.join(results_dir, f"prediction_vs_actual_final.png"))
    fig3.savefig(os.path.join(results_dir, f"weight_evolution_final.png"))
    fig4.savefig(os.path.join(results_dir, f"fitness_deviation_final.png"))
    
    # Save the final best weights and parameters to a text file
    with open(os.path.join(results_dir, "best_weights_final.txt"), 'w') as f:
        f.write(f"Final Generation: {generation}\n")
        f.write(f"Parameters:\n")
        f.write(f"  - Population size: {population_size}\n")
        f.write(f"  - Mutation rate: {mutation_rate}\n")
        f.write(f"  - Chaos parameter (r): {r}\n\n")
        f.write(f"Best predictor fitness: {stats['predictor_max']:.6f}\n")
        f.write(f"Best predictor weights:\n")
        f.write(f"  - a (x²): {best_predictor.weights[0]:.6f}\n")
        f.write(f"  - b (x): {best_predictor.weights[1]:.6f}\n")
        f.write(f"  - c (constant): {best_predictor.weights[2]:.6f}\n")
        f.write(f"Equation: f(x) = {best_predictor.weights[0]:.6f}x² + {best_predictor.weights[1]:.6f}x + {best_predictor.weights[2]:.6f}\n")
    
    # Save final checkpoint
    final_checkpoint = {
        'generation': generation,
        'fitness_history': fitness_history,
        'weight_history': weight_history,
        'predictors': predictors,
        'meta_predictors': meta_predictors,
        'results_dir': results_dir,
        'parameters': {
            'population_size': population_size,
            'mutation_rate': mutation_rate,
            'r': r
        }
    }
    
    final_checkpoint_path = os.path.join(results_dir, f"checkpoint_final.pkl")
    with open(final_checkpoint_path, 'wb') as f:
        pickle.dump(final_checkpoint, f)
    
    # Update the latest checkpoint symlink
    latest_checkpoint_path = os.path.join(results_dir, "latest_checkpoint.pkl")
    if os.path.exists(latest_checkpoint_path):
        os.remove(latest_checkpoint_path)
    os.symlink(f"checkpoint_final.pkl", latest_checkpoint_path)
    
    print(f"\nFinal results saved to: {results_dir}")
    print(f"To continue this run later, use: --continue-from {os.path.join(results_dir, 'latest_checkpoint.pkl')}")
    
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
    parser.add_argument("--continue-from", type=str, help="Path to checkpoint file to continue from")
    
    args = parser.parse_args()
    
    live_evolve(
        population_size=args.population,
        mutation_rate=args.mutation,
        r=args.r,
        update_interval=args.update,
        max_generations=args.max_gen,
        save_interval=args.save_interval,
        continue_from=args.continue_from
    )
