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
import copy
import signal
import sys

from core import evolve, chaotic_function, Predictor, MetaPredictor
from utils.metrics import calculate_fitness_stats
import visualization_manager as visuals

# Global flag for clean termination
terminate_flag = False

def signal_handler(sig, frame):
    """Handle Ctrl+C signal to ensure clean termination."""
    global terminate_flag
    print("\nReceived termination signal. Cleaning up...")
    terminate_flag = True

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def live_evolve(population_size=20, mutation_rate=0.1, r=3.8, 
                update_interval=10, max_generations=10000, 
                save_interval=100, continue_from=None,
                hierarchy_levels=3, level_scaling_factor=0.5):
    """
    Run the evolutionary algorithm with live updates and hierarchical refinement.
    
    Args:
        population_size (int): Size of the population
        mutation_rate (float): Rate of mutation for offspring
        r (float): Parameter for the chaotic function
        update_interval (int): Number of generations between visual updates
        max_generations (int): Maximum number of generations to run
        save_interval (int): Number of generations between saving results
        continue_from (str): Path to a previous run's checkpoint to continue from
        hierarchy_levels (int): Number of hierarchical levels for refinement
        level_scaling_factor (float): Factor to scale generations at each level
    """
    global terminate_flag
    terminate_flag = False
    
    # Initialize variables for continuing from a previous run
    start_generation = 0
    fitness_history = []
    weight_history = []
    predictors = None
    meta_predictors = None
    results_dir = None
    
    # Variables to track the globally best predictor across all generations
    global_best_predictor = None
    global_best_fitness = -float('inf')
    global_best_generation = 0
    
    # Hierarchical evolution tracking
    current_hierarchy_level = 0
    hierarchy_generation_counts = []
    
    # Calculate generations for each hierarchy level using scaling factor
    remaining_gens = max_generations
    for level in range(hierarchy_levels):
        level_gens = int(remaining_gens * level_scaling_factor)
        hierarchy_generation_counts.append(level_gens)
        remaining_gens -= level_gens
    
    # Add any remaining generations to the last level
    hierarchy_generation_counts[-1] += remaining_gens
    
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
                
                # Load global best predictor information
                global_best_predictor = checkpoint.get('global_best_predictor', None)
                global_best_fitness = checkpoint.get('global_best_fitness', -float('inf'))
                global_best_generation = checkpoint.get('global_best_generation', 0)
                
                # Load hierarchy information if available
                current_hierarchy_level = checkpoint.get('current_hierarchy_level', 0)
                hierarchy_generation_counts = checkpoint.get('hierarchy_generation_counts', hierarchy_generation_counts)
                
                # Override parameters if they were saved
                saved_params = checkpoint.get('parameters', {})
                population_size = saved_params.get('population_size', population_size)
                mutation_rate = saved_params.get('mutation_rate', mutation_rate)
                r = saved_params.get('r', r)
                
                visuals.print_checkpoint_info(
                    checkpoint, start_generation, population_size, mutation_rate, r,
                    global_best_predictor, global_best_fitness, global_best_generation
                )
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
        visuals.print_start_info(population_size, mutation_rate, r, update_interval)
        print(f"Hierarchical evolution with {hierarchy_levels} levels")
        print(f"Generation distribution across levels: {hierarchy_generation_counts}")
    
    # Create results directory if not continuing or if continuing but no results_dir
    if not results_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"live_evolution_{timestamp}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize population if not continuing or if predictors/meta_predictors are None
    if predictors is None or meta_predictors is None:
        predictors = [Predictor() for _ in range(population_size)]
        meta_predictors = [MetaPredictor() for _ in range(population_size)]
    
    # Set up visualization
    figures_axes = visuals.setup_visualization()
    
    # Generation counter
    generation = start_generation
    total_generations = sum(hierarchy_generation_counts)
    
    try:
        # Iterate through hierarchy levels
        while current_hierarchy_level < hierarchy_levels and not terminate_flag:
            print(f"\n===== Starting Hierarchy Level {current_hierarchy_level + 1}/{hierarchy_levels} =====")
            print(f"Planning to run {hierarchy_generation_counts[current_hierarchy_level]} generations at this level")
            
            level_generations = 0
            level_max_generations = hierarchy_generation_counts[current_hierarchy_level]
            
            # Run evolution at the current hierarchy level
            while level_generations < level_max_generations and not terminate_flag:
                # Run a batch of generations
                batch_best_predictor = None
                batch_best_fitness = -float('inf')
                batch_best_meta = None
                
                for _ in range(update_interval):
                    if terminate_flag:
                        break
                        
                    x = np.random.uniform(0, 1)  # Random initial state
                    true_value = chaotic_function(x, r)  # The actual outcome
                    
                    # Evaluate predictor fitness
                    predictor_fitness = np.array([1 - abs(p.predict(x) - true_value) for p in predictors])
                    
                    # Evaluate meta-predictor fitness
                    meta_fitness = np.array([1 - abs(m.predict_fitness(p, x) - f) 
                                            for m, p, f in zip(meta_predictors, predictors, predictor_fitness)])
                    
                    # Track the best predictor in this batch
                    current_best_idx = np.argmax(predictor_fitness)
                    current_best_fitness = predictor_fitness[current_best_idx]
                    
                    if current_best_fitness > batch_best_fitness:
                        batch_best_fitness = current_best_fitness
                        batch_best_predictor = copy.deepcopy(predictors[current_best_idx])
                        batch_best_meta = copy.deepcopy(meta_predictors[current_best_idx])
                    
                    # Check if we have a new global best predictor
                    if current_best_fitness > global_best_fitness:
                        global_best_fitness = current_best_fitness
                        global_best_predictor = copy.deepcopy(predictors[current_best_idx])
                        global_best_generation = generation
                    
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
                    level_generations += 1
                
                if terminate_flag:
                    break
                
                # After each batch, use the best predictor from this batch as a seed
                # for the next batch by replacing the worst predictor in the population
                if batch_best_predictor is not None:
                    # Find the worst predictor in the current population
                    x_test = np.random.uniform(0, 1)
                    true_value_test = chaotic_function(x_test, r)
                    current_fitness = np.array([1 - abs(p.predict(x_test) - true_value_test) for p in predictors])
                    worst_idx = np.argmin(current_fitness)
                    
                    # Replace the worst predictor with the best from the batch
                    predictors[worst_idx] = copy.deepcopy(batch_best_predictor)
                    meta_predictors[worst_idx] = copy.deepcopy(batch_best_meta)
                
                # Get the best predictor in the current population
                predictor_fitness_values, meta_fitness_values = zip(*fitness_history)
                current_best_idx = np.argmax([p_fitness for p_fitness, _ in fitness_history[-population_size:]])
                current_best_predictor = predictors[current_best_idx]
                current_best_fitness = predictor_fitness_values[-population_size + current_best_idx]
                
                # Store current best weights
                weight_history.append(current_best_predictor.weights.copy())
                
                # Calculate and print statistics
                stats = calculate_fitness_stats(fitness_history)
                visuals.print_generation_stats(
                    generation, predictor_fitness_values[-1], meta_fitness_values[-1],
                    current_best_fitness, current_best_predictor,
                    global_best_fitness, global_best_predictor, global_best_generation,
                    hierarchy_level=current_hierarchy_level+1, total_levels=hierarchy_levels
                )
                
                # Update visualization plots
                figures_axes = visuals.update_plots(
                    figures_axes, generation, fitness_history, weight_history,
                    current_best_predictor, global_best_predictor, global_best_fitness, global_best_generation, r
                )
                
                # Pause briefly to update the UI and process events
                plt.pause(0.01)
                
                # Save results periodically
                if generation % save_interval == 0:
                    visuals.save_results(
                        results_dir, generation, figures_axes,
                        population_size, mutation_rate, r,
                        current_best_fitness, current_best_predictor,
                        global_best_fitness, global_best_predictor, global_best_generation
                    )
                    
                    visuals.save_checkpoint(
                        results_dir, generation, fitness_history, weight_history,
                        predictors, meta_predictors, global_best_predictor,
                        global_best_fitness, global_best_generation,
                        population_size, mutation_rate, r,
                        current_hierarchy_level=current_hierarchy_level,
                        hierarchy_generation_counts=hierarchy_generation_counts
                    )
            
            # At the end of each hierarchy level, seed the next level with the best solution
            if not terminate_flag and current_hierarchy_level < hierarchy_levels - 1:
                print(f"\n===== Completed Hierarchy Level {current_hierarchy_level + 1}/{hierarchy_levels} =====")
                print(f"Seeding next level with best solution (fitness: {global_best_fitness:.6f})")
                
                # Create a new population centered around the best solution
                new_predictors = []
                new_meta_predictors = []
                
                # Add the global best predictor
                new_predictors.append(copy.deepcopy(global_best_predictor))
                
                # Find the best meta-predictor
                x_test = np.random.uniform(0, 1)
                true_value_test = chaotic_function(x_test, r)
                predictor_fitness_test = np.array([1 - abs(p.predict(x_test) - true_value_test) for p in predictors])
                meta_fitness_test = np.array([1 - abs(m.predict_fitness(p, x_test) - f) 
                                            for m, p, f in zip(meta_predictors, predictors, predictor_fitness_test)])
                best_meta_idx = np.argmax(meta_fitness_test)
                best_meta = copy.deepcopy(meta_predictors[best_meta_idx])
                new_meta_predictors.append(best_meta)
                
                # Create variations of the best predictor with different mutation rates
                # Use lower mutation rates for higher hierarchy levels to fine-tune
                level_mutation_rate = mutation_rate / (current_hierarchy_level + 1)
                
                for _ in range(population_size - 1):
                    new_predictors.append(global_best_predictor.mutate(level_mutation_rate))
                    new_meta_predictors.append(best_meta.mutate(level_mutation_rate))
                
                # Replace the population with the new one
                predictors = new_predictors
                meta_predictors = new_meta_predictors
                
                # Adjust mutation rate for the next level (optional)
                mutation_rate = max(0.01, mutation_rate * 0.8)  # Reduce mutation rate but keep it above 0.01
                
                print(f"Adjusted mutation rate for next level: {mutation_rate:.4f}")
            
            # Move to the next hierarchy level
            current_hierarchy_level += 1
    
    except Exception as e:
        print(f"\nError during evolution: {e}")
        import traceback
        traceback.print_exc()
    
    # Final save
    print("\nSaving final results...")
    
    visuals.save_results(
        results_dir, generation, figures_axes,
        population_size, mutation_rate, r,
        current_best_fitness, current_best_predictor,
        global_best_fitness, global_best_predictor, global_best_generation,
        is_final=True
    )
    
    visuals.save_checkpoint(
        results_dir, generation, fitness_history, weight_history,
        predictors, meta_predictors, global_best_predictor,
        global_best_fitness, global_best_generation,
        population_size, mutation_rate, r,
        current_hierarchy_level=current_hierarchy_level,
        hierarchy_generation_counts=hierarchy_generation_counts,
        is_final=True
    )
    
    # Close all remaining figures
    visuals.cleanup_visualization()
    
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
    parser.add_argument("--hierarchy-levels", type=int, default=3, help="Number of hierarchical refinement levels")
    parser.add_argument("--level-scaling", type=float, default=0.5, help="Scaling factor for generations at each level")
    
    args = parser.parse_args()
    
    try:
        live_evolve(
            population_size=args.population,
            mutation_rate=args.mutation,
            r=args.r,
            update_interval=args.update,
            max_generations=args.max_gen,
            save_interval=args.save_interval,
            continue_from=args.continue_from,
            hierarchy_levels=args.hierarchy_levels,
            level_scaling_factor=args.level_scaling,
        )
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)
