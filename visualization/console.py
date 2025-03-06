"""
Console output functions for visualization.
"""

def print_start_info(population_size, mutation_rate, r, update_interval):
    """Print information about the starting parameters."""
    print(f"Starting live evolution with parameters:")
    print(f"  - Population size: {population_size}")
    print(f"  - Mutation rate: {mutation_rate}")
    print(f"  - Chaos parameter (r): {r}")
    print(f"  - Update interval: {update_interval}")
    print(f"Press Ctrl+C to stop the evolution")

def print_generation_stats(generation, predictor_fitness, meta_fitness, current_best_fitness, current_best_predictor, 
                          global_best_fitness, global_best_predictor, global_best_generation, 
                          hierarchy_level=None, total_levels=None):
    """Print statistics for the current generation."""
    # Include hierarchy level information if provided
    hierarchy_info = ""
    if hierarchy_level is not None and total_levels is not None:
        hierarchy_info = f" (Hierarchy Level {hierarchy_level}/{total_levels})"
    
    print(f"\nGeneration {generation}{hierarchy_info}:")
    print(f"  - Current predictor fitness: {predictor_fitness:.4f}")
    print(f"  - Current meta-predictor fitness: {meta_fitness:.4f}")
    
    # Print information about current best and global best
    print(f"  - Current best predictor fitness: {current_best_fitness:.4f}")
    print(f"  - Current weights: a={current_best_predictor.weights[0]:.4f}, b={current_best_predictor.weights[1]:.4f}, c={current_best_predictor.weights[2]:.4f}")
    print(f"  - Current equation: f(x) = {current_best_predictor.weights[0]:.4f}x² + {current_best_predictor.weights[1]:.4f}x + {current_best_predictor.weights[2]:.4f}")
    
    # Print mutation rate if available
    if hasattr(current_best_predictor, 'mutation_rate'):
        print(f"  - Current best mutation rate: {current_best_predictor.mutation_rate:.4f}")
    
    print(f"  - GLOBAL best predictor fitness: {global_best_fitness:.4f} (from generation {global_best_generation})")
    print(f"  - GLOBAL weights: a={global_best_predictor.weights[0]:.4f}, b={global_best_predictor.weights[1]:.4f}, c={global_best_predictor.weights[2]:.4f}")
    print(f"  - GLOBAL equation: f(x) = {global_best_predictor.weights[0]:.4f}x² + {global_best_predictor.weights[1]:.4f}x + {global_best_predictor.weights[2]:.4f}")
    
    # Print global best mutation rate if available
    if hasattr(global_best_predictor, 'mutation_rate'):
        print(f"  - GLOBAL best mutation rate: {global_best_predictor.mutation_rate:.4f}")

def print_checkpoint_info(checkpoint, start_generation, population_size, mutation_rate, r, global_best_predictor, global_best_fitness, global_best_generation):
    """Print information about a loaded checkpoint."""
    print(f"Successfully loaded checkpoint from generation {start_generation}")
    print(f"Continuing with parameters:")
    print(f"  - Population size: {population_size}")
    print(f"  - Mutation rate: {mutation_rate}")
    print(f"  - Chaos parameter (r): {r}")
    
    # Print hierarchy information if available
    if 'current_hierarchy_level' in checkpoint:
        current_level = checkpoint.get('current_hierarchy_level', 0)
        hierarchy_counts = checkpoint.get('hierarchy_generation_counts', [])
        print(f"  - Current hierarchy level: {current_level + 1}/{len(hierarchy_counts)}")
        print(f"  - Hierarchy generation distribution: {hierarchy_counts}")
        
        # Calculate and print when weights will be updated for each level
        if hierarchy_counts:
            print(f"  - Hierarchy level progression:")
            cumulative_gens = 0
            for i, gen_count in enumerate(hierarchy_counts):
                level_start = cumulative_gens
                level_end = cumulative_gens + gen_count
                status = "COMPLETED" if i < current_level else "CURRENT" if i == current_level else "PENDING"
                progress = ""
                if i == current_level:
                    current_level_progress = start_generation - level_start
                    progress = f" (progress: {current_level_progress}/{gen_count} generations, {current_level_progress/gen_count*100:.1f}%)"
                
                print(f"    - Level {i+1}: generations {level_start+1}-{level_end} [{status}]{progress}")
                cumulative_gens += gen_count
            
            # Print when the best weights will be updated
            if current_level < len(hierarchy_counts) - 1:
                next_update = cumulative_gens = sum(hierarchy_counts[:current_level+1])
                gens_remaining = next_update - start_generation
                print(f"  - Next weights update: after generation {next_update} ({gens_remaining} generations remaining)")
    
    if global_best_predictor:
        print(f"Global best predictor (from generation {global_best_generation}):")
        print(f"  - Fitness: {global_best_fitness:.6f}")
        print(f"  - Weights: a={global_best_predictor.weights[0]:.4f}, b={global_best_predictor.weights[1]:.4f}, c={global_best_predictor.weights[2]:.4f}") 
