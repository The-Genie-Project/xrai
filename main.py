#!/usr/bin/env python3
"""
Live streaming evolution script for the XRAI package.

This script serves as the entry point for running the evolutionary algorithm,
parsing command line arguments and initiating the evolution process.
"""
import sys
import signal

from evolve import live_evolve

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Evolution of Predictors and Meta-Predictors")
    parser.add_argument("--population", type=int, default=20, help="Population size, default 20")
    parser.add_argument("--mutation", type=float, default=0.1, help="Mutation rate, default 0.1")
    parser.add_argument("--r", type=float, default=3.8, help="Chaos parameter, default 3.8")
    parser.add_argument("--update", type=int, default=10, help="Update interval (generations), default 10")
    parser.add_argument("--max-gen", type=int, default=10000, help="Maximum generations, default 10000")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval (generations), default 100")
    parser.add_argument("--continue-from", type=str, help="Path to checkpoint file to continue from")
    parser.add_argument("--hierarchy-levels", type=int, default=20, help="Number of hierarchical refinement levels, default 20")
    parser.add_argument("--level-scaling", type=float, default=0.5, help="Scaling factor for generations at each level, default 0.5")
    
    args = parser.parse_args()
    
    try:
        live_evolve(
            population_size=args.population,
            initial_mutation_rate=args.mutation,
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
