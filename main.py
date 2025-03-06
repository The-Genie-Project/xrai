#!/usr/bin/env python3
"""
Main script for the  package.

This script provides a command-line interface to run experiments with the  package.
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from core import evolve, chaotic_function, Predictor
from visualization import plot_fitness_history, plot_prediction_vs_actual, plot_chaotic_function
from utils.metrics import calculate_fitness_stats
from utils.io import save_experiment_results
from experiments.basic_experiment import run_basic_experiment
from experiments.parameter_sweep import run_parameter_sweep

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=": Evolutionary Meta-Learning Framework")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Basic experiment command
    basic_parser = subparsers.add_parser("basic", help="Run a basic experiment")
    basic_parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    basic_parser.add_argument("--population", type=int, default=20, help="Population size")
    basic_parser.add_argument("--mutation", type=float, default=0.1, help="Mutation rate")
    basic_parser.add_argument("--r", type=float, default=3.8, help="Chaos parameter")
    basic_parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    # Parameter sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run a parameter sweep experiment")
    sweep_parser.add_argument("--generations", type=str, default="50,100,200", help="Comma-separated list of generation values")
    sweep_parser.add_argument("--population", type=str, default="10,20,50", help="Comma-separated list of population size values")
    sweep_parser.add_argument("--mutation", type=str, default="0.05,0.1,0.2", help="Comma-separated list of mutation rate values")
    sweep_parser.add_argument("--r", type=str, default="3.6,3.8,4.0", help="Comma-separated list of chaos parameter values")
    sweep_parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    # Plot chaotic function command
    plot_parser = subparsers.add_parser("plot", help="Plot the chaotic function")
    plot_parser.add_argument("--r", type=float, default=3.8, help="Chaos parameter")
    plot_parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    plot_parser.add_argument("--initial", type=float, default=0.4, help="Initial value")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    if args.command == "basic":
        # Run basic experiment
        run_basic_experiment(
            num_generations=args.generations,
            population_size=args.population,
            mutation_rate=args.mutation,
            r=args.r,
            save_results=not args.no_save
        )
    
    elif args.command == "sweep":
        # Parse parameter lists
        generations_values = [int(x) for x in args.generations.split(",")]
        population_values = [int(x) for x in args.population.split(",")]
        mutation_values = [float(x) for x in args.mutation.split(",")]
        r_values = [float(x) for x in args.r.split(",")]
        
        # Run parameter sweep
        run_parameter_sweep(
            generations_values=generations_values,
            population_values=population_values,
            mutation_values=mutation_values,
            r_values=r_values,
            save_results=not args.no_save
        )
    
    elif args.command == "plot":
        # Plot chaotic function
        fig = plot_chaotic_function(
            r=args.r,
            iterations=args.iterations,
            initial_x=args.initial
        )
        plt.show()
    
    else:
        # No command specified, print help
        print("Please specify a command. Use --help for more information.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
