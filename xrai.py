"""
Legacy entry point for the  package.

This file is kept for backward compatibility. New code should use the package directly.
"""
import warnings

warnings.warn(
    "This file is deprecated. Please use the  package directly.",
    DeprecationWarning,
    stacklevel=2
)

from core import chaotic_function, Predictor, MetaPredictor, evolve
from visualization import plot_fitness_history

if __name__ == "__main__":
    # Run the evolutionary process
    fitness_history, predictors, meta_predictors = evolve(num_generations=100, population_size=20)
    
    # Plot the evolution of fitness over generations
    import matplotlib.pyplot as plt
    plot_fitness_history(fitness_history)
    plt.show()
