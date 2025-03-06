"""
Simple example of using the Genie package.
"""
import numpy as np
import matplotlib.pyplot as plt

from genie.core import evolve, chaotic_function, Predictor
from genie.visualization import plot_fitness_history, plot_prediction_vs_actual, plot_chaotic_function

# Run a simple evolution
print("Running evolution...")
fitness_history, predictors, meta_predictors = evolve(
    num_generations=50,
    population_size=20,
    mutation_rate=0.1,
    r=3.8
)

# Get the best predictor
predictor_fitness = [f[0] for f in fitness_history]
best_predictor_idx = np.argmax(predictor_fitness)
best_predictor = predictors[best_predictor_idx % len(predictors)]

print(f"Best predictor fitness: {max(predictor_fitness):.4f}")
print(f"Best predictor weights: {best_predictor.weights}")

# Plot the chaotic function
print("\nPlotting chaotic function behavior...")
plot_chaotic_function(r=3.8, iterations=100)

# Plot the fitness history
print("Plotting fitness history...")
plot_fitness_history(fitness_history)

# Plot the prediction vs actual
print("Plotting prediction vs actual...")
plot_prediction_vs_actual(best_predictor, r=3.8)

# Show all plots
plt.show()

# Demonstrate prediction
print("\nDemonstrating prediction:")
for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
    true_value = chaotic_function(x, r=3.8)
    predicted = best_predictor.predict(x)
    error = abs(true_value - predicted)
    print(f"Input: {x:.1f}, True: {true_value:.4f}, Predicted: {predicted:.4f}, Error: {error:.4f}") 
