import numpy as np
import matplotlib.pyplot as plt

def plot_fitness_history(fitness_history):
    """
    Plot the evolution of fitness over generations.
    
    Args:
        fitness_history (list): List of tuples (predictor_fitness, meta_fitness)
    """
    generations = np.arange(len(fitness_history))
    predictor_fitness, meta_fitness = zip(*fitness_history)

    plt.figure(figsize=(10, 5))
    plt.plot(generations, predictor_fitness, label="Predictor Fitness")
    plt.plot(generations, meta_fitness, label="Meta-Predictor Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution of Predictors and Meta-Predictors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_chaotic_function(r=3.8, iterations=100, initial_x=0.4):
    """
    Plot the behavior of the chaotic function.
    
    Args:
        r (float): Parameter for the chaotic function
        iterations (int): Number of iterations to plot
        initial_x (float): Initial value
    """
    from genie.core import chaotic_function
    
    x_values = np.zeros(iterations)
    x_values[0] = initial_x
    
    for i in range(1, iterations):
        x_values[i] = chaotic_function(x_values[i-1], r)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(iterations), x_values)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(f"Chaotic Function Behavior (r={r})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_prediction_vs_actual(predictor, r=3.8, samples=100):
    """
    Plot the predictions of a predictor against the actual values.
    
    Args:
        predictor: A trained predictor
        r (float): Parameter for the chaotic function
        samples (int): Number of samples to plot
    """
    from genie.core import chaotic_function
    
    x_values = np.linspace(0, 1, samples)
    actual_values = np.array([chaotic_function(x, r) for x in x_values])
    predicted_values = np.array([predictor.predict(x) for x in x_values])
    
    plt.figure(figsize=(10, 5))
    plt.scatter(x_values, actual_values, label="Actual", alpha=0.7)
    plt.scatter(x_values, predicted_values, label="Predicted", alpha=0.7)
    plt.xlabel("Input (x)")
    plt.ylabel("Output")
    plt.title("Predictor Performance: Predicted vs Actual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf() 
