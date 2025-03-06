import numpy as np
import random

# Define a chaotic function (logistic map) as the environment
def chaotic_function(x, r=3.8):
    """
    Logistic map function, a simple chaotic system.
    
    Args:
        x (float): Input value between 0 and 1
        r (float): Parameter that controls the behavior of the system (3.8 is chaotic)
    
    Returns:
        float: The next value in the chaotic sequence
    """
    return r * x * (1 - x)

# Define an individual predictor (Agent)
class Predictor:
    """
    A predictor agent that tries to predict the output of the chaotic function.
    
    Attributes:
        weights (numpy.ndarray): Weights for the quadratic prediction model
    """
    def __init__(self, weights=None):
        """
        Initialize a predictor with random or specified weights.
        
        Args:
            weights (list or numpy.ndarray, optional): Weights for the prediction model
        """
        if weights is None:
            self.weights = np.random.uniform(-1, 1, 3)  # Simple 3-weight quadratic model
        else:
            self.weights = np.array(weights)

    def predict(self, x):
        """
        Predict the output of the chaotic function.
        
        Args:
            x (float): Input value
        
        Returns:
            float: Predicted output, clipped to [0, 1]
        """
        return np.clip(self.weights[0] * x**2 + self.weights[1] * x + self.weights[2], 0, 1)

    def mutate(self, mutation_rate=0.1):
        """
        Create a mutated copy of this predictor.
        
        Args:
            mutation_rate (float): The magnitude of mutation
        
        Returns:
            Predictor: A new predictor with mutated weights
        """
        return Predictor(self.weights + np.random.uniform(-mutation_rate, mutation_rate, 3))

# Define a meta-predictor (predicts which agent will perform well)
class MetaPredictor:
    """
    A meta-predictor agent that predicts how well a predictor will perform.
    
    Attributes:
        weights (numpy.ndarray): Weights for the quadratic meta-prediction model
    """
    def __init__(self, weights=None):
        """
        Initialize a meta-predictor with random or specified weights.
        
        Args:
            weights (list or numpy.ndarray, optional): Weights for the meta-prediction model
        """
        if weights is None:
            self.weights = np.random.uniform(-1, 1, 3)  # Another 3-weight quadratic model
        else:
            self.weights = np.array(weights)

    def predict_fitness(self, predictor, x):
        """
        Predict the fitness of a predictor for a given input.
        
        Args:
            predictor (Predictor): The predictor to evaluate
            x (float): Input value
        
        Returns:
            float: Predicted fitness, clipped to [0, 1]
        """
        pred_value = predictor.predict(x)
        return np.clip(self.weights[0] * pred_value**2 + self.weights[1] * pred_value + self.weights[2], 0, 1)

    def mutate(self, mutation_rate=0.1):
        """
        Create a mutated copy of this meta-predictor.
        
        Args:
            mutation_rate (float): The magnitude of mutation
        
        Returns:
            MetaPredictor: A new meta-predictor with mutated weights
        """
        return MetaPredictor(self.weights + np.random.uniform(-mutation_rate, mutation_rate, 3))

# Evolutionary Algorithm
def evolve(num_generations=100, population_size=20, mutation_rate=0.1, r=3.8):
    """
    Run the evolutionary algorithm to evolve predictors and meta-predictors.
    
    Args:
        num_generations (int): Number of generations to evolve
        population_size (int): Size of the population
        mutation_rate (float): Rate of mutation for offspring
        r (float): Parameter for the chaotic function
    
    Returns:
        list: History of fitness values for predictors and meta-predictors
    """
    # Initialize population of predictors and meta-predictors
    predictors = [Predictor() for _ in range(population_size)]
    meta_predictors = [MetaPredictor() for _ in range(population_size)]

    fitness_history = []

    for generation in range(num_generations):
        x = np.random.uniform(0, 1)  # Random initial state
        true_value = chaotic_function(x, r)  # The actual outcome

        # Evaluate predictor fitness
        predictor_fitness = np.array([1 - abs(p.predict(x) - true_value) for p in predictors])

        # Evaluate meta-predictor fitness (how well they predict predictor fitness)
        meta_fitness = np.array([1 - abs(m.predict_fitness(p, x) - f) for m, p, f in zip(meta_predictors, predictors, predictor_fitness)])

        # Selection: Keep the top 50% of predictors and meta-predictors
        top_indices = np.argsort(predictor_fitness)[-population_size//2:]
        predictors = [predictors[i] for i in top_indices]
        meta_predictors = [meta_predictors[i] for i in top_indices]

        # Reproduce: Generate new mutated offspring
        predictors += [p.mutate(mutation_rate) for p in predictors]
        meta_predictors += [m.mutate(mutation_rate) for m in meta_predictors]

        # Store best fitness
        fitness_history.append((np.max(predictor_fitness), np.max(meta_fitness)))

    return fitness_history, predictors, meta_predictors 
