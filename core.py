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
        mutation_rate (float): Self-adaptive mutation rate for this predictor
    """
    def __init__(self, weights=None, mutation_rate=0.1):
        """
        Initialize a predictor with random or specified weights.
        
        Args:
            weights (list or numpy.ndarray, optional): Weights for the prediction model
            mutation_rate (float, optional): Initial mutation rate for this predictor
        """
        if weights is None:
            self.weights = np.random.uniform(-1, 1, 3)  # Simple 3-weight quadratic model
        else:
            self.weights = np.array(weights)
        
        self.mutation_rate = mutation_rate

    def predict(self, x):
        """
        Predict the output of the chaotic function.
        
        Args:
            x (float): Input value
        
        Returns:
            float: Predicted output, clipped to [0, 1]
        """
        return np.clip(self.weights[0] * x**2 + self.weights[1] * x + self.weights[2], 0, 1)

    def mutate(self, mutation_rate=None):
        """
        Create a mutated copy of this predictor.
        
        Args:
            mutation_rate (float, optional): The magnitude of mutation.
                                            If None, uses the predictor's own mutation_rate.
        
        Returns:
            Predictor: A new predictor with mutated weights and possibly mutated mutation rate
        """
        # Use provided mutation_rate if given, otherwise use self.mutation_rate
        mr = mutation_rate if mutation_rate is not None else self.mutation_rate
        
        # Create new weights with mutation
        new_weights = self.weights + np.random.uniform(-mr, mr, 3)
        
        # Create a new predictor with the mutated weights
        # If no external mutation_rate was provided, also mutate the mutation_rate itself
        if mutation_rate is None:
            # Mutate the mutation rate itself (meta-mutation)
            # Use log-normal distribution to ensure mutation_rate stays positive
            # and changes are symmetric in log-space
            new_mutation_rate = self.mutation_rate * np.exp(np.random.normal(0, 0.2))
            
            # Constrain mutation rate to reasonable bounds
            new_mutation_rate = np.clip(new_mutation_rate, 0.001, 0.5)
            
            return Predictor(new_weights, new_mutation_rate)
        else:
            # If external mutation_rate was provided, keep using that
            return Predictor(new_weights, mr)

# Define a meta-predictor (predicts which agent will perform well)
class MetaPredictor:
    """
    A meta-predictor agent that predicts how well a predictor will perform.
    
    Attributes:
        weights (numpy.ndarray): Weights for the quadratic meta-prediction model
        mutation_rate (float): Self-adaptive mutation rate for this meta-predictor
    """
    def __init__(self, weights=None, mutation_rate=0.1):
        """
        Initialize a meta-predictor with random or specified weights.
        
        Args:
            weights (list or numpy.ndarray, optional): Weights for the meta-prediction model
            mutation_rate (float, optional): Initial mutation rate for this meta-predictor
        """
        if weights is None:
            self.weights = np.random.uniform(-1, 1, 3)  # Another 3-weight quadratic model
        else:
            self.weights = np.array(weights)
        
        self.mutation_rate = mutation_rate

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

    def mutate(self, mutation_rate=None):
        """
        Create a mutated copy of this meta-predictor.
        
        Args:
            mutation_rate (float, optional): The magnitude of mutation.
                                            If None, uses the meta-predictor's own mutation_rate.
        
        Returns:
            MetaPredictor: A new meta-predictor with mutated weights and possibly mutated mutation rate
        """
        # Use provided mutation_rate if given, otherwise use self.mutation_rate
        mr = mutation_rate if mutation_rate is not None else self.mutation_rate
        
        # Create new weights with mutation
        new_weights = self.weights + np.random.uniform(-mr, mr, 3)
        
        # Create a new meta-predictor with the mutated weights
        # If no external mutation_rate was provided, also mutate the mutation_rate itself
        if mutation_rate is None:
            # Mutate the mutation rate itself (meta-mutation)
            # Use log-normal distribution to ensure mutation_rate stays positive
            # and changes are symmetric in log-space
            new_mutation_rate = self.mutation_rate * np.exp(np.random.normal(0, 0.2))
            
            # Constrain mutation rate to reasonable bounds
            new_mutation_rate = np.clip(new_mutation_rate, 0.001, 0.5)
            
            return MetaPredictor(new_weights, new_mutation_rate)
        else:
            # If external mutation_rate was provided, keep using that
            return MetaPredictor(new_weights, mr)
