import numpy as np

def calculate_fitness_stats(fitness_history):
    """
    Calculate statistics from fitness history.
    
    Args:
        fitness_history (list): List of tuples (predictor_fitness, meta_fitness)
    
    Returns:
        dict: Dictionary with fitness statistics
    """
    predictor_fitness, meta_fitness = zip(*fitness_history)
    
    stats = {
        "predictor_final": predictor_fitness[-1],
        "meta_final": meta_fitness[-1],
        "predictor_mean": np.mean(predictor_fitness),
        "meta_mean": np.mean(meta_fitness),
        "predictor_max": np.max(predictor_fitness),
        "meta_max": np.max(meta_fitness),
        "predictor_improvement": predictor_fitness[-1] - predictor_fitness[0],
        "meta_improvement": meta_fitness[-1] - meta_fitness[0]
    }
    
    return stats 
