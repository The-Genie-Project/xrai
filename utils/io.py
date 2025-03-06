import numpy as np
import pickle
import os

def save_model(model, filepath):
    """
    Save a model to a file.
    
    Args:
        model: The model to save (Predictor or MetaPredictor)
        filepath (str): Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """
    Load a model from a file.
    
    Args:
        filepath (str): Path to the model file
    
    Returns:
        The loaded model
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_experiment_results(fitness_history, best_predictor, best_meta_predictor, filepath):
    """
    Save experiment results to a file.
    
    Args:
        fitness_history (list): List of fitness values
        best_predictor: The best predictor model
        best_meta_predictor: The best meta-predictor model
        filepath (str): Path to save the results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    results = {
        'fitness_history': fitness_history,
        'best_predictor': best_predictor,
        'best_meta_predictor': best_meta_predictor
    }
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

def load_experiment_results(filepath):
    """
    Load experiment results from a file.
    
    Args:
        filepath (str): Path to the results file
    
    Returns:
        dict: Dictionary with experiment results
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f) 
