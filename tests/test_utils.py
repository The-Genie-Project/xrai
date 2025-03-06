import pytest
import numpy as np
import os
import tempfile

from genie.utils.metrics import mean_absolute_error, mean_squared_error, calculate_fitness_stats
from genie.utils.io import save_model, load_model, save_experiment_results, load_experiment_results
from genie.core import Predictor, MetaPredictor

def test_mean_absolute_error():
    """Test mean absolute error calculation."""
    y_true = [0.1, 0.2, 0.3, 0.4]
    y_pred = [0.2, 0.3, 0.2, 0.3]
    
    # Calculate expected result
    expected = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    # Test function
    result = mean_absolute_error(y_true, y_pred)
    assert result == expected, "Mean absolute error calculation is incorrect"

def test_mean_squared_error():
    """Test mean squared error calculation."""
    y_true = [0.1, 0.2, 0.3, 0.4]
    y_pred = [0.2, 0.3, 0.2, 0.3]
    
    # Calculate expected result
    expected = np.mean((np.array(y_true) - np.array(y_pred))**2)
    
    # Test function
    result = mean_squared_error(y_true, y_pred)
    assert result == expected, "Mean squared error calculation is incorrect"

def test_calculate_fitness_stats():
    """Test fitness statistics calculation."""
    # Create sample fitness history
    fitness_history = [
        (0.5, 0.3),  # (predictor_fitness, meta_fitness)
        (0.6, 0.4),
        (0.7, 0.5),
        (0.8, 0.6)
    ]
    
    # Calculate stats
    stats = calculate_fitness_stats(fitness_history)
    
    # Check stats
    assert stats["predictor_final"] == 0.8, "Final predictor fitness is incorrect"
    assert stats["meta_final"] == 0.6, "Final meta-predictor fitness is incorrect"
    assert stats["predictor_mean"] == 0.65, "Mean predictor fitness is incorrect"
    assert stats["meta_mean"] == 0.45, "Mean meta-predictor fitness is incorrect"
    assert stats["predictor_max"] == 0.8, "Max predictor fitness is incorrect"
    assert stats["meta_max"] == 0.6, "Max meta-predictor fitness is incorrect"
    assert stats["predictor_improvement"] == 0.3, "Predictor improvement is incorrect"
    assert stats["meta_improvement"] == 0.3, "Meta-predictor improvement is incorrect"

def test_save_load_model():
    """Test saving and loading models."""
    # Create a model
    predictor = Predictor([0.1, 0.2, 0.3])
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filepath = tmp.name
    
    try:
        # Save the model
        save_model(predictor, filepath)
        
        # Check that the file exists
        assert os.path.exists(filepath), "Model file was not created"
        
        # Load the model
        loaded_predictor = load_model(filepath)
        
        # Check that the loaded model is correct
        assert np.array_equal(predictor.weights, loaded_predictor.weights), "Loaded model weights are incorrect"
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)

def test_save_load_experiment_results():
    """Test saving and loading experiment results."""
    # Create sample data
    fitness_history = [(0.5, 0.3), (0.6, 0.4), (0.7, 0.5)]
    best_predictor = Predictor([0.1, 0.2, 0.3])
    best_meta_predictor = MetaPredictor([0.4, 0.5, 0.6])
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filepath = tmp.name
    
    try:
        # Save the results
        save_experiment_results(fitness_history, best_predictor, best_meta_predictor, filepath)
        
        # Check that the file exists
        assert os.path.exists(filepath), "Results file was not created"
        
        # Load the results
        results = load_experiment_results(filepath)
        
        # Check that the loaded results are correct
        assert results["fitness_history"] == fitness_history, "Loaded fitness history is incorrect"
        assert np.array_equal(results["best_predictor"].weights, best_predictor.weights), \
            "Loaded best predictor weights are incorrect"
        assert np.array_equal(results["best_meta_predictor"].weights, best_meta_predictor.weights), \
            "Loaded best meta-predictor weights are incorrect"
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath) 
