import pytest
import numpy as np

from genie.core import chaotic_function, Predictor, MetaPredictor, evolve

def test_chaotic_function():
    """Test the chaotic function."""
    # Test with default parameter
    x = 0.5
    result = chaotic_function(x)
    assert 0 <= result <= 1, "Chaotic function should return a value between 0 and 1"
    
    # Test with custom parameter
    r = 3.6
    result = chaotic_function(x, r)
    assert 0 <= result <= 1, "Chaotic function should return a value between 0 and 1"
    
    # Test with different inputs
    for x in [0.1, 0.3, 0.7, 0.9]:
        result = chaotic_function(x)
        assert 0 <= result <= 1, f"Chaotic function failed for input {x}"

def test_predictor_initialization():
    """Test predictor initialization."""
    # Test with default weights
    predictor = Predictor()
    assert predictor.weights.shape == (3,), "Predictor should have 3 weights"
    
    # Test with custom weights
    weights = [0.1, 0.2, 0.3]
    predictor = Predictor(weights)
    assert np.array_equal(predictor.weights, np.array(weights)), "Predictor weights not set correctly"

def test_predictor_predict():
    """Test predictor prediction."""
    # Create a predictor with known weights
    weights = [0.5, 0.3, 0.2]
    predictor = Predictor(weights)
    
    # Test prediction
    x = 0.5
    expected = np.clip(weights[0] * x**2 + weights[1] * x + weights[2], 0, 1)
    result = predictor.predict(x)
    assert result == expected, "Predictor prediction is incorrect"
    
    # Test prediction is clipped to [0, 1]
    weights = [10, 10, 10]  # These weights should produce values > 1
    predictor = Predictor(weights)
    result = predictor.predict(0.5)
    assert 0 <= result <= 1, "Predictor prediction should be clipped to [0, 1]"

def test_predictor_mutate():
    """Test predictor mutation."""
    # Create a predictor
    predictor = Predictor([0.5, 0.3, 0.2])
    
    # Test mutation with default rate
    mutated = predictor.mutate()
    assert not np.array_equal(predictor.weights, mutated.weights), "Mutation should change weights"
    
    # Test mutation with custom rate
    mutation_rate = 0.01
    mutated = predictor.mutate(mutation_rate)
    # Check that weights are within expected range
    for i in range(3):
        assert abs(predictor.weights[i] - mutated.weights[i]) <= mutation_rate, \
            f"Mutation exceeded rate for weight {i}"

def test_meta_predictor():
    """Test meta-predictor functionality."""
    # Create a meta-predictor
    meta = MetaPredictor([0.5, 0.3, 0.2])
    
    # Create a predictor
    predictor = Predictor([0.1, 0.2, 0.3])
    
    # Test fitness prediction
    x = 0.5
    pred_value = predictor.predict(x)
    expected = np.clip(meta.weights[0] * pred_value**2 + meta.weights[1] * pred_value + meta.weights[2], 0, 1)
    result = meta.predict_fitness(predictor, x)
    assert result == expected, "Meta-predictor fitness prediction is incorrect"
    
    # Test mutation
    mutated = meta.mutate(0.1)
    assert not np.array_equal(meta.weights, mutated.weights), "Mutation should change weights"

def test_evolve():
    """Test the evolutionary algorithm."""
    # Run a small evolution
    num_generations = 5
    population_size = 4
    
    fitness_history, predictors, meta_predictors = evolve(
        num_generations=num_generations,
        population_size=population_size
    )
    
    # Check results
    assert len(fitness_history) == num_generations, "Fitness history length incorrect"
    assert len(predictors) == population_size, "Final predictor population size incorrect"
    assert len(meta_predictors) == population_size, "Final meta-predictor population size incorrect"
    
    # Check fitness history format
    for fitness in fitness_history:
        assert len(fitness) == 2, "Fitness history should contain tuples of (predictor_fitness, meta_fitness)"
        assert 0 <= fitness[0] <= 1, "Predictor fitness should be between 0 and 1"
        assert 0 <= fitness[1] <= 1, "Meta-predictor fitness should be between 0 and 1" 
