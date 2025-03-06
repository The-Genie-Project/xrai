import pytest
import numpy as np
import matplotlib.pyplot as plt

from xrai.visualization import plot_fitness_history, plot_chaotic_function, plot_prediction_vs_actual
from xrai.core import Predictor

@pytest.fixture
def sample_fitness_history():
    """Fixture for sample fitness history."""
    return [
        (0.5, 0.3),  # (predictor_fitness, meta_fitness)
        (0.6, 0.4),
        (0.7, 0.5),
        (0.8, 0.6)
    ]

@pytest.fixture
def sample_predictor():
    """Fixture for sample predictor."""
    return Predictor([0.1, 0.2, 0.3])

def test_plot_fitness_history(sample_fitness_history):
    """Test plotting fitness history."""
    # Plot fitness history
    fig = plot_fitness_history(sample_fitness_history)
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure), "plot_fitness_history should return a matplotlib Figure"
    
    # Check that the figure has the expected elements
    ax = fig.axes[0]
    assert len(ax.lines) == 2, "Plot should have two lines (predictor and meta-predictor fitness)"
    assert ax.get_xlabel() == "Generation", "X-axis label should be 'Generation'"
    assert ax.get_ylabel() == "Fitness", "Y-axis label should be 'Fitness'"
    
    # Clean up
    plt.close(fig)

def test_plot_chaotic_function():
    """Test plotting chaotic function."""
    # Plot chaotic function
    fig = plot_chaotic_function(r=3.8, iterations=50, initial_x=0.4)
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure), "plot_chaotic_function should return a matplotlib Figure"
    
    # Check that the figure has the expected elements
    ax = fig.axes[0]
    assert len(ax.lines) == 1, "Plot should have one line"
    assert ax.get_xlabel() == "Iteration", "X-axis label should be 'Iteration'"
    assert ax.get_ylabel() == "Value", "Y-axis label should be 'Value'"
    
    # Clean up
    plt.close(fig)

def test_plot_prediction_vs_actual(sample_predictor):
    """Test plotting prediction vs actual."""
    # Plot prediction vs actual
    fig = plot_prediction_vs_actual(sample_predictor, r=3.8, samples=50)
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure), "plot_prediction_vs_actual should return a matplotlib Figure"
    
    # Check that the figure has the expected elements
    ax = fig.axes[0]
    assert len(ax.collections) == 2, "Plot should have two scatter collections (actual and predicted)"
    assert ax.get_xlabel() == "Input (x)", "X-axis label should be 'Input (x)'"
    assert ax.get_ylabel() == "Output", "Y-axis label should be 'Output'"
    
    # Clean up
    plt.close(fig) 
