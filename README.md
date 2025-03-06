# : Evolutionary Meta-Learning Framework

 is a Python package that simulates an evolutionary algorithm for meta-learning in chaotic environments. It evolves two types of agents:

1. **Predictors**: Agents that try to predict the output of a chaotic function (logistic map)
2. **Meta-predictors**: Agents that try to predict how well the predictors will perform

This framework demonstrates concepts of meta-learning, evolutionary algorithms, and prediction in chaotic systems.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from .core import evolve
from .visualization import plot_fitness_history

# Run the evolutionary process
fitness_history = evolve(num_generations=100, population_size=20)

# Visualize the results
plot_fitness_history(fitness_history)
```

### Running Experiments

```bash
# Run a basic experiment
python -m .experiments.basic_experiment

# Run a parameter sweep experiment
python -m .experiments.parameter_sweep
```

## Project Structure

- `/`: Main package directory
  - `core.py`: Core classes and evolutionary algorithm
  - `visualization.py`: Visualization utilities
  - `experiments/`: Experiment scripts
  - `utils/`: Utility functions
- `tests/`: Unit tests
- `examples/`: Example scripts
- `notebooks/`: Jupyter notebooks with examples and analyses

## License

MIT
