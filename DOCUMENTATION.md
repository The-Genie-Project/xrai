# Genie Documentation

## Overview

Genie is a Python package that simulates an evolutionary algorithm for meta-learning in chaotic environments. It evolves two types of agents:

1. **Predictors**: Agents that try to predict the output of a chaotic function (logistic map)
2. **Meta-predictors**: Agents that try to predict how well the predictors will perform

This framework demonstrates concepts of meta-learning, evolutionary algorithms, and prediction in chaotic systems.

## Installation

```bash
# Clone the repository
git clone https://github.com/genie-project/genie.git
cd genie

# Install the package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Core Concepts

### Chaotic Function

The chaotic function used in this package is the logistic map, defined as:

```
x_{n+1} = r * x_n * (1 - x_n)
```

where `r` is a parameter that controls the behavior of the system. When `r > 3.57`, the system exhibits chaotic behavior, making it difficult to predict.

### Predictors

Predictors are agents that try to predict the output of the chaotic function. Each predictor has a simple quadratic model with three weights:

```
prediction = w_0 * x^2 + w_1 * x + w_2
```

where `w_0`, `w_1`, and `w_2` are the weights of the model, and `x` is the input.

### Meta-predictors

Meta-predictors are agents that try to predict how well a predictor will perform for a given input. They also use a quadratic model, but they take the predictor's output as input:

```
fitness_prediction = w_0 * p^2 + w_1 * p + w_2
```

where `w_0`, `w_1`, and `w_2` are the weights of the meta-predictor, and `p` is the predictor's output.

### Evolutionary Algorithm

The evolutionary algorithm works as follows:

1. Initialize a population of predictors and meta-predictors with random weights
2. For each generation:
   - Generate a random input `x`
   - Calculate the true output of the chaotic function
   - Evaluate the fitness of each predictor (how close its prediction is to the true output)
   - Evaluate the fitness of each meta-predictor (how well it predicts the predictor's fitness)
   - Select the top 50% of predictors and meta-predictors
   - Generate new offspring by mutating the selected agents
3. Return the fitness history and the final populations

## API Reference

### Core Module

#### `chaotic_function(x, r=3.8)`

Logistic map function, a simple chaotic system.

**Parameters:**

- `x` (float): Input value between 0 and 1
- `r` (float): Parameter that controls the behavior of the system (3.8 is chaotic)

**Returns:**

- float: The next value in the chaotic sequence

#### `class Predictor`

A predictor agent that tries to predict the output of the chaotic function.

**Methods:**

- `__init__(weights=None)`: Initialize a predictor with random or specified weights
- `predict(x)`: Predict the output of the chaotic function
- `mutate(mutation_rate=0.1)`: Create a mutated copy of this predictor

#### `class MetaPredictor`

A meta-predictor agent that predicts how well a predictor will perform.

**Methods:**

- `__init__(weights=None)`: Initialize a meta-predictor with random or specified weights
- `predict_fitness(predictor, x)`: Predict the fitness of a predictor for a given input
- `mutate(mutation_rate=0.1)`: Create a mutated copy of this meta-predictor

#### `evolve(num_generations=100, population_size=20, mutation_rate=0.1, r=3.8)`

Run the evolutionary algorithm to evolve predictors and meta-predictors.

**Parameters:**

- `num_generations` (int): Number of generations to evolve
- `population_size` (int): Size of the population
- `mutation_rate` (float): Rate of mutation for offspring
- `r` (float): Parameter for the chaotic function

**Returns:**

- tuple: (fitness_history, predictors, meta_predictors)

### Visualization Module

#### `plot_fitness_history(fitness_history)`

Plot the evolution of fitness over generations.

**Parameters:**

- `fitness_history` (list): List of tuples (predictor_fitness, meta_fitness)

**Returns:**

- matplotlib.figure.Figure: The generated figure

#### `plot_chaotic_function(r=3.8, iterations=100, initial_x=0.4)`

Plot the behavior of the chaotic function.

**Parameters:**

- `r` (float): Parameter for the chaotic function
- `iterations` (int): Number of iterations to plot
- `initial_x` (float): Initial value

**Returns:**

- matplotlib.figure.Figure: The generated figure

#### `plot_prediction_vs_actual(predictor, r=3.8, samples=100)`

Plot the predictions of a predictor against the actual values.

**Parameters:**

- `predictor` (Predictor): A trained predictor
- `r` (float): Parameter for the chaotic function
- `samples` (int): Number of samples to plot

**Returns:**

- matplotlib.figure.Figure: The generated figure

### Utilities

#### Metrics

- `mean_absolute_error(y_true, y_pred)`: Calculate the mean absolute error
- `mean_squared_error(y_true, y_pred)`: Calculate the mean squared error
- `calculate_fitness_stats(fitness_history)`: Calculate statistics from fitness history

#### I/O

- `save_model(model, filepath)`: Save a model to a file
- `load_model(filepath)`: Load a model from a file
- `save_experiment_results(fitness_history, best_predictor, best_meta_predictor, filepath)`: Save experiment results
- `load_experiment_results(filepath)`: Load experiment results

### Experiments

#### Basic Experiment

`run_basic_experiment(num_generations=100, population_size=20, mutation_rate=0.1, r=3.8, save_results=True)`

Run a basic experiment with the evolutionary algorithm.

**Parameters:**

- `num_generations` (int): Number of generations to evolve
- `population_size` (int): Size of the population
- `mutation_rate` (float): Rate of mutation for offspring
- `r` (float): Parameter for the chaotic function
- `save_results` (bool): Whether to save the results

**Returns:**

- tuple: (fitness_history, best_predictor, best_meta_predictor)

#### Parameter Sweep

`run_parameter_sweep(generations_values=[50, 100, 200], population_values=[10, 20, 50], mutation_values=[0.05, 0.1, 0.2], r_values=[3.6, 3.8, 4.0], save_results=True)`

Run a parameter sweep experiment with the evolutionary algorithm.

**Parameters:**

- `generations_values` (list): Values for number of generations
- `population_values` (list): Values for population size
- `mutation_values` (list): Values for mutation rate
- `r_values` (list): Values for chaotic function parameter
- `save_results` (bool): Whether to save the results

**Returns:**

- dict: Results of the parameter sweep

## Command-line Interface

The package provides a command-line interface through the `main.py` script:

```bash
# Run a basic experiment
python main.py basic --generations 100 --population 20 --mutation 0.1 --r 3.8

# Run a parameter sweep
python main.py sweep --generations 50,100 --population 10,20 --mutation 0.05,0.1 --r 3.6,3.8

# Plot the chaotic function
python main.py plot --r 3.8 --iterations 100 --initial 0.4
```

## Examples

See the `examples` directory for example scripts and the `notebooks` directory for Jupyter notebooks with detailed examples and analyses.

## Testing

Run the tests with pytest:

```bash
pytest
```

## License

MIT
