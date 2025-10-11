# Epigenetic Genetic Algorithm (EGA) Framework

This project provides a modular and extensible framework for developing and analyzing evolutionary algorithms, centered around a novel **Epigenetic Genetic Algorithm (EGA)**. It is designed for researchers and developers to experiment with advanced genetic algorithms, compare their performance, and extend the framework with new components.

## Key Features

- **Modular Architecture:** Built on abstract base classes and a component registry, allowing for easy extension.
- **Novel Algorithms:** Includes a unique Epigenetic Genetic Algorithm (EGA) and a self-adapting version (`AdaptiveEGA`).
- **Comparative Analysis:** A robust experiment runner (`run_experiment.py`) for conducting multiple trials, aggregating results, and performing comparative analysis.
- **Parallel Execution:** Built-in support for running experimental trials in parallel to accelerate research.
- **Checkpointing & Resuming:** A checkpointing system to save and resume long-running experiments, preventing loss of progress.
- **Configuration Driven:** Experiments can be defined in YAML files for reproducibility and easy modification.
- **Automated Quality Checks:** Integrated with GitHub Actions for continuous integration (CI) to ensure code quality using `black`, `flake8`, and `unittest`.

## Security and Hardening

The framework has been hardened against common vulnerabilities to ensure robust and secure operation, particularly when running with custom or untrusted configurations.

-   **Secure Checkpointing:** The checkpointing system uses **JSON** instead of `pickle`. This prevents arbitrary code execution vulnerabilities when loading checkpoint files from untrusted sources.
-   **Configuration Validation:** Before an experiment begins, a validation step (`validate_config`) checks for potentially harmful or invalid parameters. This mitigates resource exhaustion attacks, such as specifying an excessively large population size or negative values.
-   **Checkpoint Integrity:** Checkpoints are validated upon loading to ensure they have the correct data structure and types, preventing crashes or undefined behavior from corrupted files.
-   **Hardened Component Registry:** The component registry validates the names of algorithms and fitness functions to prevent injection of malicious or malformed strings.

## The Epigenetic Concept

Traditional Genetic Algorithms (GAs) can suffer from premature convergence, where a population loses the genetic diversity needed to find a global optimum. The EGA introduces a dual-layered genetic system to address this:

1.  **Genotype:** The core, slow-mutating genetic code that preserves strong solutions.
2.  **Epigenome:** A rapidly-mutating layer that controls which genes in the genotype are *expressed*.

This separation allows for a powerful balance between **exploitation** (preserving good genotypes) and **exploration** (testing new gene combinations via the epigenome).

## Installation

To set up the project, clone the repository and install the required dependencies. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (for running checks)
pip install -r requirements-dev.txt
```

## How to Run Experiments

There are two primary ways to run experiments:

### 1. Single Run (`main.py`)

The `main.py` script is ideal for quick, single-run experiments.

**Default Run (EGA on One-Max):**
```bash
python3 main.py
```

**Custom Run (Standard GA on Deceptive Problem):**
```bash
python3 main.py --algorithm standard --fitness_func deceptive --generations 200
```
For a full list of configurable parameters, run `python3 main.py --help`.

### 2. Batch Experiments (`run_experiment.py`)

For systematic and reproducible research, `run_experiment.py` is the primary tool. It runs multiple trials, aggregates data, and supports advanced features.

**Configuration:**
Experiments are defined in a YAML file. See `example_config.yaml` for a template.

**Standard Execution:**
```bash
python3 run_experiment.py --config example_config.yaml
```

**High-Performance Parallel Execution:**
To run trials in parallel across 4 CPU cores:
```bash
python3 run_experiment.py --config example_config.yaml --parallel 4
```

**Checkpointing and Resuming:**
For long experiments, you can save progress and resume if interrupted.

```bash
# Run a long experiment, saving a checkpoint every 50 generations
python3 run_experiment.py --config example_config.yaml --generations 500 --checkpoint_interval 50

# If interrupted, resume from the last checkpoint
python3 run_experiment.py --config example_config.yaml --generations 500 --resume
```

## Output

The framework produces two primary types of output:

1.  **Visual Plots (`.png`):** Both runners generate plots of fitness over generations.
2.  **Structured Data (`.json`):** `run_experiment.py` saves a complete record of the experiment, including the full configuration and aggregated results, to a JSON file (e.g., `experiment_results.json`).

## Framework Architecture

The framework is designed for extensibility:

1.  **Abstract Base Classes:** `framework.py` defines `BaseAlgorithm`, an abstract class that enforces a consistent interface for all algorithms.
2.  **Component Registry:** `registry.py` acts as a centralized discovery service. Algorithms and fitness functions are registered with a unique name.
3.  **Dynamic Loading:** The runners (`main.py`, `run_experiment.py`) are data-driven. They use the registry to dynamically load components by name, making them automatically compatible with any new, registered component.

## Extending the Framework

Adding new components is straightforward and does not require modifying the runners.

### How to Add a New Algorithm

1.  **Create the Algorithm File:** Create a new Python file (e.g., `my_algorithm.py`).
2.  **Implement the Algorithm Class:**
    - Create a class that inherits from `BaseAlgorithm` (from `framework.py`).
    - Implement all abstract methods: `__init__`, `evolve`, `get_fittest_individual`, `get_state`, and `set_state`.
    - The `__init__` constructor must accept `**kwargs` to be compatible with the argument parser.
    - Optionally, implement `adapt_parameters` for self-adapting logic.
    *(See `dummy_algorithm.py` for a minimal example.)*
3.  **Register the Algorithm:**
    - Open `registry.py`.
    - Import your new class (e.g., `from my_algorithm import MyAlgorithm`).
    - Add a registration line inside `register_core_components()`:
      ```python
      register_algorithm('my_algo', MyAlgorithm)
      ```

### How to Add a New Fitness Function

1.  **Define the Function:** Add your new fitness function to `fitness_functions.py`. It should accept a list of bits and return a number.
2.  **Register the Function:**
    - Open `registry.py`.
    - Import your new function.
    - Add a registration line:
      ```python
      register_fitness_function('my_func', my_fitness_function)
      ```

Your new components are now available to the runners (e.g., via `--algorithm my_algo`).

## Contributing

Contributions are welcome. To ensure code quality and consistency, please follow these guidelines:

1.  **Code Style:** This project uses `black` for formatting and `flake8` for linting. Run `black .` and `flake8` before committing.
2.  **Automated Checks (CI):** All pull requests are automatically checked by a GitHub Actions workflow. Your contribution must pass all checks (formatting, linting, and unit tests).
3.  **Architectural Patterns:** When adding new components, please adhere to the existing architecture (inherit from base classes, register components).
