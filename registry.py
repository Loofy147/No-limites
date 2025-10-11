from ega import EpigeneticAlgorithm
from standard_ga import StandardAlgorithm
from dummy_algorithm import DummyAlgorithm
from adaptive_ega import AdaptiveEGA
from fitness_functions import one_max_fitness, deceptive_fitness

# --- Component Registries ---

ALGORITHMS = {}
FITNESS_FUNCTIONS = {}

# --- Registration Functions ---


import re

def register_algorithm(name, cls):
    """Registers an algorithm class with the framework.

    This function makes an algorithm available to the experiment runners
    under a unique identifier name.

    Args:
        name (str): The unique name to identify the algorithm. Must be a
            valid identifier (alphanumeric characters and underscores).
        cls (class): The algorithm class (must inherit from `BaseAlgorithm`).

    Raises:
        ValueError: If the name is not a valid identifier or if an algorithm
            with the same name is already registered.
    """
    if not re.match(r"^[a-zA-Z0-9_]+$", name):
        raise ValueError(
            f"Invalid algorithm name: '{name}'. "
            "Name must be alphanumeric with underscores."
        )
    if name in ALGORITHMS:
        raise ValueError(f"Algorithm '{name}' is already registered.")
    ALGORITHMS[name] = cls


def register_fitness_function(name, func):
    """Registers a fitness function with the framework.

    This function makes a fitness function available to the experiment
    runners under a unique identifier name.

    Args:
        name (str): The unique name to identify the fitness function.
        func (callable): The fitness function.

    Raises:
        ValueError: If a fitness function with the same name is already
            registered.
    """
    if name in FITNESS_FUNCTIONS:
        raise ValueError(f"Fitness function '{name}' is already registered.")
    FITNESS_FUNCTIONS[name] = func


# --- Register Existing Components ---


def register_core_components():
    """Registers all the standard components that ship with the framework.

    This function is called automatically when the module is imported,
    ensuring that all built-in algorithms and fitness functions are
    ready to be used by the runners.
    """
    # Register algorithms
    register_algorithm("ega", EpigeneticAlgorithm)
    register_algorithm("standard", StandardAlgorithm)
    register_algorithm("dummy", DummyAlgorithm)
    register_algorithm("adaptive_ega", AdaptiveEGA)

    # Register fitness functions
    register_fitness_function("onemax", one_max_fitness)
    register_fitness_function("deceptive", deceptive_fitness)


# Register components on import
register_core_components()
