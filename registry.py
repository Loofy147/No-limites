from ega import EpigeneticAlgorithm
from standard_ga import StandardAlgorithm
from dummy_algorithm import DummyAlgorithm
from adaptive_ega import AdaptiveEGA
from fitness_functions import one_max_fitness, deceptive_fitness

# --- Component Registries ---

ALGORITHMS = {}
FITNESS_FUNCTIONS = {}

# --- Registration Functions ---


def register_algorithm(name, cls):
    """Registers an algorithm class with the framework."""
    if name in ALGORITHMS:
        raise ValueError(f"Algorithm '{name}' is already registered.")
    ALGORITHMS[name] = cls


def register_fitness_function(name, func):
    """Registers a fitness function with the framework."""
    if name in FITNESS_FUNCTIONS:
        raise ValueError(f"Fitness function '{name}' is already registered.")
    FITNESS_FUNCTIONS[name] = func


# --- Register Existing Components ---


def register_core_components():
    """
    A convenience function to register all the standard components
    that ship with the framework.
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
