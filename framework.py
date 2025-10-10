from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """An abstract base class for all genetic algorithms in the framework.

    This class defines the required interface that all concrete algorithm
    implementations must follow, ensuring a consistent structure for
    initialization, evolution, and state management. Any new algorithm
    added to the framework must inherit from this class and implement
    all its abstract methods.

    Attributes:
        None
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the algorithm.

        This constructor should accept a flexible set of keyword arguments
        to accommodate different hyperparameter needs across various
        algorithms.

        Args:
            **kwargs: A dictionary of hyperparameters for the algorithm.
        """
        pass

    @abstractmethod
    def evolve(self, fitness_function):
        """Performs one full cycle of evolution for the algorithm's population.

        This method contains the core logic of the genetic algorithm,
        including selection, crossover, mutation, and evaluation.

        Args:
            fitness_function (callable): A function that takes an individual's
                phenotype (e.g., a list of bits) and returns a numerical
                fitness score.

        Returns:
            tuple: A tuple containing two floats:
            - The best fitness score in the new generation.
            - The average fitness score of the new generation.
        """
        pass

    @abstractmethod
    def get_fittest_individual(self):
        """Returns the best individual found by the algorithm so far.

        Returns:
            object: The individual object with the highest fitness score
            in the current population. The exact type of the object will
            depend on the concrete implementation (e.g., `Individual` or
            `StandardIndividual`).
        """
        pass

    @abstractmethod
    def get_state(self):
        """Serializes the current state of the algorithm.

        This method is used for checkpointing, allowing a long-running
        experiment to be saved and resumed later. The state should include
        the entire population and the state of the random number generators.

        Returns:
            dict: A dictionary representing the current state of the
            algorithm.
        """
        pass

    @abstractmethod
    def set_state(self, state):
        """Restores the state of the algorithm from a checkpoint.

        Args:
            state (dict): A dictionary, as returned by `get_state()`,
                representing the state to restore.
        """
        pass

    def adapt_parameters(self):
        """An optional hook for self-adapting algorithms.

        This method is called once per generation before the main `evolve`
        logic. It provides a structured way for algorithms to dynamically
        adjust their own hyperparameters (e.g., mutation rate) based on
        the state of the search. Implementations that do not need this
        functionality can simply leave it empty.
        """
        pass
