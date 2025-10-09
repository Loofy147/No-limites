from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """
    An abstract base class for all genetic algorithms in the framework.

    This class defines the required interface that all concrete algorithm
    implementations must follow, ensuring a consistent structure for
    initialization and evolution.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        The constructor for an algorithm. It should accept a flexible set of
        keyword arguments to accommodate different hyperparameter needs.
        """
        pass

    @abstractmethod
    def evolve(self, fitness_function):
        """
        Performs one full cycle of evolution for the algorithm's population.

        Args:
            fitness_function (callable): The function to evaluate the fitness
                                         of an individual's chromosome/phenotype.

        Returns:
            tuple: A tuple containing the best fitness and the average fitness
                   of the new generation.
        """
        pass

    @abstractmethod
    def get_fittest_individual(self):
        """
        Returns the best individual found by the algorithm.

        Returns:
            object: The fittest individual object.
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Returns a dictionary representing the current state of the algorithm.
        This is used for checkpointing.
        """
        pass

    @abstractmethod
    def set_state(self, state):
        """
        Restores the state of the algorithm from a state dictionary.
        This is used for resuming from a checkpoint.
        """
        pass

    def adapt_parameters(self):
        """
        An optional method for algorithms to dynamically adapt their own
        parameters during a run. Called once per generation.
        """
        pass
