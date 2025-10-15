from framework import BaseAlgorithm


class DummyAlgorithm(BaseAlgorithm):
    """A simple 'dummy' algorithm for demonstration and testing purposes.

    This algorithm does not perform any actual evolution. It serves as a minimal
    implementation of the `BaseAlgorithm` interface, demonstrating how a new
    algorithm can be created and registered with the framework. It simulates
    progress by returning fitness values that increase with each generation.
    """

    def __init__(self, **kwargs):
        """Initializes the dummy algorithm.

        This constructor intentionally ignores all parameters to maintain
        compatibility with the framework's dynamic argument passing.

        Args:
            **kwargs: A dictionary of keyword arguments, which are ignored.
        """
        print("Dummy Algorithm Initialized!")
        self.generation = 0

    def evolve(self, fitness_function):
        """Pretends to evolve and returns fixed, incrementing fitness values.

        This method simulates evolutionary progress by returning a 'best'
        and 'average' fitness that are simple multiples of the current
        generation count.

        Args:
            fitness_function (callable): The fitness function, which is ignored
                by this dummy implementation.

        Returns:
            tuple[float, float]: A tuple containing two floats:
                - The simulated best fitness for the current generation.
                - The simulated average fitness for the current generation.
        """
        self.generation += 1

        # Simulate some progress
        best_fitness = self.generation * 1.5
        avg_fitness = self.generation * 1.2

        return best_fitness, avg_fitness

    def get_fittest_individual(self):
        """Returns a placeholder string, as no real individuals exist.

        Returns:
            str: A message indicating that this is a dummy algorithm.
        """
        return "No individual, this is a dummy algorithm."

    def get_state(self):
        """Serializes the current state of the dummy algorithm.

        Returns:
            dict: A dictionary containing the current generation number.
        """
        return {"generation": self.generation}

    def set_state(self, state):
        """Restores the state of the dummy algorithm from a dictionary.

        Args:
            state (dict): A dictionary containing the state to restore,
                expected to have a 'generation' key.
        """
        self.generation = state.get("generation", 0)
