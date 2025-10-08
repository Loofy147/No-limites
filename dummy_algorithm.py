from framework import BaseAlgorithm


class DummyAlgorithm(BaseAlgorithm):
    """
    A simple 'dummy' algorithm for demonstration purposes.

    This algorithm does not perform any evolution. It simply exists to show
    how easily a new algorithm can be created and registered with the framework.
    """

    def __init__(self, **kwargs):
        """Initializes the dummy algorithm. Ignores all parameters."""
        print("Dummy Algorithm Initialized!")
        self.generation = 0

    def evolve(self, fitness_function):
        """
        Pretends to evolve, but just returns fixed dummy fitness values.
        The 'best' fitness will slowly increase to simulate progress.
        """
        self.generation += 1

        # Simulate some progress
        best_fitness = self.generation * 1.5
        avg_fitness = self.generation * 1.2

        return best_fitness, avg_fitness

    def get_fittest_individual(self):
        """Returns a placeholder string, as this is a dummy algorithm."""
        return "No individual, this is a dummy algorithm."

    def get_state(self):
        """Returns the current state of the dummy algorithm."""
        return {"generation": self.generation}

    def set_state(self, state):
        """Restores the state of the dummy algorithm."""
        self.generation = state.get("generation", 0)
