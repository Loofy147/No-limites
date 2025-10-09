from ega import EpigeneticAlgorithm


class AdaptiveEGA(EpigeneticAlgorithm):
    """
    An adaptive version of the Epigenetic Genetic Algorithm.

    This algorithm enhances the standard EGA by dynamically adapting its own
    parameters during a run. Specifically, it increases the epigenome mutation
    rate when it detects that the search has stagnated, helping it to escape
    local optima.
    """

    def __init__(
        self,
        stagnation_limit=10,
        adaptation_factor=1.5,
        **kwargs,
    ):
        """
        Initializes the AdaptiveEGA.

        Args:
            stagnation_limit (int): The number of generations without improvement
                                  before adaptation is triggered.
            adaptation_factor (float): The factor by which to increase the
                                     epigenome mutation rate during adaptation.
            **kwargs: All other parameters required by the parent EpigeneticAlgorithm.
        """
        # Initialize the parent class with all its required arguments
        super().__init__(**kwargs)

        # --- Adaptive State Initialization ---
        self.stagnation_limit = stagnation_limit
        self.adaptation_factor = adaptation_factor

        # Store the base rate to revert to after a successful adaptation
        self.base_epigenome_mutation_rate = self.epigenome_mutation_rate

        self.stagnation_counter = 0
        self.best_fitness_so_far = -float("inf")

    def adapt_parameters(self):
        """
        The core adaptation logic. Called once per generation.
        """
        current_best_fitness = self.population.get_fittest().fitness

        if current_best_fitness > self.best_fitness_so_far:
            # We've found a new best solution, so reset everything
            self.best_fitness_so_far = current_best_fitness
            self.stagnation_counter = 0
            # Revert mutation rate to its base value if it was adapted
            if self.epigenome_mutation_rate != self.base_epigenome_mutation_rate:
                print(f"    (i) New best found. Reverting epigenome mutation rate to {self.base_epigenome_mutation_rate:.3f}")
                self.epigenome_mutation_rate = self.base_epigenome_mutation_rate
        else:
            # No improvement, increment stagnation counter
            self.stagnation_counter += 1

        # If stagnation limit is reached, trigger adaptation
        if self.stagnation_counter >= self.stagnation_limit:
            print(f"    (!) Stagnation detected. Adapting epigenome mutation rate.")
            self.epigenome_mutation_rate *= self.adaptation_factor
            # Clamp the mutation rate to a maximum of 1.0
            self.epigenome_mutation_rate = min(self.epigenome_mutation_rate, 1.0)
            print(f"        New epigenome mutation rate: {self.epigenome_mutation_rate:.3f}")
            # Reset counter to give the new rate time to work
            self.stagnation_counter = 0

    def get_state(self):
        """Overrides parent to include adaptive state for checkpointing."""
        state = super().get_state()
        state["adaptive_state"] = {
            "stagnation_counter": self.stagnation_counter,
            "best_fitness_so_far": self.best_fitness_so_far,
            "epigenome_mutation_rate": self.epigenome_mutation_rate,
        }
        return state

    def set_state(self, state):
        """Overrides parent to restore adaptive state from a checkpoint."""
        super().set_state(state)
        adaptive_state = state["adaptive_state"]
        self.stagnation_counter = adaptive_state["stagnation_counter"]
        self.best_fitness_so_far = adaptive_state["best_fitness_so_far"]
        self.epigenome_mutation_rate = adaptive_state["epigenome_mutation_rate"]