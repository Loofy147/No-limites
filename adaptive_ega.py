from ega import EpigeneticAlgorithm


class AdaptiveEGA(EpigeneticAlgorithm):
    """An adaptive version of the Epigenetic Genetic Algorithm.

    This algorithm enhances the standard EGA by dynamically adapting its
    hyperparameters during a run. It monitors its own performance and, upon
    detecting stagnation (a lack of improvement in fitness), it temporarily
    increases the epigenome mutation rate to encourage exploration and escape
    local optima.

    Attributes:
        stagnation_limit (int): The number of generations without fitness
            improvement before adaptation is triggered.
        adaptation_factor (float): The multiplicative factor used to
            increase the epigenome mutation rate.
        base_epigenome_mutation_rate (float): The original, baseline
            epigenome mutation rate, which is restored after a successful
            adaptation.
        stagnation_counter (int): The number of consecutive generations
            without improvement.
        best_fitness_so_far (float): The highest fitness score observed
            during the run.
    """

    def __init__(self, stagnation_limit=10, adaptation_factor=1.5, **kwargs):
        """Initializes the AdaptiveEGA.

        Args:
            stagnation_limit (int, optional): The number of consecutive
                generations without fitness improvement before adaptation is
                triggered. Defaults to 10.
            adaptation_factor (float, optional): The factor by which the
                epigenome mutation rate is multiplied during an adaptation
                event. Defaults to 1.5.
            **kwargs: All other parameters required by the parent
                `EpigeneticAlgorithm`, which are passed down to its
                constructor.
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
        """The core adaptation logic, called once per generation.

        This method monitors the best fitness. If the fitness fails to
        improve for `stagnation_limit` generations, it increases the
        `epigenome_mutation_rate` by `adaptation_factor`. Once a new best
        fitness is found, the mutation rate is reset to its original base
        value.
        """
        current_best_fitness = self.population.get_fittest().fitness

        if current_best_fitness > self.best_fitness_so_far:
            # We've found a new best solution, so reset everything
            self.best_fitness_so_far = current_best_fitness
            self.stagnation_counter = 0
            # Revert mutation rate to its base value if it was adapted
            if self.epigenome_mutation_rate != self.base_epigenome_mutation_rate:
                print(
                    "    (i) New best found. Reverting epigenome mutation rate to "
                    f"{self.base_epigenome_mutation_rate:.3f}"
                )
                self.epigenome_mutation_rate = self.base_epigenome_mutation_rate
        else:
            # No improvement, increment stagnation counter
            self.stagnation_counter += 1

        # If stagnation limit is reached, trigger adaptation
        if self.stagnation_counter >= self.stagnation_limit:
            print("    (!) Stagnation detected. Adapting epigenome mutation rate.")
            self.epigenome_mutation_rate *= self.adaptation_factor
            # Clamp the mutation rate to a maximum of 1.0
            self.epigenome_mutation_rate = min(self.epigenome_mutation_rate, 1.0)
            print(
                "        New epigenome mutation rate: "
                f"{self.epigenome_mutation_rate:.3f}"
            )
            # Reset counter to give the new rate time to work
            self.stagnation_counter = 0

    def get_state(self):
        """Serializes the current state, including adaptive parameters.

        This method extends the parent's `get_state` to also include all
        variables related to the adaptation mechanism. This ensures that the
        adaptive behavior can be correctly resumed from a checkpoint.

        Returns:
            dict: The state dictionary, which includes the parent state and
            an 'adaptive_state' dictionary with the following structure:
            - 'stagnation_counter' (int): Current stagnation count.
            - 'best_fitness_so_far' (float): Best fitness seen so far.
            - 'epigenome_mutation_rate' (float): Current mutation rate.
            - 'base_epigenome_mutation_rate' (float): The original rate.
        """
        state = super().get_state()
        state["adaptive_state"] = {
            "stagnation_counter": self.stagnation_counter,
            "best_fitness_so_far": self.best_fitness_so_far,
            "epigenome_mutation_rate": self.epigenome_mutation_rate,
            "base_epigenome_mutation_rate": self.base_epigenome_mutation_rate,
        }
        return state

    def set_state(self, state):
        """Restores the full state, including adaptive parameters.

        This method extends the parent's `set_state` to also restore the
        state of the adaptive mechanism from a checkpoint.

        Args:
            state (dict): The state dictionary to restore. It must contain
                the parent state keys and an 'adaptive_state' dictionary.
        """
        super().set_state(state)
        adaptive_state = state["adaptive_state"]
        self.stagnation_counter = adaptive_state["stagnation_counter"]
        self.best_fitness_so_far = adaptive_state["best_fitness_so_far"]
        self.epigenome_mutation_rate = adaptive_state["epigenome_mutation_rate"]
        # Restore the base rate, which is critical for correct adaptation
        self.base_epigenome_mutation_rate = adaptive_state.get(
            "base_epigenome_mutation_rate", self.epigenome_mutation_rate
        )
