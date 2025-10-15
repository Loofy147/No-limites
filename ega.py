import random
import numpy as np
from framework import BaseAlgorithm
import multiprocessing as mp


class Individual:
    """Represents an individual in the Epigenetic Genetic Algorithm.

    Each individual has a dual-layered genetic structure consisting of a stable
    genotype and a more volatile epigenome. The phenotype, which is the
    expressed genetic material evaluated by the fitness function, is derived
    from the interaction between these two layers.

    Attributes:
        genotype (list[int]): A list of binary values representing the core
            genetic code. It mutates at a low rate.
        epigenome (list[int]): A list of binary values that acts as a mask on
            the genotype, controlling gene expression. It mutates at a higher
            rate, enabling rapid exploration of the solution space.
        fitness (float): The fitness score of the individual, computed based on
            its phenotype. Initially set to 0.
    """

    def __init__(self, size):
        """Initializes an Individual with a random genotype and epigenome.

        Args:
            size (int): The length of the genotype and epigenome bit strings.
        """
        self.genotype = [random.randint(0, 1) for _ in range(size)]
        self.epigenome = [random.randint(0, 1) for _ in range(size)]
        self.fitness = 0

    def calculate_phenotype(self):
        """Calculates the phenotype by applying the epigenome to the genotype.

        The phenotype represents the expressed genes. In this implementation,
        it is determined by a bitwise AND operation between the genotype and
        the epigenome. This means a gene is only expressed in the phenotype if
        it is present in both the genotype and the epigenome.

        Returns:
            list[int]: The calculated phenotype bit string.
        """
        return [g & e for g, e in zip(self.genotype, self.epigenome)]

    def __repr__(self):
        """Provides a string representation of the individual."""
        return (
            f"Genotype:  {''.join(map(str, self.genotype))}\n"
            f"Epigenome: {''.join(map(str, self.epigenome))}\n"
            f"Fitness:   {self.fitness}"
        )

    def to_dict(self):
        """Serializes the individual's state to a dictionary.

        This is used for checkpointing, allowing the full state of an
        individual to be saved and later restored.

        Returns:
            dict: A dictionary containing the genotype, epigenome, and fitness.
        """
        return {
            "genotype": self.genotype,
            "epigenome": self.epigenome,
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, data):
        """Creates an Individual instance from a dictionary.

        This is the counterpart to `to_dict`, used to restore an individual's
        state from a checkpoint.

        Args:
            data (dict): A dictionary containing 'genotype', 'epigenome',
                and 'fitness' keys.

        Returns:
            Individual: A new instance of the Individual class with the
            restored state.
        """
        ind = cls(len(data["genotype"]))
        ind.genotype = data["genotype"]
        ind.epigenome = data["epigenome"]
        ind.fitness = data["fitness"]
        return ind


class Population:
    """Manages a collection of `Individual` objects.

    This class encapsulates the list of individuals that make up the
    population, providing helpful methods for accessing and managing them.

    Attributes:
        individuals (list[Individual]): The list of individuals in the
            population.
    """

    def __init__(self, population_size, individual_size):
        """Initializes a population of individuals.

        Args:
            population_size (int): The number of individuals in the population.
            individual_size (int): The size of each individual's genome.
        """
        self.individuals = [
            Individual(individual_size) for _ in range(population_size)
        ]

    def get_fittest(self):
        """Finds and returns the individual with the highest fitness score.

        This method is used to identify the best solution in the current
        population.

        Returns:
            Individual: The individual with the highest fitness score.
        """
        return max(self.individuals, key=lambda ind: ind.fitness)

    def __len__(self):
        """Returns the number of individuals in the population."""
        return len(self.individuals)

    def __getitem__(self, index):
        """Allows accessing individuals in the population by index."""
        return self.individuals[index]

    def to_dict(self):
        """Serializes the entire population to a dictionary.

        This method is used for checkpointing by converting every individual
        in the population into its dictionary representation.

        Returns:
            dict: A dictionary containing the serialized list of individuals.
        """
        return {"individuals": [ind.to_dict() for ind in self.individuals]}

    @classmethod
    def from_dict(cls, data):
        """Creates a Population instance from a dictionary.

        This method restores a population's state from a checkpoint.

        Args:
            data (dict): A dictionary containing a list of serialized
                individuals under the 'individuals' key.

        Returns:
            Population: A new instance of the Population class with the
            restored individuals.
        """
        pop = cls(0, 0)  # Dummy initialization
        pop.individuals = [
            Individual.from_dict(ind_data) for ind_data in data["individuals"]
        ]
        return pop


class EpigeneticAlgorithm(BaseAlgorithm):
    """Implements the Epigenetic Genetic Algorithm (EGA).

    This algorithm evolves a population of individuals, each with a distinct
    genotype and epigenome, to solve an optimization problem. The dual-layer
    genetic representation allows for a balance between preserving good
    genetic material (genotype) and rapidly exploring the solution space
    (epigenome).

    Attributes:
        population (Population): The population of individuals.
        genotype_mutation_rate (float): The mutation rate for the genotype.
        epigenome_mutation_rate (float): The mutation rate for the epigenome.
        crossover_rate (float): The probability of crossover.
        tournament_size (int): The number of individuals in a selection
            tournament.
        elitism_size (int): The number of best individuals to carry over to
            the next generation.
    """

    def __init__(
        self,
        population_size,
        individual_size,
        genotype_mutation_rate,
        epigenome_mutation_rate,
        crossover_rate,
        tournament_size,
        elitism_size,
        fitness_timeout=10,  # Add timeout parameter
        **kwargs,
    ):
        """Initializes the EpigeneticAlgorithm.

        Args:
            population_size (int): The number of individuals in the population.
            individual_size (int): The size of the genome (genotype and
                epigenome).
            genotype_mutation_rate (float): The probability of a bit flip in
                the genotype.
            epigenome_mutation_rate (float): The probability of a bit flip in
                the epigenome.
            crossover_rate (float): The probability that crossover will occur
                between two parents.
            tournament_size (int): The number of individuals to select for a
                selection tournament.
            elitism_size (int): The number of the fittest individuals to
                carry over to the next generation without modification.
            fitness_timeout (int, optional): The maximum time in seconds
                allowed for a single fitness evaluation. Defaults to 10.
            **kwargs: Catches any unused arguments passed in from the runners.
        """
        self.population = Population(population_size, individual_size)
        self.genotype_mutation_rate = genotype_mutation_rate
        self.epigenome_mutation_rate = epigenome_mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.fitness_timeout = fitness_timeout

    def _calculate_fitness(self, individual, fitness_function):
        """Calculates and sets the fitness for a single individual.

        This method includes a cross-platform timeout mechanism to handle
        fitness functions that may hang or run indefinitely.

        Args:
            individual (Individual): The individual to evaluate.
            fitness_function (callable): The function to calculate fitness.

        Raises:
            TimeoutError: If the fitness function execution exceeds the
                `fitness_timeout`.
            RuntimeError: If the fitness worker process exits unexpectedly.
            TypeError: If the fitness function returns a non-numeric value.
        """

        # Helper function to run in a separate process
        def fitness_worker(phenotype, func, queue):
            try:
                result = func(phenotype)
                queue.put(result)
            except Exception as e:
                queue.put(e)

        phenotype = individual.calculate_phenotype()

        if self.fitness_timeout <= 0:
            # Run without a timeout
            fitness = fitness_function(phenotype)
        else:
            # Run with a timeout in a separate process
            q = mp.Queue()
            p = mp.Process(target=fitness_worker, args=(phenotype, fitness_function, q))
            p.start()
            p.join(self.fitness_timeout)

            if p.is_alive():
                p.terminate()
                p.join()
                raise TimeoutError("Fitness function evaluation timed out.")

            if q.empty():
                raise RuntimeError("Fitness function process finished but returned no result.")

            result = q.get()
            if isinstance(result, Exception):
                raise result  # Re-raise exception from the child process
            fitness = result

        if not isinstance(fitness, (int, float)):
            raise TypeError(
                f"Fitness function must return a number, but got {type(fitness)}"
            )
        individual.fitness = fitness

    def _selection(self):
        """Selects a parent from the population using tournament selection.

        A random subset of the population of size `tournament_size` is chosen,
        and the individual with the highest fitness from this subset is
        selected as a parent.

        Returns:
            Individual: The selected parent individual.
        """
        tournament = random.sample(self.population.individuals, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2):
        """Performs single-point crossover on the genotype and epigenome.

        If a random draw is below the `crossover_rate`, a single crossover
        point is chosen. The child inherits the first part of its genotype and
        epigenome from `parent1` and the second part from `parent2`. Otherwise,
        the child is a direct clone of `parent1`.

        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.

        Returns:
            Individual: A new child individual.
        """
        child = Individual(len(parent1.genotype))
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1.genotype) - 1)
            # Crossover for genotype
            child.genotype = (
                parent1.genotype[:crossover_point] + parent2.genotype[crossover_point:]
            )
            # Crossover for epigenome
            child.epigenome = (
                parent1.epigenome[:crossover_point]
                + parent2.epigenome[crossover_point:]
            )
        else:
            # If no crossover, child is a clone of the first parent
            child.genotype = parent1.genotype[:]
            child.epigenome = parent1.epigenome[:]
        return child

    def _mutate(self, individual):
        """Mutates an individual's genotype and epigenome.

        The genotype and epigenome are mutated at different, independent rates.
        Typically, the epigenome mutation rate is set higher to encourage
        exploration.

        Args:
            individual (Individual): The individual to mutate.
        """
        # Mutate genotype
        for i in range(len(individual.genotype)):
            if random.random() < self.genotype_mutation_rate:
                individual.genotype[i] = 1 - individual.genotype[i]
        # Mutate epigenome
        for i in range(len(individual.epigenome)):
            if random.random() < self.epigenome_mutation_rate:
                individual.epigenome[i] = 1 - individual.epigenome[i]

    def evolve(self, fitness_function):
        """Performs one full generation of the Epigenetic Genetic Algorithm.

        This method orchestrates the main evolutionary loop: fitness
        calculation for the current population, selection, crossover, mutation,
        and the creation of a new generation. It also includes a mechanism to
        detect non-deterministic fitness functions.

        Args:
            fitness_function (callable): The function used to evaluate the
                fitness of each individual's phenotype.

        Returns:
            tuple[float, float]: A tuple containing the best and average
            fitness scores of the new generation.

        Raises:
            ValueError: If the fitness function returns different values for
                the same phenotype within the same generation.
        """
        self.adapt_parameters()

        # Use a cache to detect non-deterministic fitness functions within this generation
        fitness_cache = {}

        # --- Fitness Calculation for Current Population ---
        for ind in self.population.individuals:
            phenotype = tuple(ind.calculate_phenotype())

            # Calculate the fitness
            self._calculate_fitness(ind, fitness_function)

            # Check against cache
            if phenotype in fitness_cache:
                if fitness_cache[phenotype] != ind.fitness:
                    raise ValueError("Non-deterministic fitness function detected!")
            else:
                fitness_cache[phenotype] = ind.fitness

        # --- Create New Generation ---
        new_population_individuals = []

        # Apply elitism
        if self.elitism_size > 0:
            sorted_population = sorted(
                self.population.individuals, key=lambda ind: ind.fitness, reverse=True
            )
            elites = sorted_population[: self.elitism_size]
            new_population_individuals.extend(elites)

        # Create the rest of the new individuals through evolution
        remaining_size = len(self.population) - self.elitism_size
        for _ in range(remaining_size):
            parent1 = self._selection()
            parent2 = self._selection()
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_population_individuals.append(child)

        self.population.individuals = new_population_individuals

        # Calculate fitness for the new generation and gather stats
        total_fitness = 0
        for ind in self.population.individuals:
            self._calculate_fitness(ind, fitness_function)
            total_fitness += ind.fitness

        best_fitness = self.population.get_fittest().fitness
        avg_fitness = total_fitness / len(self.population)

        return best_fitness, avg_fitness

    def get_fittest_individual(self):
        """Returns the best individual from the current population."""
        return self.population.get_fittest()

    def get_state(self):
        """Serializes the current state of the algorithm for checkpointing.

        This method captures all the necessary information to save and later
        resume the algorithm's execution state. It includes the full
        population and the states of the random number generators.

        Returns:
            dict: A dictionary representing the current state, with the
            following structure:
            - 'population' (dict): The serialized population state.
            - 'random_state' (tuple): The state of Python's `random` module.
            - 'numpy_random_state' (dict): The state of NumPy's random
              number generator.
        """
        return {
            "population": self.population.to_dict(),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        }

    def set_state(self, state):
        """Restores the algorithm's state from a checkpoint.

        Args:
            state (dict): A dictionary, as returned by `get_state()`,
                representing the state to restore. It must contain keys
                for 'population', 'random_state', and 'numpy_random_state'.
        """
        self.population = Population.from_dict(state["population"])
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_state"])
