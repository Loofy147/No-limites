import random
import numpy as np
from framework import BaseAlgorithm


class StandardIndividual:
    """Represents an individual in a standard Genetic Algorithm.

    This is a simpler representation compared to the EGA's Individual, as it
    only has a single genotype that directly corresponds to its phenotype.

    Attributes:
        genotype (list[int]): The genetic code of the individual, which is
            also its phenotype.
        fitness (float): The fitness score of the individual.
    """

    def __init__(self, size):
        """Initializes an Individual with a random genotype.

        Args:
            size (int): The length of the genotype bit string.
        """
        self.genotype = [random.randint(0, 1) for _ in range(size)]
        self.fitness = 0

    def __repr__(self):
        """Provides a string representation of the individual."""
        return (
            f"Genotype: {''.join(map(str, self.genotype))}\n"
            f"Fitness:  {self.fitness}"
        )


class StandardPopulation:
    """Manages a collection of `StandardIndividual` objects.

    This class provides a container for the individuals in the population
    and includes helper methods for accessing them.

    Attributes:
        individuals (list[StandardIndividual]): The list of individuals.
    """

    def __init__(self, population_size, individual_size):
        """Initializes a population of standard individuals.

        Args:
            population_size (int): The number of individuals in the population.
            individual_size (int): The size of each individual's genome.
        """
        self.individuals = [
            StandardIndividual(individual_size) for _ in range(population_size)
        ]

    def get_fittest(self):
        """Finds and returns the individual with the highest fitness score.

        Returns:
            StandardIndividual: The individual with the highest fitness score.
        """
        return max(self.individuals, key=lambda ind: ind.fitness)

    def __len__(self):
        """Returns the number of individuals in the population."""
        return len(self.individuals)

    def __getitem__(self, index):
        """Allows accessing individuals in the population by index."""
        return self.individuals[index]


class StandardAlgorithm(BaseAlgorithm):
    """Implements a standard Genetic Algorithm (SGA).

    This class serves as a baseline for comparison against the EGA. It
    uses a traditional evolutionary approach with a single genetic layer
    (genotype) and standard operators for selection, crossover, and mutation.

    Attributes:
        population_size (int): The number of individuals in the population.
        population (StandardPopulation): The population of individuals.
        mutation_rate (float): The probability of a bit flip in the genotype.
        crossover_rate (float): The probability of crossover.
        tournament_size (int): The number of individuals in a selection
            tournament.
        elitism_size (int): The number of best individuals to carry over.
    """

    def __init__(
        self,
        population_size,
        individual_size,
        mutation_rate,
        crossover_rate,
        tournament_size,
        elitism_size,
        **kwargs,
    ):
        """Initializes the StandardAlgorithm.

        Args:
            population_size (int): The number of individuals in the population.
            individual_size (int): The size of the individual genome.
            mutation_rate (float): The probability of a bit flip mutation in
                the genotype.
            crossover_rate (float): The probability that crossover will occur
                between two parents.
            tournament_size (int): The number of individuals to select for a
                selection tournament.
            elitism_size (int): The number of the fittest individuals to
                carry over to the next generation without modification.
            **kwargs: Catches any unused arguments passed in from the runners.
        """
        self.population_size = population_size
        self.population = StandardPopulation(self.population_size, individual_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size

    def _calculate_fitness(self, individual, fitness_function):
        """Calculates and sets the fitness for an individual.

        Args:
            individual (StandardIndividual): The individual to evaluate.
            fitness_function (callable): The function to score the genotype.
        """
        # In a standard GA, the genotype is the phenotype
        individual.fitness = fitness_function(individual.genotype)

    def _selection(self):
        """Selects a parent from the population using tournament selection.

        A random subset of the population of size `tournament_size` is chosen,
        and the individual with the highest fitness from this subset is
        selected as a parent.

        Returns:
            StandardIndividual: The selected parent individual.
        """
        tournament = random.sample(self.population.individuals, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2):
        """Performs single-point crossover on the genotype of two parents.

        If a random draw is below the `crossover_rate`, a single crossover
        point is chosen. The child inherits the first part of its genotype
        from `parent1` and the second part from `parent2`. Otherwise, the child
        is a direct clone of `parent1`.

        Args:
            parent1 (StandardIndividual): The first parent.
            parent2 (StandardIndividual): The second parent.

        Returns:
            StandardIndividual: A new child individual.
        """
        child = StandardIndividual(len(parent1.genotype))
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1.genotype) - 1)
            child.genotype = (
                parent1.genotype[:crossover_point] + parent2.genotype[crossover_point:]
            )
        else:
            child.genotype = parent1.genotype[:]
        return child

    def _mutate(self, individual):
        """Mutates an individual's genotype.

        Args:
            individual (StandardIndividual): The individual to mutate.
        """
        for i in range(len(individual.genotype)):
            if random.random() < self.mutation_rate:
                individual.genotype[i] = 1 - individual.genotype[i]

    def evolve(self, fitness_function):
        """Performs one full generation of the standard Genetic Algorithm.

        This method orchestrates the main evolutionary loop: fitness
        calculation, selection, crossover, mutation, and the creation of a
        new generation.

        Args:
            fitness_function (callable): The function used to evaluate the
                fitness of each individual's genotype.

        Returns:
            tuple[float, float]: A tuple containing the best and average
            fitness scores of the new generation.
        """
        # Allow the algorithm to adapt its own parameters
        self.adapt_parameters()

        # Calculate fitness for the current population
        for ind in self.population.individuals:
            self._calculate_fitness(ind, fitness_function)

        new_population_individuals = []

        # Elitism
        if self.elitism_size > 0:
            sorted_population = sorted(
                self.population.individuals, key=lambda ind: ind.fitness, reverse=True
            )
            elites = sorted_population[: self.elitism_size]
            new_population_individuals.extend(elites)

        # Create the rest of the new generation
        for _ in range(self.population_size - self.elitism_size):
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
        """Returns the best individual from the population."""
        return self.population.get_fittest()

    def get_state(self):
        """Serializes the current state of the algorithm for checkpointing.

        This method captures the full state of the algorithm, including the
        population and the states of the random number generators, allowing
        for a complete and accurate resume.

        Returns:
            dict: A dictionary representing the current state, with the
            following structure:
            - 'population' (StandardPopulation): The current population object.
            - 'random_state' (tuple): The state of Python's `random` module.
            - 'numpy_random_state' (dict): The state of NumPy's random
              number generator.
        """
        return {
            "population": self.population,
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
        self.population = state["population"]
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_state"])
