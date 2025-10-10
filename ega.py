import random
import numpy as np
from framework import BaseAlgorithm


class Individual:
    """Represents an individual in the EGA population.

    An individual has a dual-layered genetic structure: a stable genotype
    and a rapidly mutating epigenome. The phenotype, which is evaluated
    by the fitness function, is a result of the interaction between these
    two layers.

    Attributes:
        genotype (list[int]): The underlying, slow-mutating genetic code.
        epigenome (list[int]): A layer that masks or expresses genes in the
            genotype. It mutates at a higher rate to allow for rapid
            exploration.
        fitness (float): The fitness score of the individual, calculated
            based on its phenotype.
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


class Population:
    """Represents a collection of `Individual` objects.

    Attributes:
        individuals (list[Individual]): A list of individuals in the
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
        """Returns the individual with the highest fitness in the population.

        Returns:
            Individual: The fittest individual.
        """
        return max(self.individuals, key=lambda ind: ind.fitness)

    def __len__(self):
        """Returns the number of individuals in the population."""
        return len(self.individuals)

    def __getitem__(self, index):
        """Allows accessing individuals in the population by index."""
        return self.individuals[index]


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
        **kwargs,
    ):
        """Initializes the Epigenetic Genetic Algorithm.

        Args:
            population_size (int): The number of individuals in the population.
            individual_size (int): The size of the genome.
            genotype_mutation_rate (float): The mutation rate for the genotype.
            epigenome_mutation_rate (float): The mutation rate for the
                epigenome.
            crossover_rate (float): The rate at which to perform crossover.
            tournament_size (int): The number of individuals to select for a
                tournament.
            elitism_size (int): The number of fittest individuals to carry over
                to the next generation.
            **kwargs: Catches any unused arguments passed from the runner.
        """
        self.population = Population(population_size, individual_size)
        self.genotype_mutation_rate = genotype_mutation_rate
        self.epigenome_mutation_rate = epigenome_mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size

    def _calculate_fitness(self, individual, fitness_function):
        """Calculates and sets the fitness for a single individual.

        Args:
            individual (Individual): The individual to evaluate.
            fitness_function (callable): The function to score the phenotype.
        """
        phenotype = individual.calculate_phenotype()
        individual.fitness = fitness_function(phenotype)

    def _selection(self):
        """Selects an individual using tournament selection.

        A random subset of the population is chosen (the "tournament"), and
        the fittest individual from this subset is selected.

        Returns:
            Individual: The selected individual.
        """
        tournament = random.sample(self.population.individuals, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2):
        """Performs single-point crossover on two parents.

        Crossover is applied independently to both the genotype and the
        epigenome, creating a new child.

        Args:
            parent1 (Individual): The first parent.
            parent2 (Individual): The second parent.

        Returns:
            Individual: The resulting child.
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
        """Performs one full cycle of evolution.

        The process includes:
        1. Optional parameter adaptation.
        2. Fitness calculation for the current population.
        3. Elitism to preserve the best individuals.
        4. Creation of a new generation through selection, crossover, and
           mutation.
        5. Fitness calculation for the new generation.

        Args:
            fitness_function (callable): The function to evaluate fitness.

        Returns:
            tuple: A tuple containing the best fitness and the average fitness
                   of the new generation.
        """
        # Allow the algorithm to adapt its own parameters
        self.adapt_parameters()

        # Calculate fitness for the current population to prepare for selection
        for ind in self.population.individuals:
            self._calculate_fitness(ind, fitness_function)

        new_population_individuals = []

        # Apply elitism: carry over the best individuals
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
        """Returns the current state of the algorithm for checkpointing."""
        return {
            "population": self.population,
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        }

    def set_state(self, state):
        """Restores the algorithm's state from a checkpoint.

        Args:
            state (dict): The state dictionary to restore.
        """
        self.population = state["population"]
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_state"])
