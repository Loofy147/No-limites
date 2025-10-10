import random
import numpy as np
from framework import BaseAlgorithm


class StandardIndividual:
    """
    Represents an individual in a standard Genetic Algorithm.
    It has a single genotype and no epigenome.
    """

    def __init__(self, size):
        self.genotype = [random.randint(0, 1) for _ in range(size)]
        self.fitness = 0

    def __repr__(self):
        return (
            f"Genotype: {''.join(map(str, self.genotype))}\n"
            f"Fitness:  {self.fitness}"
        )


class StandardPopulation:
    """
    Represents a collection of StandardIndividuals.
    """

    def __init__(self, population_size, individual_size):
        self.individuals = [
            StandardIndividual(individual_size) for _ in range(population_size)
        ]

    def get_fittest(self):
        return max(self.individuals, key=lambda ind: ind.fitness)

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]


class StandardAlgorithm(BaseAlgorithm):
    """
    Implements a standard Genetic Algorithm for comparison.
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
        self.population_size = population_size
        self.population = StandardPopulation(self.population_size, individual_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size

    def _calculate_fitness(self, individual, fitness_function):
        # In a standard GA, the genotype is the phenotype
        individual.fitness = fitness_function(individual.genotype)

    def _selection(self):
        tournament = random.sample(self.population.individuals, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2):
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
        for i in range(len(individual.genotype)):
            if random.random() < self.mutation_rate:
                individual.genotype[i] = 1 - individual.genotype[i]

    def evolve(self, fitness_function):
        """
        Performs one full cycle of evolution and returns performance stats.
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
        """Returns the current state of the algorithm for checkpointing."""
        return {
            "population": self.population,
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        }

    def set_state(self, state):
        """Restores the algorithm's state from a checkpoint."""
        self.population = state["population"]
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_state"])
