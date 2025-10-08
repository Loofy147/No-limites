import random
from framework import BaseAlgorithm


class Individual:
    """
    Represents an individual in the population.

    An individual has both a genotype (the underlying genetic material) and an
    epigenome (which controls the expression of the genotype).
    """

    def __init__(self, size):
        """
        Initializes an Individual with a random genotype and epigenome.

        Args:
            size (int): The length of the genotype and epigenome strings.
        """
        self.genotype = [random.randint(0, 1) for _ in range(size)]
        self.epigenome = [random.randint(0, 1) for _ in range(size)]
        self.fitness = 0

    def calculate_phenotype(self):
        """
        Calculates the phenotype by applying the epigenome to the genotype.

        The phenotype is the expressed set of genes, determined by a bitwise
        AND operation between the genotype and epigenome.
        """
        return [g & e for g, e in zip(self.genotype, self.epigenome)]

    def __repr__(self):
        return (
            f"Genotype:  {''.join(map(str, self.genotype))}\n"
            f"Epigenome: {''.join(map(str, self.epigenome))}\n"
            f"Fitness:   {self.fitness}"
        )


class Population:
    """
    Represents a collection of individuals.
    """

    def __init__(self, population_size, individual_size):
        """
        Initializes a population of individuals.

        Args:
            population_size (int): The number of individuals in the population.
            individual_size (int): The size of each individual's genome.
        """
        self.individuals = [Individual(individual_size) for _ in range(population_size)]

    def get_fittest(self):
        """
        Returns the individual with the highest fitness in the population.
        """
        return max(self.individuals, key=lambda ind: ind.fitness)

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]


class EpigeneticAlgorithm(BaseAlgorithm):
    """
    Implements the Epigenetic Genetic Algorithm.

    This algorithm evolves a population of individuals, each with a genotype
    and an epigenome, to solve an optimization problem.
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
        """
        Initializes the Epigenetic Genetic Algorithm.

        Args:
            population_size (int): The number of individuals in the population.
            individual_size (int): The size of the genome.
            genotype_mutation_rate (float): The mutation rate for the genotype.
            epigenome_mutation_rate (float): The mutation rate for the epigenome.
            crossover_rate (float): The rate at which to perform crossover.
            tournament_size (int): The number of individuals to select for a
                                 tournament.
            elitism_size (int): The number of fittest individuals to carry over
                              to the next generation.
        """
        self.population = Population(population_size, individual_size)
        self.genotype_mutation_rate = genotype_mutation_rate
        self.epigenome_mutation_rate = epigenome_mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size

    def _calculate_fitness(self, individual, fitness_function):
        """Calculates and sets the fitness for an individual."""
        phenotype = individual.calculate_phenotype()
        individual.fitness = fitness_function(phenotype)

    def _selection(self):
        """
        Selects an individual from the population using tournament selection.
        """
        tournament = random.sample(self.population.individuals, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2):
        """
        Performs crossover on both the genotype and epigenome.
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
            child.genotype = parent1.genotype[:]
            child.epigenome = parent1.epigenome[:]
        return child

    def _mutate(self, individual):
        """
        Mutates the genotype and epigenome at different rates.
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
        """
        Performs one full cycle of evolution and returns performance stats.

        The process includes:
        1.  Fitness calculation for the current population.
        2.  Selection of parents.
        3.  Elitism to preserve the best individuals.
        4.  Crossover and mutation to create the new generation.

        Returns:
            tuple: A tuple containing the best fitness and the average fitness
                   of the new generation.
        """
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
        for _ in range(len(self.population) - self.elitism_size):
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
