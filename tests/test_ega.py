import unittest
from unittest import mock
import random
from ega import Individual, Population, EpigeneticAlgorithm


class TestIndividual(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.individual = Individual(self.size)

    def test_initialization(self):
        self.assertEqual(len(self.individual.genotype), self.size)
        self.assertEqual(len(self.individual.epigenome), self.size)
        self.assertEqual(self.individual.fitness, 0)

    def test_calculate_phenotype(self):
        self.individual.genotype = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]
        self.individual.epigenome = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1]
        phenotype = self.individual.calculate_phenotype()
        expected_phenotype = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
        self.assertEqual(phenotype, expected_phenotype)


class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.pop_size = 10
        self.ind_size = 5
        self.population = Population(self.pop_size, self.ind_size)

    def test_initialization(self):
        self.assertEqual(len(self.population.individuals), self.pop_size)
        self.assertIsInstance(self.population[0], Individual)
        self.assertEqual(len(self.population[0].genotype), self.ind_size)

    def test_get_fittest(self):
        for i, ind in enumerate(self.population.individuals):
            ind.fitness = i
        fittest = self.population.get_fittest()
        self.assertEqual(fittest.fitness, self.pop_size - 1)


class TestEpigeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        # Set a fixed seed for reproducibility
        random.seed(42)
        self.ega = EpigeneticAlgorithm(
            population_size=10,
            individual_size=8,
            genotype_mutation_rate=0.1,
            epigenome_mutation_rate=0.5,
            crossover_rate=0.8,
            tournament_size=3,
            elitism_size=1,
        )

    def test_initialization(self):
        self.assertEqual(len(self.ega.population), 10)
        self.assertEqual(self.ega.elitism_size, 1)

    def test_crossover(self):
        parent1 = Individual(8)
        parent2 = Individual(8)
        parent1.genotype = [0] * 8
        parent1.epigenome = [0] * 8
        parent2.genotype = [1] * 8
        parent2.epigenome = [1] * 8

        # Force crossover to happen by mocking random calls
        with mock.patch("random.random", return_value=0.0), mock.patch(
            "random.randint", return_value=4
        ):
            child = self.ega._crossover(parent1, parent2)
            self.assertEqual(child.genotype, [0, 0, 0, 0, 1, 1, 1, 1])
            self.assertEqual(child.epigenome, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_mutate(self):
        individual = Individual(8)
        individual.genotype = [0] * 8
        individual.epigenome = [0] * 8

        # Force mutation to happen by mocking random.random to be less than any rate
        with mock.patch("random.random", return_value=0.0):
            self.ega._mutate(individual)
            self.assertEqual(individual.genotype, [1] * 8)
            self.assertEqual(individual.epigenome, [1] * 8)

    def test_evolve_with_elitism(self):
        # Dummy fitness function
        def dummy_fitness(phenotype):
            return sum(phenotype)

        # Create a guaranteed elite individual with a perfect genotype and epigenome
        fittest_ind = self.ega.population[0]
        fittest_ind.genotype = [1] * 8
        fittest_ind.epigenome = [1] * 8

        # Keep track of the original fittest individual's genotype and epigenome
        original_genotype = fittest_ind.genotype[:]
        original_epigenome = fittest_ind.epigenome[:]

        self.ega.evolve(dummy_fitness)

        # Check if the elite individual is present in the new population
        new_population_genomes = [
            (ind.genotype, ind.epigenome) for ind in self.ega.population
        ]
        self.assertIn((original_genotype, original_epigenome), new_population_genomes)


if __name__ == "__main__":
    unittest.main()
