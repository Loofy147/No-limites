import unittest
from unittest import mock
import random
from standard_ga import StandardIndividual, StandardPopulation, StandardAlgorithm

class TestStandardIndividual(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.individual = StandardIndividual(self.size)

    def test_initialization(self):
        self.assertEqual(len(self.individual.genotype), self.size)
        self.assertEqual(self.individual.fitness, 0)

class TestStandardPopulation(unittest.TestCase):
    def setUp(self):
        self.pop_size = 10
        self.ind_size = 5
        self.population = StandardPopulation(self.pop_size, self.ind_size)

    def test_initialization(self):
        self.assertEqual(len(self.population.individuals), self.pop_size)
        self.assertIsInstance(self.population[0], StandardIndividual)
        self.assertEqual(len(self.population[0].genotype), self.ind_size)

    def test_get_fittest(self):
        for i, ind in enumerate(self.population.individuals):
            ind.fitness = i
        fittest = self.population.get_fittest()
        self.assertEqual(fittest.fitness, self.pop_size - 1)

class TestStandardAlgorithm(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.sga = StandardAlgorithm(
            population_size=10,
            individual_size=8,
            mutation_rate=0.1,
            crossover_rate=0.8,
            tournament_size=3,
            elitism_size=1
        )

    def test_initialization(self):
        self.assertEqual(len(self.sga.population), 10)
        self.assertEqual(self.sga.elitism_size, 1)

    def test_crossover(self):
        parent1 = StandardIndividual(8)
        parent2 = StandardIndividual(8)
        parent1.genotype = [0] * 8
        parent2.genotype = [1] * 8

        with mock.patch('random.random', return_value=0.0), \
             mock.patch('random.randint', return_value=4):
            child = self.sga._crossover(parent1, parent2)
            self.assertEqual(child.genotype, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_mutate(self):
        individual = StandardIndividual(8)
        individual.genotype = [0] * 8

        with mock.patch('random.random', return_value=0.0):
            self.sga._mutate(individual)
            self.assertEqual(individual.genotype, [1] * 8)

    def test_evolve_with_elitism(self):
        def dummy_fitness(genotype):
            return sum(genotype)

        fittest_ind = self.sga.population[0]
        fittest_ind.genotype = [1] * 8

        original_genotype = fittest_ind.genotype[:]

        self.sga.evolve(dummy_fitness)

        new_population_genomes = [ind.genotype for ind in self.sga.population]
        self.assertIn(original_genotype, new_population_genomes)

if __name__ == '__main__':
    unittest.main()