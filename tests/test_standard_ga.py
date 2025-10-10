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
        self.population_size = 10
        self.sga = StandardAlgorithm(
            population_size=self.population_size,
            individual_size=8,
            mutation_rate=0.1,
            crossover_rate=0.8,
            tournament_size=3,
            elitism_size=1,
        )

    def test_initialization(self):
        self.assertEqual(len(self.sga.population), 10)
        self.assertEqual(self.sga.elitism_size, 1)

    def test_crossover(self):
        parent1 = StandardIndividual(8)
        parent2 = StandardIndividual(8)
        parent1.genotype = [0] * 8
        parent2.genotype = [1] * 8

        with mock.patch("random.random", return_value=0.0), mock.patch(
            "random.randint", return_value=4
        ):
            child = self.sga._crossover(parent1, parent2)
            self.assertEqual(child.genotype, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_mutate(self):
        individual = StandardIndividual(8)
        individual.genotype = [0] * 8

        with mock.patch("random.random", return_value=0.0):
            self.sga._mutate(individual)
            self.assertEqual(individual.genotype, [1] * 8)

    def test_evolve_with_elitism(self):
        """
        Test that elitism correctly preserves the fittest individual object
        across generations.
        """
        # Define a simple fitness function for the test
        def dummy_fitness(genotype):
            return sum(genotype)

        # 1. Create a population where one individual is exceptionally fit
        # Make all other individuals have low fitness
        for ind in self.sga.population.individuals:
            ind.genotype = [0] * 8

        # Create a single, clearly fittest individual
        fittest_individual = self.sga.population.individuals[0]
        fittest_individual.genotype = [1] * 8

        # 2. Calculate fitness and verify the setup
        for ind in self.sga.population.individuals:
            ind.fitness = dummy_fitness(ind.genotype)

        # Sanity check that our chosen individual is indeed the fittest
        self.assertIs(self.sga.population.get_fittest(), fittest_individual)
        self.assertEqual(fittest_individual.fitness, 8)

        # 3. Evolve the population for one generation
        self.sga.evolve(dummy_fitness)

        # 4. Assert that the *exact same object* is still in the new population
        # This is a stronger check than just checking for the genotype.
        self.assertIn(
            fittest_individual,
            self.sga.population.individuals,
            "The fittest individual object should be preserved by elitism.",
        )

    def test_population_size_is_maintained(self):
        """
        Test that the population size remains constant after evolution,
        especially when elitism is active.
        """
        initial_pop_size = self.sga.population_size
        self.assertEqual(len(self.sga.population), initial_pop_size)

        # Evolve the population
        self.sga.evolve(lambda g: sum(g))

        # Assert that the population size is unchanged
        self.assertEqual(
            len(self.sga.population),
            initial_pop_size,
            "Population size should not change after evolution.",
        )


if __name__ == "__main__":
    unittest.main()
