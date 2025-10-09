import unittest
from adaptive_ega import AdaptiveEGA

class TestAdaptiveEGA(unittest.TestCase):
    def setUp(self):
        """Set up a default AdaptiveEGA instance for testing."""
        self.base_args = {
            "population_size": 10,
            "individual_size": 8,
            "genotype_mutation_rate": 0.01,
            "epigenome_mutation_rate": 0.05,
            "crossover_rate": 0.8,
            "tournament_size": 3,
            "elitism_size": 1,
            "stagnation_limit": 5,
            "adaptation_factor": 2.0,
        }
        self.algo = AdaptiveEGA(**self.base_args)

    def test_initialization(self):
        """Test that the adaptive parameters are initialized correctly."""
        self.assertEqual(self.algo.stagnation_limit, 5)
        self.assertEqual(self.algo.adaptation_factor, 2.0)
        self.assertEqual(self.algo.stagnation_counter, 0)
        self.assertEqual(self.algo.best_fitness_so_far, -float("inf"))
        self.assertEqual(self.algo.base_epigenome_mutation_rate, 0.05)

    def test_adaptation_trigger(self):
        """Test that the epigenome mutation rate adapts after stagnation."""
        # Simulate stagnation by keeping fitness the same
        self.algo.best_fitness_so_far = 20
        # Set a dummy fittest individual to avoid errors
        self.algo.population.individuals[0].fitness = 20

        # Stagnate for the limit
        for _ in range(5):
            self.algo.adapt_parameters()

        # On the 5th call, the adaptation should trigger
        self.assertEqual(self.algo.epigenome_mutation_rate, 0.05 * 2.0)
        # Counter should reset after adaptation
        self.assertEqual(self.algo.stagnation_counter, 0)

    def test_adaptation_reset_on_improvement(self):
        """Test that the mutation rate reverts to base after improvement."""
        # First, trigger an adaptation
        self.algo.best_fitness_so_far = 20
        self.algo.population.individuals[0].fitness = 20
        for _ in range(5):
            self.algo.adapt_parameters()

        self.assertNotEqual(self.algo.epigenome_mutation_rate, self.algo.base_epigenome_mutation_rate)

        # Now, simulate finding a better solution
        self.algo.population.individuals[0].fitness = 25 # New best fitness
        self.algo.adapt_parameters()

        # The mutation rate should have reverted to its base value
        self.assertEqual(self.algo.epigenome_mutation_rate, self.algo.base_epigenome_mutation_rate)
        self.assertEqual(self.algo.stagnation_counter, 0)
        self.assertEqual(self.algo.best_fitness_so_far, 25)

if __name__ == "__main__":
    unittest.main()