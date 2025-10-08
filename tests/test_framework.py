import unittest
from registry import (
    register_algorithm,
    register_fitness_function,
    ALGORITHMS,
    FITNESS_FUNCTIONS,
)
from framework import BaseAlgorithm


class TestFramework(unittest.TestCase):
    def setUp(self):
        """
        Clean up the registries before each test to ensure isolation.
        """
        ALGORITHMS.clear()
        FITNESS_FUNCTIONS.clear()

    def test_register_algorithm(self):
        """Test that a new algorithm can be successfully registered."""

        class MyTestAlgo(BaseAlgorithm):
            def __init__(self, **kwargs):
                pass

            def evolve(self, fitness_function):
                return 1, 1

            def get_fittest_individual(self):
                return "test"

        self.assertNotIn("my_test_algo", ALGORITHMS)
        register_algorithm("my_test_algo", MyTestAlgo)
        self.assertIn("my_test_algo", ALGORITHMS)
        self.assertIs(ALGORITHMS["my_test_algo"], MyTestAlgo)

    def test_register_fitness_function(self):
        """Test that a new fitness function can be successfully registered."""

        def my_test_func(c):
            return 1

        self.assertNotIn("my_test_func", FITNESS_FUNCTIONS)
        register_fitness_function("my_test_func", my_test_func)
        self.assertIn("my_test_func", FITNESS_FUNCTIONS)
        self.assertIs(FITNESS_FUNCTIONS["my_test_func"], my_test_func)

    def test_duplicate_algorithm_registration_raises_error(self):
        """Test that registering a duplicate algorithm name raises a ValueError."""

        class MyTestAlgo(BaseAlgorithm):
            def __init__(self, **kwargs):
                pass

            def evolve(self, fitness_function):
                pass

            def get_fittest_individual(self):
                pass

        register_algorithm("duplicate_algo", MyTestAlgo)
        with self.assertRaises(ValueError):
            register_algorithm("duplicate_algo", MyTestAlgo)

    def test_duplicate_fitness_function_registration_raises_error(self):
        """Test that registering a duplicate fitness function name raises a ValueError."""

        def my_test_func(c):
            return 1

        register_fitness_function("duplicate_func", my_test_func)
        with self.assertRaises(ValueError):
            register_fitness_function("duplicate_func", my_test_func)


if __name__ == "__main__":
    unittest.main()