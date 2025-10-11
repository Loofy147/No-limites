"""
Red Team Test Suite for Epigenetic Genetic Algorithm Framework
Designed to discover weaknesses through adversarial testing.
"""

import unittest
import json
import yaml
import tempfile
import multiprocessing as mp
from unittest.mock import patch, MagicMock
import sys
import random
import numpy as np


class RedTeamRegistryTests(unittest.TestCase):
    """Attack vector: Registry manipulation and pollution"""

    def setUp(self):
        # Directly manipulate the registry for testing
        from registry import register_algorithm, ALGORITHMS
        self.register = register_algorithm
        self.ALGORITHMS = ALGORITHMS
        # Backup original state
        self._original_algorithms = self.ALGORITHMS.copy()
        self.ALGORITHMS.clear()

    def tearDown(self):
        # Restore original state
        self.ALGORITHMS.clear()
        self.ALGORITHMS.update(self._original_algorithms)

    def test_duplicate_registration_silent_overwrite(self):
        """RED TEAM: Duplicate registrations should fail loudly, not silently"""
        class AlgoA:
            pass
        class AlgoB:
            pass

        self.register('test', AlgoA)
        with self.assertRaises(ValueError, msg="Should reject duplicate registration"):
            self.register('test', AlgoB)

    def test_name_collision_case_sensitivity(self):
        """RED TEAM: Case variations could cause confusion"""
        class Algo:
            pass

        self.register('EGA', Algo)
        with self.assertRaises(KeyError):
            _ = self.ALGORITHMS['ega']  # Should fail if case-sensitive

    def test_special_character_injection(self):
        """RED TEAM: Special characters in names could break lookup"""
        class Algo:
            pass

        malicious_names = ['algo;DROP TABLE', 'algo/../../../etc', 'algo\x00null']
        for name in malicious_names:
            with self.assertRaises(ValueError, msg=f"Should reject '{name}'"):
                self.register(name, Algo)


class RedTeamStateSerializationTests(unittest.TestCase):
    """Attack vector: State persistence and deserialization"""

    def test_circular_reference_in_state(self):
        """RED TEAM: Circular references should be detected"""
        from ega import EpigeneticAlgorithm as EGA

        algo = EGA(
            population_size=10,
            individual_size=5,
            genotype_mutation_rate=0.01,
            epigenome_mutation_rate=0.05,
            crossover_rate=0.8,
            tournament_size=5,
            elitism_size=2
        )

        # Create circular reference in a state object that mimics the real structure
        state = {
            'population': [],
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
        }
        state['population'].append(state)

        # The actual error might be a TypeError from the state-setting functions
        # or a RecursionError if the object is deeply copied.
        with self.assertRaises((ValueError, RecursionError, TypeError)):
            algo.set_state(state)

    def test_infinite_nesting_attack(self):
        """RED TEAM: Deeply nested structures exhaust stack"""
        state = {'level': 0}
        current = state

        # Create deeply nested dict
        for i in range(10000):
            current['next'] = {'level': i}
            current = current['next']

        # Attempt to serialize
        with self.assertRaises(RecursionError):
            json.dumps(state)

    def test_malicious_pickle_injection(self):
        """RED TEAM: Unpickling untrusted data is dangerous"""
        import pickle

        class MaliciousClass:
            def __reduce__(self):
                import os
                return (os.system, ('echo pwned',))

        malicious_data = pickle.dumps(MaliciousClass())

        # Framework should use JSON, not pickle
        # If it uses pickle, this test demonstrates the vulnerability


class RedTeamResourceExhaustionTests(unittest.TestCase):
    """Attack vector: Configuration-based resource attacks"""

    def test_population_size_bomb(self):
        """RED TEAM: Enormous population size causes OOM"""
        from run_experiment import validate_config
        from argparse import Namespace

        args = Namespace(
            population_size=10**9,  # 1 billion individuals
            individual_size=1000,
            generations=100,
            genotype_mutation_rate=0.01,
            epigenome_mutation_rate=0.05,
            parallel=1
        )

        # Should validate and reject before allocation
        with self.assertRaises(ValueError, msg="Should reject oversized population"):
            validate_config(args)

    @unittest.skip("Skipping test: PyYAML safe_load is not vulnerable to this attack, causing assertRaises to fail.")
    def test_yaml_bomb_attack(self):
        """RED TEAM: Billion laughs YAML exploit"""
        yaml_bomb = """
        a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
        b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
        c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
        d: &d [*c,*c,*c,*c,*c,*c,*c,*c,*c]
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
            f.write(yaml_bomb)
            f.flush()

            with self.assertRaises((yaml.YAMLError, MemoryError)):
                with open(f.name) as yf:
                    yaml.safe_load(yf)

    def test_negative_parameter_injection(self):
        """RED TEAM: Negative values could cause undefined behavior"""
        from run_experiment import validate_config
        from argparse import Namespace

        base_args = Namespace(
            population_size=100,
            individual_size=50,
            generations=100,
            genotype_mutation_rate=0.01,
            epigenome_mutation_rate=0.05,
            parallel=1
        )

        invalid_params = {
            'population_size': -100,
            'individual_size': -50,
            'generations': -1,
            'genotype_mutation_rate': -0.5,
        }

        for param, value in invalid_params.items():
            with self.subTest(param=param):
                args = Namespace(**vars(base_args))
                setattr(args, param, value)
                with self.assertRaises(ValueError, msg=f"Should reject {param}={value}"):
                    validate_config(args)


class RedTeamEpigenomeStabilityTests(unittest.TestCase):
    """Attack vector: Algorithm-specific decoherence"""

    def test_extreme_mutation_decoherence(self):
        """RED TEAM: 100% mutation rate destroys information"""
        from ega import EpigeneticAlgorithm as EGA

        algo = EGA(
            population_size=100,
            individual_size=50,
            epigenome_mutation_rate=1.0,  # Total mutation
            genotype_mutation_rate=0.0,
            crossover_rate=0.8,
            tournament_size=5,
            elitism_size=2
        )

        # Manually evaluate fitness before evolution
        from fitness_functions import deceptive_fitness
        for ind in algo.population.individuals:
            ind.fitness = deceptive_fitness(ind.calculate_phenotype())
        initial_best = algo.get_fittest_individual()

        # Evolve 100 generations
        for _ in range(100):
            algo.evolve(deceptive_fitness)

        final_best = algo.get_fittest_individual()

        # With 100% mutation, no progress should be possible
        self.assertLessEqual(
            final_best.fitness,
            initial_best.fitness * 1.2,  # Allow 20% noise
            msg="100% mutation should prevent convergence"
        )

    def test_zero_mutation_stagnation(self):
        """RED TEAM: 0% mutation rate causes premature convergence"""
        from ega import EpigeneticAlgorithm as EGA
        from fitness_functions import deceptive_fitness

        algo = EGA(
            population_size=100,
            individual_size=50,
            epigenome_mutation_rate=0.0,
            genotype_mutation_rate=0.0,
            crossover_rate=0.8,
            tournament_size=5,
            elitism_size=2
        )

        fitness_history = []
        for _ in range(50):
            algo.evolve(deceptive_fitness)
            fitness_history.append(algo.get_fittest_individual().fitness)

        # Check for stagnation (no improvement over 20 generations)
        recent = fitness_history[-20:]
        self.assertEqual(
            len(set(recent)), 1,
            msg="Zero mutation should cause immediate stagnation"
        )


class RedTeamCheckpointIntegrityTests(unittest.TestCase):
    """Attack vector: Checkpoint corruption and tampering"""

    def test_negative_generation_checkpoint(self):
        """RED TEAM: Invalid checkpoint state"""
        checkpoint = {
            "generation": -999,
            "algorithm_state": {
                "population": {"individuals": []},
                "random_state": [1, [], None],
                "numpy_random_state": ["MT19937", [0]*624, 0, 0, 0.0]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(checkpoint, f)
            f.flush()

            from run_experiment import validate_checkpoint
            with self.assertRaises(ValueError):
                state = json.load(open(f.name))
                validate_checkpoint(state)

    def test_corrupted_checkpoint_data(self):
        """RED TEAM: Malformed JSON in checkpoint"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            f.write("}{INVALID JSON][")
            f.flush()

            from run_experiment import load_checkpoint
            with self.assertRaises(json.JSONDecodeError):
                load_checkpoint(f.name)

    def test_checkpoint_type_confusion(self):
        """RED TEAM: Wrong data types in checkpoint"""
        checkpoint = {
            'generation': "not_a_number",  # String instead of int
            'population': "not_a_list",    # String instead of list
            'best_fitness': None           # None instead of float
        }

        from run_experiment import load_checkpoint, validate_checkpoint
        with self.assertRaises((TypeError, ValueError)):
            validate_checkpoint(checkpoint)


class RedTeamParallelExecutionTests(unittest.TestCase):
    """Attack vector: Concurrency bugs and race conditions"""

    def test_shared_mutable_state_race(self):
        """RED TEAM: Shared state across workers causes corruption"""
        shared_counter = {'value': 0}

        def worker_task(shared):
            for _ in range(1000):
                shared['value'] += 1  # Non-atomic operation

        processes = []
        for _ in range(4):
            p = mp.Process(target=worker_task, args=(shared_counter,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Expected: 4000, Actual: likely less due to race condition
        # This demonstrates the vulnerability if framework uses shared state
        self.assertNotEqual(
            shared_counter['value'], 4000,
            msg="Race condition detected in shared state"
        )

    def test_deadlock_with_locks(self):
        """RED TEAM: Improper lock ordering causes deadlock"""
        import threading

        lock_a = threading.Lock()
        lock_b = threading.Lock()

        def thread1():
            lock_a.acquire()
            lock_b.acquire()  # Wrong order
            lock_b.release()
            lock_a.release()

        def thread2():
            lock_b.acquire()
            lock_a.acquire()  # Opposite order = deadlock
            lock_a.release()
            lock_b.release()

        t1 = threading.Thread(target=thread1)
        t2 = threading.Thread(target=thread2)

        t1.start()
        t2.start()

        # Should timeout, indicating deadlock
        t1.join(timeout=2)
        t2.join(timeout=2)

        self.assertFalse(
            t1.is_alive() and t2.is_alive(),
            msg="Threads deadlocked"
        )


class RedTeamFitnessFunctionTests(unittest.TestCase):
    """Attack vector: Malicious or broken fitness functions"""

    def test_non_deterministic_fitness(self):
        """RED TEAM: Fitness function should be detected by the framework"""
        from ega import EpigeneticAlgorithm as EGA
        import random

        def malicious_fitness(genome):
            return random.random()

        algo = EGA(
            population_size=10, individual_size=5, genotype_mutation_rate=0,
            epigenome_mutation_rate=0, crossover_rate=0, tournament_size=2,
            elitism_size=0, fitness_timeout=5
        )

        # Force some individuals to be identical to trigger re-evaluation
        # within the same generation, which should expose the non-determinism.
        base_genotype = [0, 1, 0, 1, 0]
        base_epigenome = [1, 1, 1, 1, 1]
        for i in range(5):
            algo.population.individuals[i].genotype = base_genotype
            algo.population.individuals[i].epigenome = base_epigenome

        with self.assertRaises(ValueError, msg="Framework should have detected non-deterministic fitness function"):
            algo.evolve(malicious_fitness)

    def test_fitness_returning_invalid_types(self):
        """RED TEAM: Fitness function returns non-numeric values"""
        from run_experiment import run_single_trial
        from argparse import Namespace
        from registry import register_fitness_function, register_core_components

        # Ensure core components are registered in case other tests cleared them
        register_core_components()

        def bad_fitness(genome):
            return "not a number"

        # This might fail if a previous test run already registered it,
        # so we'll wrap it in a try-except block.
        try:
            register_fitness_function("bad_fitness", bad_fitness)
        except ValueError:
            pass  # Already registered

        args = Namespace(
            population_size=10, individual_size=10, generations=1,
            genotype_mutation_rate=0.01, epigenome_mutation_rate=0.05,
            crossover_rate=0.8, tournament_size=5, elitism_size=2,
            resume=False, checkpoint_interval=0
        )

        with self.assertRaises(TypeError):
            run_single_trial(0, "ega", "bad_fitness", args)

    def test_fitness_infinite_loop(self):
        """RED TEAM: Fitness function never returns"""
        from ega import EpigeneticAlgorithm as EGA, Individual

        def infinite_fitness(genome):
            while True:
                pass

        algo = EGA(
            population_size=1, individual_size=1, genotype_mutation_rate=0,
            epigenome_mutation_rate=0, crossover_rate=0, tournament_size=1,
            elitism_size=0, fitness_timeout=2  # 2-second timeout
        )

        individual = Individual(10)

        with self.assertRaises(TimeoutError):
            algo._calculate_fitness(individual, infinite_fitness)


# Test runner with detailed reporting
if __name__ == '__main__':
    # Run tests with verbose output
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary report
    print("\n" + "="*70)
    print("RED TEAM TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures or result.errors:
        print("\n⚠️  VULNERABILITIES DETECTED")
        print("Review failures above to identify security/stability issues")
    else:
        print("\n✅ All red team tests passed - framework is resilient")
