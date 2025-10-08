import argparse
import multiprocessing as mp
import os
import random
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import yaml

from registry import ALGORITHMS, FITNESS_FUNCTIONS, register_core_components


def run_single_trial(trial_id, algorithm_name, fitness_function_name, args):
    """
    Runs a single instance of a genetic algorithm for a set number of generations.
    This function is designed to be called by a worker process.
    """
    # --- Process-Safe Seeding ---
    # Ensure each parallel trial has a unique random seed by combining
    # the time, its own process ID, and the trial ID.
    seed = int(time.time()) + os.getpid() + trial_id
    random.seed(seed)
    np.random.seed(seed)

    print(f"  Starting Trial {trial_id + 1} (PID: {os.getpid()})...")
    try:
        algorithm_class = ALGORITHMS[algorithm_name]
        fitness_function = FITNESS_FUNCTIONS[fitness_function_name]
    except KeyError as e:
        raise ValueError(f"Component not found in registry: {e}")

    # Pass all argparse arguments to the constructor.
    algorithm = algorithm_class(**vars(args))

    fitness_history = []
    for _ in range(args.generations):
        best_fitness, avg_fitness = algorithm.evolve(fitness_function)
        fitness_history.append((best_fitness, avg_fitness))

    print(f"  Finished Trial {trial_id + 1}.")
    return fitness_history


def main(args):
    """
    Main function to run and manage experiments.
    """
    print("--- Starting Comparative Experiment ---")
    print(f"Running {args.trials} trials for each setup...")

    # --- Define Experiments ---
    experiments = [
        {
            "label": "EGA (Deceptive)",
            "algorithm_name": "ega",
            "fitness_function_name": "deceptive",
        },
        {
            "label": "Standard GA (Deceptive)",
            "algorithm_name": "standard",
            "fitness_function_name": "deceptive",
        },
    ]

    aggregated_results = []

    for exp in experiments:
        print(f"\nRunning experiment: {exp['label']}")

        # Use functools.partial to create a worker function with fixed arguments
        worker_function = partial(
            run_single_trial,
            algorithm_name=exp["algorithm_name"],
            fitness_function_name=exp["fitness_function_name"],
            args=args,
        )

        if args.parallel > 1:
            print(f"  Running {args.trials} trials in parallel on {args.parallel} cores...")
            # Use a multiprocessing pool to run trials in parallel
            with mp.Pool(processes=args.parallel) as pool:
                all_trial_histories = pool.map(worker_function, range(args.trials))
        else:
            print(f"  Running {args.trials} trials in serial...")
            # Run trials sequentially for debugging or when parallelism is not desired
            all_trial_histories = [worker_function(i) for i in range(args.trials)]

        # Aggregate results using numpy
        mean_history = np.mean(all_trial_histories, axis=0).tolist()
        aggregated_results.append((exp["label"], mean_history))

    # Plot the aggregated results
    plot_comparative_history(
        aggregated_results,
        title="EGA vs. Standard GA on Deceptive Problem",
        output_file=args.output_file,
    )

    print("\n--- Experiment Complete ---")


def plot_comparative_history(results, title, output_file):
    """
    Plots the best and average fitness for multiple experimental runs.

    Args:
        results (list): A list of tuples, where each tuple is
                        (label, aggregated_fitness_history).
        title (str): The title for the plot.
        output_file (str): The filename to save the plot to.
    """
    plt.figure(figsize=(12, 8))

    for label, history in results:
        generations = range(len(history))
        best_fitness = [f[0] for f in history]
        avg_fitness = [f[1] for f in history]

        # Plot best fitness with a solid line
        (line,) = plt.plot(generations, best_fitness, label=f"{label} (Best)")
        # Plot average fitness with a dashed line of the same color
        plt.plot(
            generations,
            avg_fitness,
            linestyle="--",
            color=line.get_color(),
            label=f"{label} (Avg)",
        )

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness over Trials")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"\nComparative plot saved to {output_file}")


if __name__ == "__main__":
    # --- Argument Parsing with Config File Support ---
    parser = argparse.ArgumentParser(
        description="Run comparative experiments for Genetic Algorithms."
    )
    parser.add_argument("--config", type=str, help="Path to a YAML configuration file.")

    # First, parse just the config file argument
    config_args, remaining_argv = parser.parse_known_args()

    config = {}
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)

    # Now, create a new parser for all arguments
    parser = argparse.ArgumentParser(
        parents=[parser], add_help=False
    )  # Inherit --config
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of trials to run for each experiment."
    )
    parser.add_argument("--population_size", type=int, default=100, help="Size of the population.")
    parser.add_argument(
        "--individual_size",
        type=int,
        default=20,
        help="Size of the individual genome for deceptive problem.",
    )
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to run.")
    parser.add_argument(
        "--genotype_mutation_rate", type=float, default=0.01, help="Mutation rate for the genotype."
    )
    parser.add_argument(
        "--epigenome_mutation_rate", type=float, default=0.05, help="Mutation rate for the epigenome."
    )
    parser.add_argument(
        "--mutation_rate", type=float, default=0.01, help="Mutation rate for the standard GA."
    )
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="Crossover rate.")
    parser.add_argument(
        "--tournament_size", type=int, default=5, help="Size of the selection tournament."
    )
    parser.add_argument(
        "--elitism_size", type=int, default=2, help="Number of elite individuals to carry over."
    )
    parser.add_argument(
        "--output_file", type=str, default='comparative_plot.png', help="File to save the final comparison plot to."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of cores to use for parallel execution. Defaults to 1 (serial)."
    )

    # Set defaults from config file, then parse the remaining args
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)

    main(args)
