import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import yaml

from registry import ALGORITHMS, FITNESS_FUNCTIONS


def load_checkpoint(filename):
    """Loads a checkpoint from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def validate_checkpoint(state):
    """Validates the structure and types of the loaded checkpoint state."""
    required_keys = {
        "generation": int,
        "algorithm_state": dict,
    }
    for key, expected_type in required_keys.items():
        if key not in state:
            raise ValueError(f"Checkpoint is missing required key: '{key}'")
        if not isinstance(state[key], expected_type):
            raise TypeError(
                f"Checkpoint key '{key}' has wrong type: "
                f"expected {expected_type}, got {type(state[key])}"
            )
    if state["generation"] < 0:
        raise ValueError("Checkpoint generation cannot be negative.")
    return True


def run_single_trial(trial_id, algorithm_name, fitness_function_name, args):
    """Runs a single trial of a genetic algorithm experiment."""
    seed = int(time.time()) + os.getpid() + trial_id
    random.seed(seed)
    np.random.seed(seed)

    print(f"  Starting Trial {trial_id + 1} (PID: {os.getpid()})...")
    try:
        algorithm_class = ALGORITHMS[algorithm_name]
        fitness_function = FITNESS_FUNCTIONS[fitness_function_name]
    except KeyError as e:
        raise ValueError(f"Component not found in registry: {e}")

    algorithm = algorithm_class(**vars(args))
    start_generation = 0
    fitness_history = []

    checkpoint_filename = (
        f"checkpoint_{algorithm_name}_{fitness_function_name}_trial_{trial_id}.json"
    )
    if args.resume and os.path.exists(checkpoint_filename):
        print(f"    Resuming from checkpoint: {checkpoint_filename}")
        state = load_checkpoint(checkpoint_filename)
        validate_checkpoint(state)

        # --- Parameter Validation for Resume ---
        original_args_dict = state["args"]
        critical_params = [
            "population_size", "individual_size", "genotype_mutation_rate",
            "epigenome_mutation_rate", "mutation_rate", "crossover_rate",
            "tournament_size", "elitism_size", "stagnation_limit",
            "adaptation_factor",
        ]
        for param in critical_params:
            if param in original_args_dict and hasattr(args, param):
                original_val = original_args_dict[param]
                current_val = getattr(args, param)
                if original_val != current_val:
                    raise ValueError(
                        f"Mismatched critical parameter '{param}' on resume. "
                        f"Checkpoint had {original_val}, but current run has "
                        f"{current_val}."
                    )

        algorithm.set_state(state["algorithm_state"])
        start_generation = state["generation"]
        fitness_history = state["fitness_history"]

    for generation in range(start_generation, args.generations):
        best_fitness, avg_fitness = algorithm.evolve(fitness_function)
        fitness_history.append((best_fitness, avg_fitness))

        if (
            args.checkpoint_interval > 0
            and (generation + 1) % args.checkpoint_interval == 0
        ):
            state = {
                "trial_id": trial_id,
                "generation": generation + 1,
                "algorithm_state": algorithm.get_state(),
                "fitness_history": fitness_history,
                "args": vars(args),
            }
            with open(checkpoint_filename, "w") as f:
                json.dump(state, f, indent=4)
            print(
                f"    ... Saved checkpoint for Trial {trial_id + 1} at "
                f"generation {generation + 1}"
            )

    print(f"  Finished Trial {trial_id + 1}.")
    return fitness_history


def validate_config(args):
    """Validates configuration parameters to prevent resource exhaustion.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Raises:
        ValueError: If a parameter is outside its acceptable range.
    """
    MAX_POPULATION_SIZE = 10**6  # Set a reasonable limit
    if not 0 < args.population_size <= MAX_POPULATION_SIZE:
        raise ValueError(
            f"Population size must be between 1 and {MAX_POPULATION_SIZE}."
        )

    if args.individual_size <= 0:
        raise ValueError("Individual size must be positive.")
    if args.generations <= 0:
        raise ValueError("Number of generations must be positive.")
    if not 0 <= args.genotype_mutation_rate <= 1:
        raise ValueError("Genotype mutation rate must be between 0 and 1.")
    if not 0 <= args.epigenome_mutation_rate <= 1:
        raise ValueError("Epigenome mutation rate must be between 0 and 1.")
    if args.parallel > mp.cpu_count():
        print(
            f"Warning: Number of parallel workers ({args.parallel}) "
            f"exceeds CPU count ({mp.cpu_count()})."
        )


def main(args):
    """Main function to run and manage comparative experiments.

    This function orchestrates a series of experiments as defined in the
    `experiments` list. It runs multiple trials for each experiment,
    handles parallel execution, aggregates the results, plots a comparative
    graph, and saves the structured results to a JSON file.

    Args:
        args (argparse.Namespace): An object containing the parsed
            command-line arguments and configuration from the YAML file.
    """
    validate_config(args)  # Validate the configuration first

    print("--- Starting Comparative Experiment ---")
    print(f"Running {args.trials} trials for each setup...")

    # --- Define Experiments ---
    experiments = [
        {
            "label": "Adaptive EGA (Deceptive)",
            "algorithm_name": "adaptive_ega",
            "fitness_function_name": "deceptive",
        },
        {
            "label": "Standard EGA (Deceptive)",
            "algorithm_name": "ega",
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
            print(
                f"  Running {args.trials} trials in parallel on "
                f"{args.parallel} cores..."
            )
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

    # --- Save Structured Results ---
    results_data = {
        "config": vars(args),
        "results": aggregated_results,
    }
    with open(args.results_file, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"Structured results saved to {args.results_file}")

    print("\n--- Experiment Complete ---")


def plot_comparative_history(results, title, output_file):
    """Plots and saves a comparative graph of aggregated fitness histories.

    Args:
        results (list[tuple[str, list[tuple[float, float]]]]): A list of
            tuples, where each tuple contains a label for an experiment and
            its aggregated fitness history (mean over trials).
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
        "--trials",
        type=int,
        default=10,
        help="Number of trials to run for each experiment.",
    )
    parser.add_argument(
        "--population_size", type=int, default=100, help="Size of the population."
    )
    parser.add_argument(
        "--individual_size",
        type=int,
        default=20,
        help="Size of the individual genome for deceptive problem.",
    )
    parser.add_argument(
        "--generations", type=int, default=100, help="Number of generations to run."
    )
    parser.add_argument(
        "--genotype_mutation_rate",
        type=float,
        default=0.01,
        help="Mutation rate for the genotype.",
    )
    parser.add_argument(
        "--epigenome_mutation_rate",
        type=float,
        default=0.05,
        help="Mutation rate for the epigenome.",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.01,
        help="Mutation rate for the standard GA.",
    )
    parser.add_argument(
        "--crossover_rate", type=float, default=0.8, help="Crossover rate."
    )
    parser.add_argument(
        "--tournament_size",
        type=int,
        default=5,
        help="Size of the selection tournament.",
    )
    parser.add_argument(
        "--elitism_size",
        type=int,
        default=2,
        help="Number of elite individuals to carry over.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="comparative_plot.png",
        help="File to save the final comparison plot to.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of cores to use for parallel execution. Defaults to 1 (serial).",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=0,
        help="Save a checkpoint every N generations. 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="checkpoint.pkl",
        help="Path to save checkpoint file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume experiment from the last checkpoint.",
    )
    # --- Adaptive Algorithm Parameters ---
    parser.add_argument(
        "--stagnation_limit",
        type=int,
        default=15,
        help="Generations to wait before adapting.",
    )
    parser.add_argument(
        "--adaptation_factor",
        type=float,
        default=1.5,
        help="Factor to increase mutation rate by.",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="experiment_results.json",
        help="File to save the structured results to.",
    )

    # Set defaults from config file, then parse the remaining args
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)

    main(args)
