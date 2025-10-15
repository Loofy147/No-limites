import argparse
import matplotlib.pyplot as plt
from registry import ALGORITHMS, FITNESS_FUNCTIONS


def plot_fitness_history(results, title, output_file="fitness_plot.png"):
    """Plots the best and average fitness over generations and saves the plot.

    This function generates a plot visualizing the performance of one or more
    evolutionary algorithm runs. It plots the best fitness and average fitness
    per generation, making it easy to see the progress of the search.

    Args:
        results (list[tuple[str, list[tuple[float, float]]]]): A list where
            each item is a tuple containing a label for the run (e.g.,
            "EGA") and its corresponding fitness history. The history is a
            list of (best_fitness, average_fitness) tuples for each
            generation.
        title (str): The title to display on the plot.
        output_file (str, optional): The path to save the generated plot image.
            Defaults to "fitness_plot.png".
    """
    plt.figure(figsize=(12, 6))

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
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"\nFitness plot saved to {output_file}")


def main(args):
    """Main function to run a single genetic algorithm experiment.

    This function orchestrates a single run of a selected algorithm on a
    chosen fitness function. It handles component loading, algorithm
    initialization, the main evolution loop, and plotting the final results.

    Args:
        args (argparse.Namespace): An object containing the parsed
            command-line arguments that configure the experiment.
    """
    # --- Select Components from Registry ---
    try:
        fitness_function = FITNESS_FUNCTIONS[args.fitness_func]
        algorithm_class = ALGORITHMS[args.algorithm]
    except KeyError as e:
        raise ValueError(f"Component not found in registry: {e}")

    # --- Determine Target Fitness ---
    if args.fitness_func == "onemax":
        target_fitness = args.individual_size
    elif args.fitness_func == "deceptive":
        target_fitness = args.individual_size * 2
    else:
        # This case is for future, more complex fitness functions
        target_fitness = float("inf")

    # --- Initialize Algorithm ---
    # Pass all argparse arguments to the constructor.
    # The algorithm's __init__ will pick the ones it needs.
    algorithm = algorithm_class(**vars(args))

    print(
        f"--- Running {args.algorithm.upper()} with "
        f"{args.fitness_func.upper()} function ---"
    )
    print(f"Configuration: {vars(args)}")
    print(f"Target fitness: {target_fitness}\n")

    fitness_history = []

    # --- Evolution Loop ---
    for generation in range(args.generations):
        best_fitness, avg_fitness = algorithm.evolve(fitness_function)
        fitness_history.append((best_fitness, avg_fitness))

        print(
            f"Generation {generation + 1}/{args.generations} | "
            f"Best Fitness: {best_fitness:.2f} | "
            f"Avg Fitness: {avg_fitness:.2f}"
        )

        # Check for solution
        if best_fitness >= target_fitness:
            print("\n--- Global Optimum Found! ---")
            break

    # Get the final best individual
    final_fittest = algorithm.get_fittest_individual()
    print("\n--- Final Best Individual ---")
    print(final_fittest)

    # Plot the fitness history
    title = f"{args.algorithm.upper()} on {args.fitness_func.upper()} Problem"
    results = [(f"{args.algorithm.upper()}", fitness_history)]
    plot_fitness_history(results, title, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Epigenetic Genetic Algorithm."
    )
    parser.add_argument(
        "--population_size", type=int, default=100, help="Size of the population."
    )
    parser.add_argument(
        "--individual_size", type=int, default=50, help="Size of the individual genome."
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
        default="fitness_plot.png",
        help="File to save the fitness plot to.",
    )

    # Arguments for comparative analysis
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ega",
        choices=ALGORITHMS.keys(),
        help="The algorithm to run.",
    )
    parser.add_argument(
        "--fitness_func",
        type=str,
        default="onemax",
        choices=FITNESS_FUNCTIONS.keys(),
        help="The fitness function to use.",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.01,
        help="Mutation rate for the standard GA.",
    )

    args = parser.parse_args()
    main(args)
