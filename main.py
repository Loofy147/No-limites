import argparse
import matplotlib.pyplot as plt
from ega import EpigeneticAlgorithm
from standard_ga import StandardAlgorithm

def plot_fitness_history(results, title, output_file="fitness_plot.png"):
    """
    Plots the best and average fitness for one or more runs over generations.

    Args:
        results (list): A list of tuples, where each tuple is
                        (label, fitness_history).
        title (str): The title for the plot.
        output_file (str): The filename to save the plot to.
    """
    plt.figure(figsize=(12, 6))

    for label, history in results:
        generations = range(len(history))
        best_fitness = [f[0] for f in history]
        avg_fitness = [f[1] for f in history]

        # Plot best fitness with a solid line
        line, = plt.plot(generations, best_fitness, label=f"{label} (Best)")
        # Plot average fitness with a dashed line of the same color
        plt.plot(generations, avg_fitness, linestyle='--', color=line.get_color(), label=f"{label} (Avg)")

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"\nFitness plot saved to {output_file}")

def one_max_fitness(phenotype):
    """
    The fitness function for the One-Max problem.
    Calculates the sum of the bits in the phenotype.
    """
    return sum(phenotype)

def deceptive_fitness(chromosome):
    """
    A deceptive fitness function with a local optimum (trap).

    - The global optimum is a string of all 1s, which gives the highest fitness.
    - A local optimum is a string of all 0s, which gives a high, but not the highest, fitness.
    - Otherwise, fitness is the count of 0s, deceptively guiding the search
      towards the local optimum.
    """
    # Global optimum (all 1s) gets the highest score
    if sum(chromosome) == len(chromosome):
        return len(chromosome) * 2

    # The deceptive trap: reward strings of 0s
    num_zeros = chromosome.count(0)

    # Local optimum (all 0s) gets a high score, but lower than the global optimum
    if num_zeros == len(chromosome):
        return len(chromosome)

    return num_zeros

def main(args):
    """
    Main function to run the selected Genetic Algorithm.
    """
    # --- Select Fitness Function ---
    if args.fitness_func == 'onemax':
        fitness_function = one_max_fitness
        target_fitness = args.individual_size
    elif args.fitness_func == 'deceptive':
        fitness_function = deceptive_fitness
        target_fitness = args.individual_size * 2
    else:
        raise ValueError("Invalid fitness function specified.")

    # --- Initialize Algorithm ---
    if args.algorithm == 'ega':
        algorithm = EpigeneticAlgorithm(
            population_size=args.population_size,
            individual_size=args.individual_size,
            genotype_mutation_rate=args.genotype_mutation_rate,
            epigenome_mutation_rate=args.epigenome_mutation_rate,
            crossover_rate=args.crossover_rate,
            tournament_size=args.tournament_size,
            elitism_size=args.elitism_size
        )
    elif args.algorithm == 'standard':
        algorithm = StandardAlgorithm(
            population_size=args.population_size,
            individual_size=args.individual_size,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            tournament_size=args.tournament_size,
            elitism_size=args.elitism_size
        )
    else:
        raise ValueError("Invalid algorithm specified.")

    print(f"--- Running {args.algorithm.upper()} with {args.fitness_func.upper()} function ---")
    print(f"Configuration: {vars(args)}")
    print(f"Target fitness: {target_fitness}\n")

    fitness_history = []

    # --- Evolution Loop ---
    for generation in range(args.generations):
        best_fitness, avg_fitness = algorithm.evolve(fitness_function)
        fitness_history.append((best_fitness, avg_fitness))

        print(f"Generation {generation + 1}/{args.generations} | "
              f"Best Fitness: {best_fitness:.2f} | "
              f"Avg Fitness: {avg_fitness:.2f}")

        # Check for solution
        if best_fitness >= target_fitness:
            print("\n--- Global Optimum Found! ---")
            break

    # Get the final best individual
    final_fittest = algorithm.population.get_fittest()
    print("\n--- Final Best Individual ---")
    print(final_fittest)

    # Plot the fitness history
    title = f"{args.algorithm.upper()} on {args.fitness_func.upper()} Problem"
    results = [(f"{args.algorithm.upper()}", fitness_history)]
    plot_fitness_history(results, title, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Epigenetic Genetic Algorithm.")
    parser.add_argument('--population_size', type=int, default=100, help='Size of the population.')
    parser.add_argument('--individual_size', type=int, default=50, help='Size of the individual genome.')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations to run.')
    parser.add_argument('--genotype_mutation_rate', type=float, default=0.01, help='Mutation rate for the genotype.')
    parser.add_argument('--epigenome_mutation_rate', type=float, default=0.05, help='Mutation rate for the epigenome.')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='Crossover rate.')
    parser.add_argument('--tournament_size', type=int, default=5, help='Size of the selection tournament.')
    parser.add_argument('--elitism_size', type=int, default=2, help='Number of elite individuals to carry over.')
    parser.add_argument('--output_file', type=str, default='fitness_plot.png', help='File to save the fitness plot to.')

    # Arguments for comparative analysis
    parser.add_argument('--algorithm', type=str, default='ega', choices=['ega', 'standard'], help='The algorithm to run.')
    parser.add_argument('--fitness_func', type=str, default='onemax', choices=['onemax', 'deceptive'], help='The fitness function to use.')
    parser.add_argument('--mutation_rate', type=float, default=0.01, help='Mutation rate for the standard GA.')


    args = parser.parse_args()
    main(args)