import argparse
import matplotlib.pyplot as plt
from ega import EpigeneticAlgorithm

def plot_fitness_history(fitness_history, output_file="fitness_plot.png"):
    """
    Plots the best and average fitness over generations and saves it to a file.
    """
    generations = range(len(fitness_history))
    best_fitness = [f[0] for f in fitness_history]
    avg_fitness = [f[1] for f in fitness_history]

    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, label="Best Fitness")
    plt.plot(generations, avg_fitness, label="Average Fitness")
    plt.title("EGA Performance: Fitness Over Generations")
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

def main(args):
    """
    Main function to run the Epigenetic Genetic Algorithm.
    """
    # Initialize the algorithm with parameters from argparse
    ega = EpigeneticAlgorithm(
        population_size=args.population_size,
        individual_size=args.individual_size,
        genotype_mutation_rate=args.genotype_mutation_rate,
        epigenome_mutation_rate=args.epigenome_mutation_rate,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
        elitism_size=args.elitism_size
    )

    print("--- Starting Epigenetic Genetic Algorithm for the One-Max Problem ---")
    print(f"Configuration: {vars(args)}")
    print(f"Target fitness: {args.individual_size}\n")

    fitness_history = []

    # Evolution loop
    for generation in range(args.generations):
        # Evolve the population and get stats
        best_fitness, avg_fitness = ega.evolve(one_max_fitness)
        fitness_history.append((best_fitness, avg_fitness))

        print(f"Generation {generation + 1}/{args.generations} | "
              f"Best Fitness: {best_fitness:.2f} | "
              f"Avg Fitness: {avg_fitness:.2f}")

        # Check for solution
        if best_fitness == args.individual_size:
            print("\n--- Solution Found! ---")
            break

    # Get the final best individual
    final_fittest = ega.population.get_fittest()
    print("\n--- Final Best Individual ---")
    print(final_fittest)

    # Plot the fitness history
    plot_fitness_history(fitness_history)

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

    args = parser.parse_args()
    main(args)