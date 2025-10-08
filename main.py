from ega import EpigeneticAlgorithm

# --- Parameters ---
POPULATION_SIZE = 100
INDIVIDUAL_SIZE = 50
GENERATIONS = 100
# A lower, stable mutation rate for the core genetic code
GENOTYPE_MUTATION_RATE = 0.01
# A higher, more adaptive mutation rate for gene expression
EPIGENOME_MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
# Number of top individuals to carry over to the next generation
ELITISM_SIZE = 2

def one_max_fitness(phenotype):
    """
    The fitness function for the One-Max problem.
    Calculates the sum of the bits in the phenotype.
    """
    return sum(phenotype)

def main():
    """
    Main function to run the Epigenetic Genetic Algorithm.
    """
    # Initialize the algorithm with our parameters
    ega = EpigeneticAlgorithm(
        population_size=POPULATION_SIZE,
        individual_size=INDIVIDUAL_SIZE,
        genotype_mutation_rate=GENOTYPE_MUTATION_RATE,
        epigenome_mutation_rate=EPIGENOME_MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE,
        elitism_size=ELITISM_SIZE
    )

    print("--- Starting Epigenetic Genetic Algorithm for the One-Max Problem ---")
    print(f"Target fitness: {INDIVIDUAL_SIZE}\n")

    # Evolution loop
    for generation in range(GENERATIONS):
        # Evolve the population for one generation
        ega.evolve(one_max_fitness)

        # Recalculate fitness for the new population to find the best
        for ind in ega.population.individuals:
            phenotype = ind.calculate_phenotype()
            ind.fitness = one_max_fitness(phenotype)

        fittest_individual = ega.population.get_fittest()

        print(f"Generation {generation + 1}/{GENERATIONS} | "
              f"Best Fitness: {fittest_individual.fitness}")

        # Check for solution
        if fittest_individual.fitness == INDIVIDUAL_SIZE:
            print("\n--- Solution Found! ---")
            break

    # Get the final best individual
    final_fittest = ega.population.get_fittest()
    print("\n--- Final Best Individual ---")
    print(final_fittest)

if __name__ == "__main__":
    main()