def one_max_fitness(phenotype):
    """
    The fitness function for the One-Max problem.
    Calculates the sum of the bits in the phenotype.
    """
    return sum(phenotype)


def deceptive_fitness(chromosome):
    """
    A deceptive fitness function with a local optimum (trap).

    - The global optimum is a string of all 1s, which gives the highest
      fitness.
    - A local optimum is a string of all 0s, which gives a high, but not
      the highest, fitness.
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
