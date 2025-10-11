def one_max_fitness(phenotype):
    """The fitness function for the One-Max problem.

    This is a classic toy problem in genetic algorithms. The fitness of an
    individual is simply the number of 1s in its phenotype. The goal is to
    evolve a bit string of all 1s.

    Args:
        phenotype (list[int]): The bit string to evaluate.

    Returns:
        int: The sum of the bits in the phenotype.
    """
    return sum(phenotype)


def deceptive_fitness(chromosome):
    """A deceptive fitness function with a local optimum (trap).

    This function is designed to test an algorithm's ability to escape
    local optima and maintain diversity. It has two main features:
    - A global optimum at a string of all 1s, which yields the highest score.
    - A deceptive local optimum at a string of all 0s, which is rewarded
      with a high score to "trap" simplistic hill-climbing algorithms.

    The fitness landscape guides the search toward the all-0s solution,
    but a much higher reward is hidden at the all-1s solution.

    Args:
        chromosome (list[int]): The bit string to evaluate.

    Returns:
        int: The fitness score.
    """
    # Global optimum (all 1s) gets the highest score
    if sum(chromosome) == len(chromosome):
        return len(chromosome) * 2

    # The deceptive trap: reward strings of 0s
    num_zeros = chromosome.count(0)

    # Local optimum (all 0s) gets a high score, but lower than the global optimum
    if num_zeros == len(chromosome):
        return len(chromosome)

    # Otherwise, the fitness is the number of zeros, guiding the search
    # toward the deceptive trap.
    return num_zeros
