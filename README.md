# Epigenetic Genetic Algorithm (EGA)

This project introduces a novel concept in evolutionary computation: the **Epigenetic Genetic Algorithm (EGA)**. It is inspired by the biological process of epigenetics, where gene expression is controlled by mechanisms other than changes to the DNA sequence itself.

## The Concept

Traditional Genetic Algorithms (GAs) can sometimes suffer from premature convergence, where the population gets stuck in a local optimum, losing the genetic diversity needed to find the global optimum.

The EGA addresses this by introducing a two-tiered system for each individual:

1.  **Genotype:** This is the core, underlying genetic code, similar to a standard GA. It evolves at a slow, stable rate, preserving good genetic material over time.
2.  **Epigenome:** This is a secondary layer of information that acts as a "mask" or "switch" for the genotype. It controls which genes in the genotype are actually *expressed* (i.e., contribute to the phenotype). The epigenome evolves at a much higher mutation rate.

This dual system allows for a powerful balance between **exploration** and **exploitation**:
- The rapidly changing **epigenome** allows an individual to quickly test new combinations of expressed genes and adapt to the fitness landscape without altering its core genetic code.
- The slowly changing **genotype** preserves strong genetic building blocks, preventing the loss of good solutions.

## How It Works

In this implementation, the phenotype (the solution that is evaluated by the fitness function) is calculated by performing a bitwise AND operation between the genotype and the epigenome.

- **Genotype Mutation Rate:** Kept low to ensure stability.
- **Epigenome Mutation Rate:** Kept high to encourage rapid adaptation and exploration.
- **Elitism:** The best individuals from one generation are automatically carried over to the next, ensuring that the best-found solutions are never lost.

This allows the algorithm to "test" turning genes on or off via the epigenome before committing to a change in the underlying genotype.

## Comparative Analysis

To demonstrate the capabilities of the Epigenetic Genetic Algorithm (EGA), this project now includes a framework for comparative analysis. You can run the EGA against a **Standard Genetic Algorithm (SGA)** on different types of problems.

### Fitness Functions

Two fitness functions are available to test the algorithms:

1.  **One-Max (`onemax`):** A classic, simple optimization problem where the goal is to evolve a binary string of all 1s. This is useful for baseline performance testing.
2.  **Deceptive Function (`deceptive`):** A more challenging problem designed with a "trap" local optimum. It rewards strings of all 0s, but the true global optimum is a string of all 1s, which receives a much higher score. This function is designed to test an algorithm's ability to maintain diversity and escape local optima.

## Installation

To run the project, first install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## How to Run Experiments

The `main.py` script is now a flexible runner that allows you to configure and run experiments.

To run an experiment with default parameters (EGA on One-Max), execute:

```bash
python3 main.py
```

You can also customize the algorithm's parameters using command-line arguments. For a full list of options, run:

```bash
python3 main.py --help
```

Example of a custom run:
```bash
python3 main.py --algorithm standard --fitness_func deceptive --generations 200 --output_file standard_vs_deceptive.png
```

## Output

After each run, the script will:
1.  Print the best and average fitness for each generation to the console.
2.  Display the genotype, epigenome, and final fitness of the best individual found.
3.  Generate a plot named `fitness_plot.png` that visually represents the best and average fitness over the generations.