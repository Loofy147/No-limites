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

## Installation

To run the project, first install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## How to Run the Example

This repository includes a simple example that uses the EGA to solve the classic "One-Max" problem.

To run the example with default parameters, execute the `main.py` script:

```bash
python3 main.py
```

You can also customize the algorithm's parameters using command-line arguments. For a full list of options, run:

```bash
python3 main.py --help
```

Example of a custom run:
```bash
python3 main.py --generations 200 --population_size 50 --epigenome_mutation_rate 0.1
```

## Output

After each run, the script will:
1.  Print the best and average fitness for each generation to the console.
2.  Display the genotype, epigenome, and final fitness of the best individual found.
3.  Generate a plot named `fitness_plot.png` that visually represents the best and average fitness over the generations.