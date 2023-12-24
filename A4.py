import random
from deap import base, creator, tools

# Define the TSP problem
class TSPProblem:
    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances

    def evaluate(self, individual):
        total_distance = 0
        for i in range(len(individual) - 1):
            total_distance += self.distances[individual[i]][individual[i + 1]]
        total_distance += self.distances[individual[-1]][individual[0]]  # Return to the origin
        return total_distance,

# Create DEAP classes for individuals and populations
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize DEAP toolbox
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(cities)), len(cities))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create TSP problem instance
cities = ["CityA", "CityB", "CityC", "CityD"]
distances = [[0, 2, 3, 1],
             [2, 0, 4, 5],
             [3, 4, 0, 6],
             [1, 5, 6, 0]]
tsp_problem = TSPProblem(cities, distances)

# Evaluation function
toolbox.register("evaluate", tsp_problem.evaluate)

# Genetic Algorithm
def main():
    population_size = 50
    generations = 100

    # Create an initial population
    population = toolbox.population(n=population_size)

    for gen in range(generations):
        # Evaluate the entire population
        fitness_values = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitness_values):
            ind.fitness.values = fit

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate offspring fitness
        fitness_values = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitness_values):
            ind.fitness.values = fit

        # Replace the old population by the offspring
        population[:] = offspring

        # Get the best individual in the current generation
        best_ind = tools.selBest(population, 1)[0]
        print(f"Generation {gen + 1}, Best Distance: {best_ind.fitness.values[0]}")

    # Get the final best individual
    best_individual = tools.selBest(population, 1)[0]
    print("\nBest Route:", best_individual)

if __name__ == "__main__":
    main()
