import random
import tsplib95
import matplotlib.pyplot as plt

#Load the dataset, I chose to use the library dataset from here: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/

cities = tsplib95.load('ulysses16.tsp')

#print(cities)

#function to parse coordinates from the data format
def parse_coordinates(data):
    coordinates ={}
    parsing = False

    for line in data:
        if line.startswith("NODE_COORD_SECTION"):
            parsing = True
        elif line.startswith("EOF"):
            parsing = False
        elif parsing:
            parts = line.split()
            node = int(parts[0])
            x, y = map(float, parts[1:])
            coordinates[node] = (x, y)

    return coordinates

# Parse coordinates from the data
coordinates =parse_coordinates(cities.node_coords)

# Function to calculate the total distance of a route
def calculate_total_distance(route, coordinates):
    total_distance = 0
    for i in range(len(route) - 1):
        city1, city2 = route[i], route[i + 1]
        x1, y1 = coordinates[city1]
        x2, y2 = coordinates[city2]
        total_distance += ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    total_distance += ((coordinates[route[-1]][0] - coordinates[route[0]][0])**2 +
                       (coordinates[route[-1]][1] - coordinates[route[0]][1])**2)**0.5
    return total_distance
    

# Function to calculate the total distance of a route
def total_distance_calculation(route, get_weight_fn):
    total_distance =0
    for i in range (len(route)-1):
        total_distance += get_weight_fn(route[i],route[i+1])
    total_distance +=get_weight_fn(route[-1], route[0]) # this line of code returns the starting city number
    return total_distance

# Selection Techniques

## Roulette Wheel technique

def roulette_wheel_selection(population, fitness_value):
    total_fitness = sum(fitness_values)
    selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_index = random.choices(range(len(population)), weights=selection_probabilities)[0]
    return population[selected_index]

## Tournament technique

def tournament_selection(population, fitness_values, tournament_size=5):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    selected_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[selected_index]
