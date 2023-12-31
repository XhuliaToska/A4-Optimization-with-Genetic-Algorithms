import random
import tsplib95
import matplotlib.pyplot as plt

#Load the dataset, I chose to use the library dataset from here: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/

cities = tsplib95.load('ulysses16.tsp')

#print(cities)

# Function to calculate the total distance of a route

def total_distance_calculation(route, get_weight_fn):
    total_distance =0
    for i in range (len(route)-1):
        total_distance += get_weight_fn(route[i],route[i+1])
    total_distance +=get_weight_fn(route[-1], route[0]) # this line of code returns the starting city number
    return total_distance
