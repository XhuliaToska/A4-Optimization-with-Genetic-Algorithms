import random
import tsplib95
import matplotlib.pyplot as plt

#Load the dataset, I chose to use the library dataset from here: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/

cities = tsplib95.load('ulysses16.tsp')

print(cities)