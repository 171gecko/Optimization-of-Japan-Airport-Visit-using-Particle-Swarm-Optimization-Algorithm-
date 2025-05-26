import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Seed for reproducibility
random.seed(100)

# Load airport data from CSV
airports = pd.read_csv("/content/airports.csv")

# Define Particle class
class Particle:
    def __init__(self, airports, velocity_limit):
        self.airports = airports
        self.position = list(range(len(airports)))
        random.shuffle(self.position)
        self.velocity = [0] * len(self.position)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.velocity_limit = velocity_limit

    def distance(self, airport1, airport2):
        lat1, lon1 = radians(airport1['Lat']), radians(airport1['Long'])
        lat2, lon2 = radians(airport2['Lat']), radians(airport2['Long'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        radius = 6371.0  # Earth radius in kilometers
        return radius * c

    def fitness(self):
        fitness = 0
        for i in range(len(self.position) - 1):
            airport1, airport2 = self.airports.iloc[self.position[i]], self.airports.iloc[self.position[i + 1]]
            fitness += self.distance(airport1, airport2)
        fitness += self.distance(self.airports.iloc[self.position[-1]], self.airports.iloc[self.position[0]])
        return fitness

# Particle Swarm Optimization function
def particle_swarm_optimization(airports, particle_count, iterations, w, c1, c2, velocity_limit, iteration_limit):
    particles = [Particle(airports, velocity_limit) for _ in range(particle_count)]
    global_best_position = None
    global_best_fitness = float('inf')
    all_fitness_values = []
    iteration_counter = 0

    for iteration in range(iterations):
        for particle in particles:
            current_fitness = particle.fitness()
            if current_fitness < particle.best_fitness:
                particle.best_fitness = current_fitness
                particle.best_position = particle.position.copy()
            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle.position.copy()

        for particle in particles:
            visited_airports = set()
            for i in range(len(particle.position)):
                r1, r2 = random.random(), random.random()
                particle.velocity[i] = (
                    w * particle.velocity[i]
                    + c1 * r1 * (particle.best_position[i] - particle.position[i])
                    + c2 * r2 * (global_best_position[i] - particle.position[i])
                )
                particle.velocity[i] = max(-velocity_limit, min(particle.velocity[i], velocity_limit))
                proposed_position = limit_check(particle.position[i] + int(particle.velocity[i]), len(airports))
                while proposed_position in visited_airports:
                    proposed_position = random.randint(0, len(airports) - 1)
                visited_airports.add(proposed_position)
                particle.position[i] = proposed_position

        all_fitness_values.append(global_best_fitness)

        if iteration > 0 and all_fitness_values[-1] == all_fitness_values[-2]:
            iteration_counter += 1
        else:
            iteration_counter = 0
        if iteration_counter >= iteration_limit:
            break

    return global_best_position, global_best_fitness, all_fitness_values

# Helper function to check position limits
def limit_check(coord, max_value):
    return max(0, min(coord, max_value - 1))

# Test PSO with different configurations
def test_pso_performance(airports):
    configurations = [
        (20, 5000, 500),
        (30, 4000, 400),
        (40, 3000, 300),
        (50, 2000, 200),
        (60, 1000, 100),
        (70, 6000, 600),
    ]

    results = []
    for particle_count, iterations, iteration_limit in configurations:
        print(f"Testing with particle_count={particle_count}, iterations={iterations}, iteration_limit={iteration_limit}")
        best_position, best_fitness, all_fitness_values = particle_swarm_optimization(
            airports, particle_count, iterations, w=1, c1=1, c2=2, velocity_limit=1234.8, iteration_limit=iteration_limit
        )
        results.append((particle_count, iterations, iteration_limit, best_fitness))
        print(f"Best Fitness: {best_fitness:.2f}\n")

    return results

# Run tests
results = test_pso_performance(airports)
for result in results:
    print(f"Configuration: particle_count={result[0]}, iterations={result[1]}, iteration_limit={result[2]} -> Best Fitness: {result[3]:.2f}")
