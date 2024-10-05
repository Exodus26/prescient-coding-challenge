import numpy as np
import random

# Parameters
BIT_LENGTH = 10  # Length of each agent's bit string
POPULATION_SIZE = 20  # Number of agents in the population
MUTATION_RATE = 0.01  # Probability of mutation for each bit
GENERATIONS = 100  # Number of generations to evolve
TARGET = np.random.randint(2, size=BIT_LENGTH)  # Target bit string (randomly generated)

# Agent class representing an individual in the population
class Agent:
    def __init__(self):
        # Initialize agent with a random bit string
        self.chromosome = np.random.randint(2, size=BIT_LENGTH)
        self.fitness = 0

    # Calculate the fitness as the number of matching bits with the target
    def calculate_fitness(self):
        self.fitness = np.sum(self.chromosome == TARGET)

# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self):
        # Initialize population
        self.population = [Agent() for _ in range(POPULATION_SIZE)]

    # Evaluate the fitness of all agents
    def evaluate_population(self):
        for agent in self.population:
            agent.calculate_fitness()

    # Select two parents using tournament selection
    def select_parents(self):
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        parent1 = max(tournament, key=lambda agent: agent.fitness)
        parent2 = max(random.sample(self.population, tournament_size), key=lambda agent: agent.fitness)
        return parent1, parent2

    # Perform crossover between two parents to produce an offspring
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(0, BIT_LENGTH)
        offspring = Agent()
        offspring.chromosome[:crossover_point] = parent1.chromosome[:crossover_point]
        offspring.chromosome[crossover_point:] = parent2.chromosome[crossover_point:]
        return offspring

    # Perform mutation on an offspring
    def mutate(self, agent):
        for i in range(BIT_LENGTH):
            if np.random.rand() < MUTATION_RATE:
                agent.chromosome[i] = 1 - agent.chromosome[i]  # Flip the bit

    # Create the next generation of the population
    def create_next_generation(self):
        new_population = []
        self.evaluate_population()
        for _ in range(POPULATION_SIZE):
            # Select two parents and produce offspring
            parent1, parent2 = self.select_parents()
            offspring = self.crossover(parent1, parent2)
            self.mutate(offspring)
            new_population.append(offspring)
        self.population = new_population

    # Run the genetic algorithm for a specified number of generations
    def run(self):
        for generation in range(GENERATIONS):
            self.create_next_generation()
            self.evaluate_population()
            best_fitness = max(agent.fitness for agent in self.population)
            print(f"Generation {generation}, Best fitness: {best_fitness}")
            
            # If an agent has reached maximum fitness, stop early
            if best_fitness == BIT_LENGTH:
                print("Target reached!")
                break

# Run the Genetic Algorithm
ga = GeneticAlgorithm()
ga.run()
