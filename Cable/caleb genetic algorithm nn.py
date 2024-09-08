import numpy as np
import random as r

LAYERS = [90,40,40,40,40,90]
EGG_PENALTY = 10
COUNT_PENALTY = 30

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def end_sigmoid(x):
    return 10 / (1 + np.exp(-x))

class Recurrent_Neural_Network:
    def __init__(self, layers):
        self.layers = len(layers)
        self.weights = np.ndarray(len(layers)-1,np.ndarray)
        self.hidden_weights = np.ndarray(len(layers)-2,np.ndarray)
        self.biases = np.ndarray(len(layers)-1,np.ndarray)

        for i in range(len(layers)-1):
            self.weights[i] = np.random.randn(layers[i], layers[i+1])
            self.biases[i] = np.random.randn(1,layers[i+1])

class Neural_Network:
    def __init__(self, layers):
        self.weights = np.ndarray(len(layers)-1,np.ndarray)
        self.biases = np.ndarray(len(layers)-1,np.ndarray)

        for i in range(len(layers)-1):
            self.weights[i] = np.random.randn(layers[i], layers[i+1])
            self.biases[i] = np.random.randn(1,layers[i+1])
        
    def feedforward(self,inputs):
        temp_output = inputs.copy()
        for i in self.layers - 1:
            temp_output = sigmoid(np.dot(temp_output,self.weights[i])+self.biases[i])
        temp_output /= np.max(temp_output)*10
        temp_output /= np.sum(temp_output)
        return temp_output

class Agent:
    def __init__(self, layers):
        self.network = Neural_Network(layers)
        self.fitness = 0
    
    def calculate_fitness(self, input_dataset): #input dataset is just the normal dataset with date time. first value is zeros
        self.total_earnings = 0
        for X in input_dataset: #have 
            predictions = self.network.feedforward(X)
            
            
            
    



# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, input_size, hidden_size, output_size, population_size):
        self.population = [Agent(input_size, hidden_size, output_size) for _ in range(population_size)]
        self.population_size = population_size

    # Evaluate fitness of the population
    def evaluate_population(self, X, y):
        for agent in self.population:
            agent.calculate_fitness(X, y)

    # Select two parents using tournament selection
    def select_parents(self):
        tournament_size = 5
        tournament = r.sample(self.population, tournament_size)
        parent1 = max(tournament, key=lambda agent: agent.fitness)
        parent2 = max(r.sample(self.population, tournament_size), key=lambda agent: agent.fitness)
        return parent1, parent2

    # Crossover between two parent neural networks
    def crossover(self, parent1, parent2):
        child = Agent(parent1.network.weights_input_hidden.shape[0],
                      parent1.network.weights_input_hidden.shape[1],
                      parent1.network.weights_hidden_output.shape[1])
        
        # Crossover between input-hidden weights
        crossover_point = np.random.randint(0, parent1.network.weights_input_hidden.size)
        child.network.weights_input_hidden.flat[:crossover_point] = parent1.network.weights_input_hidden.flat[:crossover_point]
        child.network.weights_input_hidden.flat[crossover_point:] = parent2.network.weights_input_hidden.flat[crossover_point:]

        # Crossover between hidden-output weights
        crossover_point = np.random.randint(0, parent1.network.weights_hidden_output.size)
        child.network.weights_hidden_output.flat[:crossover_point] = parent1.network.weights_hidden_output.flat[:crossover_point]
        child.network.weights_hidden_output.flat[crossover_point:] = parent2.network.weights_hidden_output.flat[crossover_point:]

        return child

    # Mutation: Randomly perturb weights
    def mutate(self, agent, mutation_rate=0.01):
        for w in [agent.network.weights_input_hidden, agent.network.weights_hidden_output]:
            mutation_mask = np.random.rand(*w.shape) < mutation_rate
            w[mutation_mask] += np.random.randn(np.sum(mutation_mask))

    # Create the next generation
    def create_next_generation(self, X, y):
        new_population = []
        self.evaluate_population(X, y)
        
        for _ in range(self.population_size):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population

    # Run the genetic algorithm
    def run(self, X, y, generations):
        for generation in range(generations):
            self.create_next_generation(X, y)
            self.evaluate_population(X, y)
            best_fitness = max(agent.fitness for agent in self.population)
            print(f"Generation {generation}, Best fitness: {best_fitness}")

# XOR problem dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])  # XOR outputs

# Create the genetic algorithm to evolve a neural network
ga = GeneticAlgorithm(input_size=2, hidden_size=5, output_size=1, population_size=20)

# Run the genetic algorithm for 200 generations
ga.run(X, y, generations=200)

# Test the best agent from the last generation
best_agent = max(ga.population, key=lambda agent: agent.fitness)
predictions = best_agent.network.feedforward(X)
print("Predictions:", predictions)
