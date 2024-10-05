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

# Customizable Recurrent Neural Network class
class RecurrentNeuralNetwork:
    def __init__(self, layer_sizes):
        # layer_sizes = [input_size, hidden1_size, hidden2_size, ..., output_size]
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize the weights for each layer
        self.weights = [] #TODO: change to numpy array
        self.recurrent_weights = [] #TODO: change to numpy array

        for i in range(self.num_layers - 1):
            # Weights between input and hidden or hidden and hidden layers
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1])) #TODO: change to numpy
            if i > 0:  # Recurrent weights for hidden layers
                self.recurrent_weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i + 1])) #TODO: change to numpy

        # Initialize the hidden states for each hidden layer
        self.hidden_states = [np.zeros(layer_size) for layer_size in layer_sizes[1:-1]] #TODO change tot numpy

    # Feedforward through the RNN for a sequence of inputs
    def feedforward(self, X):
        outputs = []
        # Reset hidden states at the start of each sequence
        self.hidden_states = [np.zeros(layer_size) for layer_size in self.layer_sizes[1:-1]]

        for x in X:
            input_to_next_layer = x

            # Loop through all layers except the output layer
            for i in range(self.num_layers - 2):
                if i == 0:
                    # Input to first hidden layer
                    input_to_next_layer = sigmoid(np.dot(input_to_next_layer, self.weights[i]))
                else:
                    # Hidden layers with recurrent connections
                    self.hidden_states[i - 1] = sigmoid(np.dot(input_to_next_layer, self.weights[i]) + np.dot(self.hidden_states[i - 1], self.recurrent_weights[i - 1]))
                    input_to_next_layer = self.hidden_states[i - 1]

            # Output layer
            output = sigmoid(np.dot(input_to_next_layer, self.weights[-1]))
            outputs.append(output)

        return np.array(outputs)

# Agent class representing an individual recurrent neural network in the population
class Agent:
    def __init__(self, layer_sizes):
        self.network = RecurrentNeuralNetwork(layer_sizes)
        self.fitness = 0

    # Fitness function based on error (the lower, the better)
    def calculate_fitness(self, X, y):
        predictions = self.network.feedforward(X)
        self.fitness = np.sum(predictions)**2

    def normalise_fitness(self, maximum_fitness,factor):
        self.fitness /= maximum_fitness/factor

# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, layer_sizes, population_size):
        self.population = [Agent(layer_sizes) for _ in range(population_size)]
        self.population_size = population_size

    # Evaluate fitness of the population
    def evaluate_population(self, X, y):
        temp_max=1
        for agent in self.population:
            agent.calculate_fitness(X, y)
            if agent.fitness > temp_max:
                temp_max = agent.fitness

        for agent in self.population:
            agent.normalise_fitness(temp_max,10)
        
    # Select two parents using tournament selection
    def select_parents(self, tourney_size = 5):
        tournament = r.sample(self.population, tourney_size)
        parent1 = max(tournament, key=lambda agent: agent.fitness)
        parent2 = max(r.sample(self.population, tourney_size), key=lambda agent: agent.fitness)
        return parent1, parent2

    # Crossover between two parent recurrent neural networks
    def crossover(self, parent1, parent2):
        child = Agent(parent1.network.layer_sizes)

        # Crossover for standard and recurrent weights
        for i in range(len(parent1.network.weights)):
            crossover_point = np.random.randint(0, parent1.network.weights[i].size)
            child.network.weights[i].flat[:crossover_point] = parent1.network.weights[i].flat[:crossover_point]
            child.network.weights[i].flat[crossover_point:] = parent2.network.weights[i].flat[crossover_point:]

        for i in range(len(parent1.network.recurrent_weights)):
            crossover_point = np.random.randint(0, parent1.network.recurrent_weights[i].size)
            child.network.recurrent_weights[i].flat[:crossover_point] = parent1.network.recurrent_weights[i].flat[:crossover_point]
            child.network.recurrent_weights[i].flat[crossover_point:] = parent2.network.recurrent_weights[i].flat[crossover_point:]

        return child

    # Mutation: Randomly perturb weights
    def mutate(self, agent, mutation_rate=0.01):
        for w in agent.network.weights:
            mutation_mask = np.random.rand(*w.shape) < mutation_rate
            w[mutation_mask] += np.random.randn(np.sum(mutation_mask))

        for rw in agent.network.recurrent_weights:
            mutation_mask = np.random.rand(*rw.shape) < mutation_rate
            rw[mutation_mask] += np.random.randn(np.sum(mutation_mask))

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

# Example: XOR problem dataset (customizable RNN)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])  # XOR outputs

# Customize the RNN architecture: [input_size, hidden1_size, ..., output_size]
layer_sizes = [2, 5, 3, 1]  # Example with 1 input layer, 2 hidden layers, and 1 output layer

# Create the genetic algorithm to evolve a customizable recurrent neural network
ga = GeneticAlgorithm(layer_sizes=layer_sizes, population_size=20)

# Run the genetic algorithm for 200 generations
ga.run(X, y, generations=200)

# Test the best agent from the last generation
best_agent = max(ga.population, key=lambda agent: agent.fitness)
predictions = best_agent.network.feedforward(X)
print("Predictions:", predictions)
