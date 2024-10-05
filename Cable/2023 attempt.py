import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return x*(1-x)

def mse(actual, predicted):
    return np.mean((actual-predicted)**2)

def accuracy(actual, predicted):
    return np.mean(actual == predicted)

def process_output(output, max_number=10):
    # Step 1: Ensure the sum of the array is 100
    scaled_output = output / np.sum(output) * 100 if np.sum(output) != 0 else np.array([100 / len(output) for _ in output])
    
    # Step 2: Ensure no element exceeds the max_number (e.g., 10)
    # If any element is larger than max_number, reduce it and distribute the remainder
    excess = 0
    for i in range(len(scaled_output)):
        if scaled_output[i] > max_number:
            excess += scaled_output[i] - max_number
            scaled_output[i] = max_number
    
    # Step 3: Distribute the excess back proportionally to elements that are below max_number
    below_max_indices = [i for i in range(len(scaled_output)) if scaled_output[i] < max_number]
    
    while excess > 0 and below_max_indices:
        for i in below_max_indices:
            available_increase = max_number - scaled_output[i]
            increase = min(excess, available_increase)
            scaled_output[i] += increase
            excess -= increase
            if scaled_output[i] >= max_number:
                below_max_indices.remove(i)

    # Step 4: Ensure final adjustment so the sum is exactly 100
    final_scaling_factor = 100 / np.sum(scaled_output)
    scaled_output *= final_scaling_factor

    return scaled_output

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, output_activation = (sigmoid, d_sigmoid)):
        self.layers = [input_size] + hidden_layers + [output_size]

        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)]
        self.biases = [np.random.randn(1, self.layers[i + 1]) for i in range(len(self.layers) - 1)]
        self.o_a, self.do_a = output_activation

    def feedforward(self, x):
        self.layer_outputs = [x]
        for i in range(len(self.weights)):
            Z = np.dot(x, self.weights[i]) + self.biases[i]
            Y = sigmoid(Z) if i < len(self.weights) - 1 else self.o_a(Z)
            self.layer_outputs.append(x)
        return x
    
    def backpropagate(self, x, y, learning_rate):
        output = self.feedforward(x)

        # process it?

        error = y - output
        gradients = np.array([error * self.output_activation_derivative(error)])

