import numpy as np

# Activation function: Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss Function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        
        # Initialize biases
        self.bias_hidden = np.random.randn(1, hidden_size)
        self.bias_output = np.random.randn(1, output_size)

    def feedforward(self, X):
        # Input to hidden layer
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        # Hidden to output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = sigmoid(self.output_layer_input)
        
        return self.output_layer_output

    def backpropagate(self, X, y, learning_rate):
        # Feedforward
        output = self.feedforward(X)
        
        # Calculate the error (loss) and gradients
        output_error = y - output  # Loss gradient for output layer
        output_gradient = output_error * sigmoid_derivative(output)
        
        # Calculate hidden layer error and gradient
        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * sigmoid_derivative(self.hidden_layer_output)
        
        # Update the weights and biases
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_gradient) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_gradient) * learning_rate
        self.bias_output += np.sum(output_gradient, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass and backpropagation
            self.backpropagate(X, y, learning_rate)
            
            # Calculate and print loss every 100 epochs
            if epoch % 100 == 0:
                loss = mse_loss(y, self.feedforward(X))
                print(f'Epoch {epoch}, Loss: {loss}')

# Sample XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Create the neural network
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Train the neural network
nn.train(X, y, epochs=1000, learning_rate=0.4)

# Test the neural network
output = nn.feedforward(X)
print("Predicted Output:\n", output)
