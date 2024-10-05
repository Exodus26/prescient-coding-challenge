import numpy as np # My balls itch

# Activation function: Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Custom activation function (ReLU) and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Mean Squared Error Loss Function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Accuracy Calculation
def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)  # Get the index of the max log-probability
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == true_labels)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, output_activation=sigmoid):
        self.layers = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights and biases for each layer
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)]
        self.biases = [np.random.randn(1, self.layers[i + 1]) for i in range(len(self.layers) - 1)]
        self.output_activation = output_activation
        self.output_activation_derivative = sigmoid_derivative if output_activation == sigmoid else relu_derivative

    def feedforward(self, X):
        self.layer_outputs = [X]  # Store all layer outputs for backpropagation
        for i in range(len(self.weights)):
            Z = np.dot(X, self.weights[i]) + self.biases[i]
            X = sigmoid(Z) if i < len(self.weights) - 1 else self.output_activation(Z)
            self.layer_outputs.append(X)
        return X

    def process_output(self, output):
        # Custom processing of the output, e.g., scaling and thresholding
        # Here we will scale the output and apply a threshold of 0.5
        output_scaled = output * 2  # Scale to range [0, 2]
        output_processed = (output_scaled > 1).astype(float)  # Thresholding
        return output_processed

    def backpropagate(self, X, y, learning_rate):
        output = self.feedforward(X)
        
        # Process the output before backpropagation
        output_processed = self.process_output(output)
        
        # Output layer error
        output_error = y - output_processed
        gradients = [output_error * self.output_activation_derivative(output_processed)]
        
        # Backpropagate the error to hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(gradients[0], self.weights[i].T)
            gradients.insert(0, error * sigmoid_derivative(self.layer_outputs[i]))
        
        # Update weights and biases using broadcasting
        for i in range(len(self.weights)):
            self.weights[i] += np.dot(self.layer_outputs[i].T, gradients[i]) * learning_rate
            self.biases[i] += np.sum(gradients[i], axis=0, keepdims=True) * learning_rate

    def train(self, X, Y, epochs, learning_rate, batch_size=1, max_accurate=0.8):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):  # Loop through the dataset in batches
                x = X[i:i + batch_size]  # Select the current batch of inputs
                y = Y[i:i + batch_size]  # Select the current batch of outputs
                self.backpropagate(x, y, learning_rate)  # Train on the current batch
            
            # Calculate and print loss and accuracy every 100 epochs
            if epoch % 100 == 0:
                loss = mse_loss(Y, self.feedforward(X))
                acc = accuracy(Y, self.feedforward(X))
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
                
                # Stop training if maximum accuracy is reached
                if acc >= max_accurate:
                    print(f'Maximum accuracy of {max_accurate} reached at epoch {epoch}. Stopping training.')
                    return

# User inputs for the network
input_size = 28 * 28  # Example input size (to be defined as per your dataset)
hidden_layers = [30, 30, 40]  # Example: 3 hidden layers with sizes 30, 30, and 40
output_size = 10  # Example output size (for multi-class classification)

# Example synthetic training data (to be replaced with actual data)
num_samples = 1000  # Define number of samples
x_train = np.random.rand(num_samples, input_size)  # Random input data
y_train = np.zeros((num_samples, output_size))  # Placeholder for output data
y_train[np.arange(num_samples), np.random.randint(0, output_size, num_samples)] = 1  # Random one-hot labels

# Training parameters
epochs = 1000
learning_rate = 0.1
max_accurate = 0.8
batch_size = 1

# Create and train the neural network with a custom output activation function
nn = NeuralNetwork(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, output_activation=sigmoid)
nn.train(x_train, y_train, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, max_accurate=max_accurate)

# Example test set (to be replaced with actual test data)
x_test = np.random.rand(100, input_size)  # Random test data
y_test = np.zeros((100, output_size))  # Placeholder for output data
y_test[np.arange(100), np.random.randint(0, output_size, 100)] = 1  # Random one-hot labels

# Test the neural network on the test set
test_output = nn.feedforward(x_test)
test_acc = accuracy(y_test, test_output)
print(f'Test Accuracy: {test_acc:.4f}')
