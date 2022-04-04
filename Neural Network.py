# Neural Network for the AND gate problem
import numpy as np


def MSE(y, Y):
    return np.mean((y - Y)**2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Initialize weights
        self.weights_input_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                                     (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                      (self.hidden_nodes, self.output_nodes))

        self.activation_function = lambda x: sigmoid(x)

    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        hidden_inputs = np.dot(X, self.weights_input_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = final_inputs
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        error = y - final_outputs
        output_error_term = error
        hidden_error = np.dot(self.weights_hidden_output, output_error_term)
        hidden_error_term = hidden_error * \
            hidden_outputs * (1 - hidden_outputs)
        delta_weights_i_h += hidden_error_term * X[:, None]
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weights_hidden_output += self.learning_rate * delta_weights_h_o / n_records
        self.weights_input_hidden += self.learning_rate * delta_weights_i_h / n_records

    def run(self, features):
        hidden_inputs = np.dot(features, self.weights_input_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = final_inputs
        return final_outputs


def main():
    # Generate training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    nn = NeuralNetwork(2, 4, 1, 0.5)
    # Test the neural network 3000 times and keep the model with the lowest MSE
    lowest_mse = float('inf')
    for i in range(3000):
        nn.train(X, y)
        mse = MSE(nn.run(X), y)
        if mse < lowest_mse:
            lowest_mse = mse
            lowest_mse_nn = nn
    # Test the neural network on the test data
    print(lowest_mse_nn.run(X))
    print(lowest_mse)


if __name__ == "__main__":
    main()
