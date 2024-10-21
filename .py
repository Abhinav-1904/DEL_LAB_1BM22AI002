#How does a feed forward neural network with multi layer perceptron architecture solve the XOR problem.

import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])
np.random.seed(42)
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1
weights_input_hidden = np.random.uniform(-1, 1, (input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_layer_size, output_layer_size))

learning_rate = 0.1
epochs = 10000
for epoch in range(epochs):
  
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

print("Final predictions:")
for i in range(len(X)):
    hidden_layer_output = sigmoid(np.dot(X[i], weights_input_hidden))
    predicted_output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output))
    print(f"Input: {X[i]}, Prediction: {np.round(predicted_output[0])}")

#how does a feed forward neural network handle multi class multification tasks and what are the key steps involved in propogating input network to the network to produce classs probabilities using activation function like softmax
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

input_size = X.shape[1]
hidden_size = 5
output_size = y_onehot.shape[1]

np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.random.randn(hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.random.randn(output_size)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    hidden_layer_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = softmax(output_layer_input)

    loss = -np.mean(np.sum(y_train * np.log(predicted_output + 1e-9), axis=1))

    output_error = predicted_output - y_train
    hidden_error = np.dot(output_error, weights_hidden_output.T) * hidden_layer_output * (1 - hidden_layer_output)

    weights_hidden_output -= learning_rate * np.dot(hidden_layer_output.T, output_error) / X_train.shape[0]
    bias_output -= learning_rate * np.mean(output_error, axis=0)
    weights_input_hidden -= learning_rate * np.dot(X_train.T, hidden_error) / X_train.shape[0]
    bias_hidden -= learning_rate * np.mean(hidden_error, axis=0)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

hidden_layer_input = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = softmax(output_layer_input)

predicted_classes = np.argmax(predicted_output, axis=1)
true_classes = np.argmax(y_test, axis=1)

accuracy = np.mean(predicted_classes == true_classes)
print(f'Accuracy: {accuracy * 100:.2f}%')
