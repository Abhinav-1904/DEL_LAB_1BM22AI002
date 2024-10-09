import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

def build_and_train_model(hidden_activation, output_activation):
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation=hidden_activation))
    model.add(Dense(1, activation=output_activation))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
    model.fit(X, y, epochs=5000, verbose=0)
    return model

model_relu_sigmoid = build_and_train_model('relu', 'sigmoid')
print("ReLU Activation (Hidden Layer) and Sigmoid Activation (Output Layer) Results:")
predictions_relu_sigmoid = model_relu_sigmoid.predict(X)
print(np.round(predictions_relu_sigmoid))
model_tanh_sigmoid = build_and_train_model('tanh', 'sigmoid')
print("\nTanh Activation (Hidden Layer) and Sigmoid Activation (Output Layer) Results:")
predictions_tanh_sigmoid = model_tanh_sigmoid.predict(X)
print(np.round(predictions_tanh_sigmoid))

import matplotlib.pyplot as plt

def get_activation_function(name):
    if name == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))
    elif name == 'tanh':
        return lambda x: np.tanh(x)
    elif name == 'relu':
        return lambda x: np.maximum(0, x)

def plot_activation_function(activation_func, x_range, title):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = activation_func(x)
    
    plt.plot(x, y, label=title)
    plt.title(title)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()

x_range = (-5, 5)

plt.figure(figsize=(8, 4))

sigmoid = get_activation_function('sigmoid')
plot_activation_function(sigmoid, x_range, 'Sigmoid Activation Function')

tanh = get_activation_function('tanh')
plot_activation_function(tanh, x_range, 'Hyperbolic Tangent (Tanh) Activation Function')

relu = get_activation_function('relu')
plot_activation_function(relu, x_range, 'ReLU Activation Function')

plt.tight_layout()
plt.show()
