import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def perceptron(X, y, learning_rate=0.1, n_iterations=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(n_iterations):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights) + bias
            y_predicted = 1 if linear_output >= 0 else 0
            update = learning_rate * (y[idx] - y_predicted)
            weights += update * x_i
            bias += update
    return weights, bias

def predict(X, weights, bias):
    linear_output = np.dot(X, weights) + bias
    return np.where(linear_output >= 0, 1, 0)

def evaluate_model(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    return conf_matrix, class_report

if __name__ == "__main__":
    df = pd.read_csv('dataset.csv')
    X = df[['Age', 'Commision (in value)', 'Duration', 'Net Sales']].to_numpy(dtype=np.float64)
    y = df['Claim'].to_numpy(dtype=np.float64)
    
    weights, bias = perceptron(X, y, learning_rate=0.1, n_iterations=10)
    print("Weight of each feature:", weights)
    
    predictions = predict(X, weights, bias)
    
    print("\nFirst 20 Predictions:")
    for i in range(20):
        features = X[i]
        print(f"Features: Age={features[0]}, Commission={features[1]}, Duration={features[2]}, Net Sales={features[3]} | Predicted Claim: {predictions[i]}")
    
    conf_matrix, class_report = evaluate_model(y, predictions)
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
