import csv
import math
import os
import random
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIDDEN_NODES = 6
LEARNING_RATE = 0.1
EPOCHS = 500

IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
IRIS_FILE = "iris.csv"

CLASS_MAP = {
    "Iris-setosa":     [1, 0, 0],
    "Iris-versicolor": [0, 1, 0],
    "Iris-virginica":  [0, 0, 1],
}
CLASS_INDEX = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}


def load_iris():
    if not os.path.exists(IRIS_FILE):
        print(f"Downloading iris dataset...")
        urllib.request.urlretrieve(IRIS_URL, IRIS_FILE)

    data = []
    with open(IRIS_FILE) as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            try:
                features = [float(row[i]) for i in range(4)]
            except ValueError:
                continue
            label = row[4].strip()
            data.append((features, CLASS_MAP[label], CLASS_INDEX[label]))
    return data


def normalize(data):
    """Min-max normalize features using stats from the provided data."""
    mins = [min(d[0][i] for d in data) for i in range(4)]
    maxs = [max(d[0][i] for d in data) for i in range(4)]
    result = []
    for features, onehot, idx in data:
        normed = [(features[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(4)]
        result.append((normed, onehot, idx))
    return result, mins, maxs


def normalize_with(data, mins, maxs):
    result = []
    for features, onehot, idx in data:
        normed = [(features[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(4)]
        result.append((normed, onehot, idx))
    return result


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def predict(hidden_weights, output_weights, point):
    """Forward pass. Returns (outputs, hidden_activations).

    output_weights is a 3 x (h+1) matrix.
    Returns y as a list of 3 values, a as a list of h values.
    """
    h = len(hidden_weights)
    a = []
    for j in range(h):
        z = hidden_weights[j][0]
        for i, xi in enumerate(point):
            z += hidden_weights[j][i + 1] * xi
        a.append(sigmoid(z))

    y = []
    for k in range(3):
        z_k = output_weights[k][0]
        for j in range(h):
            z_k += output_weights[k][j + 1] * a[j]
        y.append(sigmoid(z_k))

    return y, a


def train(hidden_weights, output_weights, point, target, lr):
    """One forward + backward pass on a single sample."""
    y, a = predict(hidden_weights, output_weights, point)
    h = len(hidden_weights)

    # Output deltas (one per class).
    delta_out = [(target[k] - y[k]) * y[k] * (1.0 - y[k]) for k in range(3)]

    # Hidden deltas — use original output_weights before any update.
    delta_hidden = []
    for j in range(h):
        e_j = sum(output_weights[k][j + 1] * delta_out[k] for k in range(3))
        delta_hidden.append(a[j] * (1.0 - a[j]) * e_j)

    # Update output weights.
    for k in range(3):
        output_weights[k][0] += lr * delta_out[k]
        for j in range(h):
            output_weights[k][j + 1] += lr * delta_out[k] * a[j]

    # Update hidden weights.
    for j in range(h):
        hidden_weights[j][0] += lr * delta_hidden[j]
        for i, xi in enumerate(point):
            hidden_weights[j][i + 1] += lr * delta_hidden[j] * xi

    return hidden_weights, output_weights


def epoch_train(hidden_weights, output_weights, train_set, lr):
    indices = list(range(len(train_set)))
    random.shuffle(indices)
    for idx in indices:
        features, onehot, _ = train_set[idx]
        hidden_weights, output_weights = train(hidden_weights, output_weights, features, onehot, lr)
    return hidden_weights, output_weights


def evaluate(hidden_weights, output_weights, dataset):
    """Returns (mse, accuracy)."""
    total_mse = 0.0
    correct = 0
    for features, onehot, true_idx in dataset:
        y, _ = predict(hidden_weights, output_weights, features)
        total_mse += sum((onehot[k] - y[k]) ** 2 for k in range(3)) / 3.0
        if y.index(max(y)) == true_idx:
            correct += 1
    mse = total_mse / len(dataset)
    accuracy = correct / len(dataset)
    return mse, accuracy


def init_weights(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]


if __name__ == "__main__":
    random.seed(42)

    data = load_iris()
    random.shuffle(data)

    test_set = data[:30]
    train_set = data[30:]

    train_norm, mins, maxs = normalize(train_set)
    test_norm = normalize_with(test_set, mins, maxs)

    n_inputs = 4
    hidden_weights = init_weights(HIDDEN_NODES, n_inputs + 1)
    output_weights = init_weights(3, HIDDEN_NODES + 1)

    errors = []
    for e in range(EPOCHS):
        hidden_weights, output_weights = epoch_train(
            hidden_weights, output_weights, train_norm, LEARNING_RATE
        )
        mse, acc = evaluate(hidden_weights, output_weights, train_norm)
        errors.append(mse)
        if e % 50 == 0:
            print(f"Epoch {e:4d}: train MSE = {mse:.4f}  train acc = {acc:.2%}")

    _, test_acc = evaluate(hidden_weights, output_weights, test_norm)
    print(f"\nTest accuracy: {test_acc:.2%}")

    plt.figure()
    plt.plot(range(EPOCHS), errors)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training MSE")
    plt.title("Iris NN Training Error (Sigmoid hidden layer)")
    plt.tight_layout()
    plt.savefig("iris_training_error.png")
    print("Saved iris_training_error.png")
