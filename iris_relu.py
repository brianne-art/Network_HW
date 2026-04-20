"""Iris classifier with ReLU at the hidden layer, sigmoid at the output layer.

Produces a side-by-side comparison plot of ReLU vs sigmoid training error.
"""

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
        print("Downloading iris dataset...")
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


def relu(x):
    return max(0.0, x)


def relu_derivative(z):
    """Derivative of ReLU with respect to the pre-activation z."""
    return 1.0 if z > 0.0 else 0.0


# ── ReLU network ─────────────────────────────────────────────────────────────

def predict_relu(hidden_weights, output_weights, point):
    """Forward pass with ReLU hidden layer, sigmoid output.

    Returns (outputs, hidden_activations, hidden_pre_activations).
    z_hidden is needed for the ReLU derivative during backprop.
    """
    h = len(hidden_weights)
    z_hidden = []
    a = []
    for j in range(h):
        z = hidden_weights[j][0]
        for i, xi in enumerate(point):
            z += hidden_weights[j][i + 1] * xi
        z_hidden.append(z)
        a.append(relu(z))

    y = []
    for k in range(3):
        z_k = output_weights[k][0]
        for j in range(h):
            z_k += output_weights[k][j + 1] * a[j]
        y.append(sigmoid(z_k))

    return y, a, z_hidden


def train_relu(hidden_weights, output_weights, point, target, lr):
    y, a, z_hidden = predict_relu(hidden_weights, output_weights, point)
    h = len(hidden_weights)

    delta_out = [(target[k] - y[k]) * y[k] * (1.0 - y[k]) for k in range(3)]

    # Hidden deltas — use original output_weights before any update.
    delta_hidden = []
    for j in range(h):
        e_j = sum(output_weights[k][j + 1] * delta_out[k] for k in range(3))
        delta_hidden.append(relu_derivative(z_hidden[j]) * e_j)

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


def evaluate_relu(hidden_weights, output_weights, dataset):
    total_mse = 0.0
    correct = 0
    for features, onehot, true_idx in dataset:
        y, _, _ = predict_relu(hidden_weights, output_weights, features)
        total_mse += sum((onehot[k] - y[k]) ** 2 for k in range(3)) / 3.0
        if y.index(max(y)) == true_idx:
            correct += 1
    return total_mse / len(dataset), correct / len(dataset)


# ── Sigmoid network (re-used from iris_nn.py for the comparison) ─────────────

def predict_sigmoid(hidden_weights, output_weights, point):
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


def train_sigmoid(hidden_weights, output_weights, point, target, lr):
    y, a = predict_sigmoid(hidden_weights, output_weights, point)
    h = len(hidden_weights)

    delta_out = [(target[k] - y[k]) * y[k] * (1.0 - y[k]) for k in range(3)]

    delta_hidden = []
    for j in range(h):
        e_j = sum(output_weights[k][j + 1] * delta_out[k] for k in range(3))
        delta_hidden.append(a[j] * (1.0 - a[j]) * e_j)

    for k in range(3):
        output_weights[k][0] += lr * delta_out[k]
        for j in range(h):
            output_weights[k][j + 1] += lr * delta_out[k] * a[j]

    for j in range(h):
        hidden_weights[j][0] += lr * delta_hidden[j]
        for i, xi in enumerate(point):
            hidden_weights[j][i + 1] += lr * delta_hidden[j] * xi

    return hidden_weights, output_weights


def evaluate_sigmoid(hidden_weights, output_weights, dataset):
    total_mse = 0.0
    correct = 0
    for features, onehot, true_idx in dataset:
        y, _ = predict_sigmoid(hidden_weights, output_weights, features)
        total_mse += sum((onehot[k] - y[k]) ** 2 for k in range(3)) / 3.0
        if y.index(max(y)) == true_idx:
            correct += 1
    return total_mse / len(dataset), correct / len(dataset)


# ── Shared helpers ────────────────────────────────────────────────────────────

def init_weights(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]


def run_training(train_fn, eval_fn, train_set, test_set, seed, label):
    random.seed(seed)
    n_inputs = 4
    hw = init_weights(HIDDEN_NODES, n_inputs + 1)
    ow = init_weights(3, HIDDEN_NODES + 1)

    errors = []
    for e in range(EPOCHS):
        indices = list(range(len(train_set)))
        random.shuffle(indices)
        for idx in indices:
            features, onehot, _ = train_set[idx]
            hw, ow = train_fn(hw, ow, features, onehot, LEARNING_RATE)
        mse, acc = eval_fn(hw, ow, train_set)
        errors.append(mse)
        if e % 50 == 0:
            print(f"[{label}] Epoch {e:4d}: train MSE = {mse:.4f}  train acc = {acc:.2%}")

    _, test_acc = eval_fn(hw, ow, test_set)
    print(f"[{label}] Test accuracy: {test_acc:.2%}\n")
    return errors


if __name__ == "__main__":
    SEED = 42

    random.seed(SEED)
    data = load_iris()
    random.shuffle(data)

    test_raw = data[:30]
    train_raw = data[30:]

    train_norm, mins, maxs = normalize(train_raw)
    test_norm = normalize_with(test_raw, mins, maxs)

    print("=== ReLU hidden layer ===")
    errors_relu = run_training(train_relu, evaluate_relu, train_norm, test_norm, SEED, "ReLU")

    print("=== Sigmoid hidden layer ===")
    errors_sig = run_training(train_sigmoid, evaluate_sigmoid, train_norm, test_norm, SEED, "Sigmoid")

    plt.figure()
    epochs_range = range(EPOCHS)
    plt.plot(epochs_range, errors_relu, label="ReLU hidden")
    plt.plot(epochs_range, errors_sig,  label="Sigmoid hidden")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training MSE")
    plt.title("Iris NN Training Error: ReLU vs Sigmoid Hidden Layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig("iris_relu_vs_sigmoid.png")
    print("Saved iris_relu_vs_sigmoid.png")
