import math
import random

HIDDEN_NODES = 4
LEARNING_RATE = 0.1
EPOCHS = 10000

XOR_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_TARGETS = [0, 1, 1, 0]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def predict(hidden_weights, output_weights, point):
    """Forward pass. Returns (output, hidden_activations)."""
    h = len(hidden_weights)
    a = []
    for j in range(h):
        z = hidden_weights[j][0]
        for i, xi in enumerate(point):
            z += hidden_weights[j][i + 1] * xi
        a.append(sigmoid(z))

    z_out = output_weights[0]
    for j in range(h):
        z_out += output_weights[j + 1] * a[j]
    y = sigmoid(z_out)

    return y, a


def train(hidden_weights, output_weights, point, target, lr):
    """One forward + backward pass on a single sample."""
    y, a = predict(hidden_weights, output_weights, point)
    h = len(hidden_weights)

    delta_out = (target - y) * y * (1.0 - y)

    # Hidden deltas computed with original output_weights (before any update).
    delta_hidden = []
    for j in range(h):
        e_j = output_weights[j + 1] * delta_out
        delta_hidden.append(a[j] * (1.0 - a[j]) * e_j)

    # Update output weights.
    output_weights[0] += lr * delta_out
    for j in range(h):
        output_weights[j + 1] += lr * delta_out * a[j]

    # Update hidden weights.
    for j in range(h):
        hidden_weights[j][0] += lr * delta_hidden[j]
        for i, xi in enumerate(point):
            hidden_weights[j][i + 1] += lr * delta_hidden[j] * xi

    return hidden_weights, output_weights


def epoch(hidden_weights, output_weights, X, T, lr):
    for point, target in zip(X, T):
        hidden_weights, output_weights = train(hidden_weights, output_weights, point, target, lr)
    return hidden_weights, output_weights


def evaluate(hidden_weights, output_weights, X, T):
    total = 0.0
    for point, target in zip(X, T):
        y, _ = predict(hidden_weights, output_weights, point)
        total += (target - y) ** 2
    return total / len(X)


def init_weights(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]


if __name__ == "__main__":
    random.seed(None)

    n_inputs = 2
    hidden_weights = init_weights(HIDDEN_NODES, n_inputs + 1)
    output_weights = [random.uniform(-1, 1) for _ in range(HIDDEN_NODES + 1)]

    for e in range(EPOCHS):
        hidden_weights, output_weights = epoch(
            hidden_weights, output_weights, XOR_INPUTS, XOR_TARGETS, LEARNING_RATE
        )
        if e % 1000 == 0:
            err = evaluate(hidden_weights, output_weights, XOR_INPUTS, XOR_TARGETS)
            print(f"Epoch {e:5d}: MSE = {err:.4f}")

    print("\nFinal predictions:")
    for point, target in zip(XOR_INPUTS, XOR_TARGETS):
        y, _ = predict(hidden_weights, output_weights, point)
        print(f"  {point}  target={target}  output={y:.4f}")
