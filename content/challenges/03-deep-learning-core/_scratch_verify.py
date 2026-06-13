import numpy as np


def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


# ---- SOLUTION ----
def forward_pass(layers, x):
    """Run a forward pass of a multi-layer network.

    Each layer is a dict with keys "W" (weight matrix of shape
    (n_neurons, n_inputs)) and "b" (bias vector of shape (n_neurons,)).
    For every layer compute z = W @ a + b, then a = sigmoid(z), where
    a starts as the input x and the output of each layer feeds the next.

    Args:
        layers: list of dicts, each {"W": np.ndarray, "b": np.ndarray}.
        x: 1D np.ndarray, the input feature vector.

    Returns:
        1D np.ndarray, the activated output of the final layer.

    Raises:
        ValueError: if any layer weight matrix columns do not match the
            size of the activation vector entering that layer.
    """
    a = np.asarray(x, dtype=float)
    for layer in layers:
        W = np.asarray(layer["W"], dtype=float)
        b = np.asarray(layer["b"], dtype=float)
        if W.shape[1] != a.shape[0]:
            raise ValueError(
                f"shape mismatch: W has {W.shape[1]} columns "
                f"but input has {a.shape[0]} elements"
            )
        z = W @ a + b
        a = sigmoid(z)
    return a


# ---- TESTS ----

# Test 1: single layer linear + sigmoid, identity-ish
layers = [{"W": np.array([[1.0, 0.0], [0.0, 1.0]]), "b": np.array([0.0, 0.0])}]
out = forward_pass(layers, np.array([0.0, 0.0]))
assert out.shape == (2,)
assert np.allclose(out, [0.5, 0.5]), out

# Test 2: hand-tuned XOR network (2-2-1) classifies all four cases
hidden = {"W": np.array([[20.0, 20.0], [-20.0, -20.0]]), "b": np.array([-10.0, 30.0])}
output = {"W": np.array([[20.0, 20.0]]), "b": np.array([-30.0])}
net = [hidden, output]
xor_cases = [([0.0, 0.0], 0), ([0.0, 1.0], 1), ([1.0, 0.0], 1), ([1.0, 1.0], 0)]
for inp, expected in xor_cases:
    y = forward_pass(net, np.array(inp))
    assert y.shape == (1,)
    pred = 1 if y[0] >= 0.5 else 0
    assert pred == expected, (inp, y, expected)

# Test 3: large magnitude inputs saturate without overflow (clip works)
big = forward_pass([{"W": np.array([[1000.0]]), "b": np.array([0.0])}], np.array([1.0]))
assert np.isfinite(big[0])
assert big[0] >= 0.999, big
small = forward_pass([{"W": np.array([[-1000.0]]), "b": np.array([0.0])}], np.array([1.0]))
assert np.isfinite(small[0])
assert small[0] <= 0.001, small

# Test 4: three-layer network produces correct output shape and value range
three = [
    {"W": np.ones((4, 3)) * 0.1, "b": np.zeros(4)},
    {"W": np.ones((2, 4)) * 0.1, "b": np.zeros(2)},
    {"W": np.ones((1, 2)) * 0.1, "b": np.zeros(1)},
]
y = forward_pass(three, np.array([1.0, 2.0, 3.0]))
assert y.shape == (1,)
assert np.all((y > 0.0) & (y < 1.0))

# Test 5: shape mismatch raises ValueError
bad = [{"W": np.array([[1.0, 2.0, 3.0]]), "b": np.array([0.0])}]
raised = False
try:
    forward_pass(bad, np.array([1.0, 2.0]))
except ValueError:
    raised = True
assert raised

# Test 6: bias offset shifts the output as expected
b_off = forward_pass([{"W": np.array([[0.0]]), "b": np.array([10.0])}], np.array([5.0]))
assert b_off[0] > 0.999, b_off
b_neg = forward_pass([{"W": np.array([[0.0]]), "b": np.array([-10.0])}], np.array([5.0]))
assert b_neg[0] < 0.001, b_neg

print("ALL TESTS PASSED")
