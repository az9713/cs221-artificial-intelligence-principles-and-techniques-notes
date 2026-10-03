"""Original CS221 learning fixture. Standard library; no file/network effects.

Public scalars accept finite built-in int/float, excluding bool, converted to
binary64. Nonfinite conversions/intermediates raise ValueError. Ordinary finite
rounding, including subnormal underflow, is allowed. Arithmetic is not exact.
Graphs are immutable; backward returns a fresh mapping, accumulating every
operand edge. Data are nonempty sequences of (input,target) pairs; theta=(w,b).
"""
from dataclasses import dataclass
import math


def finite(value):
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError("expected a built-in real scalar, excluding bool")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("scalar is outside binary64 range") from exc
    if not math.isfinite(result):
        raise ValueError("scalar or computed intermediate is nonfinite")
    return result


def total(values):
    try:
        return finite(math.fsum(values))
    except OverflowError as exc:
        raise ValueError("sum is outside binary64 range") from exc


@dataclass(frozen=True, eq=False)
class Scalar:
    value: float
    edges: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "value", finite(self.value))
        if type(self.edges) is not tuple:
            raise ValueError("operand edges must be a tuple")
        for edge in self.edges:
            if type(edge) is not tuple or len(edge) != 2:
                raise ValueError("each edge is (Scalar, local derivative)")
            parent, derivative = edge
            if not isinstance(parent, Scalar):
                raise ValueError("edge dependency must be a Scalar")
            finite(derivative)

    def __add__(self, other):
        other = as_scalar(other)
        return Scalar(finite(self.value + other.value),
                      ((self, 1.0), (other, 1.0)))

    __radd__ = __add__

    def __mul__(self, other):
        other = as_scalar(other)
        return Scalar(finite(self.value * other.value),
                      ((self, other.value), (other, self.value)))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + -as_scalar(other)

    def __rsub__(self, other):
        return as_scalar(other) + -self


def as_scalar(value):
    return value if isinstance(value, Scalar) else Scalar(value)


def backward(root, seed=1.0):
    if not isinstance(root, Scalar):
        raise ValueError("root must be a Scalar")
    seed = finite(seed)
    # Iterative DFS: dependencies before consumers; gray nodes detect cycles.
    order, state = [], {}
    stack = [(root, False)]
    while stack:
        node, finishing = stack.pop()
        if finishing:
            state[node] = 2
            order.append(node)
        elif state.get(node) == 1:
            raise ValueError("graph contains a cycle")
        elif state.get(node) != 2:
            state[node] = 1
            stack.append((node, True))
            for parent, _ in reversed(node.edges):
                stack.append((parent, False))
    adjoints = dict.fromkeys(order, 0.0)
    adjoints[root] = seed
    for node in reversed(order):
        for parent, local in node.edges:
            contribution = finite(adjoints[node] * local)
            adjoints[parent] = finite(adjoints[parent] + contribution)
    return adjoints


def data_pairs(data):
    if type(data) not in (list, tuple) or not data:
        raise ValueError("data must be a nonempty list/tuple of pairs")
    result = []
    for pair in data:
        if type(pair) not in (list, tuple) or len(pair) != 2:
            raise ValueError("each example must contain input and target")
        result.append((finite(pair[0]), finite(pair[1])))
    return tuple(result)


def parameters(theta):
    if type(theta) not in (list, tuple) or len(theta) != 2:
        raise ValueError("theta must contain weight and bias")
    return finite(theta[0]), finite(theta[1])


def predict(theta, x):
    weight, bias = parameters(theta)
    return finite(finite(weight * finite(x)) + bias)


def loss_gradient(data, theta):
    data = data_pairs(data)
    weight, bias = parameters(theta)
    losses, weight_terms, bias_terms = [], [], []
    for x, y in data:
        residual = finite(finite(finite(weight * x) + bias) - y)
        losses.append(finite(residual * residual))
        twice = finite(2.0 * residual)
        weight_terms.append(finite(twice * x))
        bias_terms.append(twice)
    n = len(data)
    return total(losses) / n, (total(weight_terms) / n, total(bias_terms) / n)


def graph_loss_gradient(data, theta):
    data = data_pairs(data)
    weight, bias = map(Scalar, parameters(theta))
    objective = Scalar(0.0)
    for x, y in data:
        residual = weight * x + bias - y
        objective = objective + residual * residual
    objective = objective * (1.0 / len(data))
    adjoints = backward(objective)
    return objective.value, (adjoints[weight], adjoints[bias])


def descend(data, theta, rate, steps):
    data = data_pairs(data)
    theta = parameters(theta)
    rate = finite(rate)
    if rate <= 0 or type(steps) is not int or steps < 0:
        raise ValueError("rate must be positive; steps a nonnegative integer")
    history = [loss_gradient(data, theta)[0]]
    for _ in range(steps):
        _, gradient = loss_gradient(data, theta)
        theta = tuple(finite(t - finite(rate * g))
                      for t, g in zip(theta, gradient))
        history.append(loss_gradient(data, theta)[0])
    return theta, tuple(history)


def central_gradient(data, theta, epsilon=1e-5):
    data = data_pairs(data)
    theta = parameters(theta)
    epsilon = finite(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    result = []
    for j in range(2):
        above, below = list(theta), list(theta)
        above[j] = finite(theta[j] + epsilon)
        below[j] = finite(theta[j] - epsilon)
        if above[j] == theta[j] or below[j] == theta[j]:
            raise ValueError("perturbation is not resolved at this parameter")
        numerator = finite(loss_gradient(data, above)[0]
                           - loss_gradient(data, below)[0])
        denominator = finite(2.0 * epsilon)
        result.append(finite(numerator / denominator))
    return tuple(result)


DATA = ((1.0, 4.0), (2.0, 6.0), (4.0, 7.0))


def main():
    initial = (0.0, 1.0)
    loss, gradient = loss_gradient(DATA, initial)
    graph_loss, graph_gradient = graph_loss_gradient(DATA, initial)
    numeric = central_gradient(DATA, initial)
    assert math.isclose(loss, 70 / 3)
    assert math.isclose(graph_loss, loss)
    assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)
               for a, b in zip(gradient, graph_gradient))
    assert all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)
               for a, b in zip(gradient, numeric))
    u = Scalar(2.0)
    shared = u * u + u
    assert shared.value == 6 and backward(shared)[u] == 5
    assert backward(shared)[u] == 5  # fresh adjoints on each pass
    first, first_losses = descend(DATA, initial, 0.01, 1)
    _, stable = descend(DATA, initial, 0.01, 10)
    _, unstable = descend(DATA, initial, 0.2, 10)
    optimum = (13 / 14, 7 / 2)
    optimum_loss, optimum_gradient = loss_gradient(DATA, optimum)
    assert math.isclose(optimum_loss, 3 / 14)
    assert max(map(abs, optimum_gradient)) < 1e-12
    assert all(b < a for a, b in zip(stable, stable[1:]))
    assert unstable[-1] > unstable[0]
    largest = 8 + 2 * math.sqrt(130) / 3
    print(f"initial mean loss: {loss:.9f}")
    print("initial gradient:", ", ".join(f"{g:.9f}" for g in gradient))
    print("graph gradient:", ", ".join(f"{g:.9f}" for g in graph_gradient))
    print("central gradient:", ", ".join(f"{g:.9f}" for g in numeric))
    print(f"shared u*u+u: value={shared.value:.1f}, "
          f"derivative={backward(shared)[u]:.1f}")
    print("first parameters:", ", ".join(f"{t:.9f}" for t in first))
    print(f"first mean loss: {first_losses[-1]:.9f}")
    print(f"strict rate upper bound: {2 / largest:.9f}")
    print(f"ten steps at 0.01: {stable[-1]:.9f}")
    print(f"ten steps at 0.20: {unstable[-1]:.9f}")
    print(f"minimum mean loss: {optimum_loss:.9f}")


if __name__ == "__main__":
    main()
