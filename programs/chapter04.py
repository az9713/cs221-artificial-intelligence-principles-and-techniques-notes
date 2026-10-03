'Original finite neural-network fixtures. Standard ' \
    'library; no external effects.\n\nAll scalar inputs must ' \
    'be int/float, not bool, convertible to finite ' \
    'binary64,\nand at most 1e6 in magnitude. Shapes are ' \
    'fixed where stated. Rounded arithmetic\nand saturation ' \
    'are intentional; this is not an arbitrary-precision ' \
    'solver.\nInvalid values, shapes, or updates outside the ' \
    'supported range raise ValueError.\n'

import itertools
import math

LIMIT = 1e6


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected an int or float, excluding bool")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("number is not representable") from exc
    if not math.isfinite(value) or abs(value) > LIMIT:
        raise ValueError("expected finite magnitude at most 1e6")
    return value


def vector(values, size=None):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("expected a nonempty list or tuple")
    if (size is not None and len(values) != size) or len(values) > 10000:
        raise ValueError("unsupported vector length")
    return tuple(number(x) for x in values)


def _sigmoid(score):
    # Internal scores are finite; this avoids exp(positive large value).
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    e = math.exp(score)
    return e / (1.0 + e)


def loss_gradient(parameters, x, target):
    (
        'Two inputs, two ReLU units, one logit; return loss and '
        'nine derivatives.\n\n    Parameter order: '
        'W00,W01,W10,W11,b0,b1,v0,v1,c. Target is int 0 or 1.\n  '
        '  The ReLU backward convention at zero is zero. Inputs '
        'are never mutated.\n    '
    )
    w00, w01, w10, w11, b0, b1, v0, v1, c = vector(parameters, 9)
    x0, x1 = vector(x, 2)
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target not in (0, 1)
    ):
        raise ValueError("target must be integer 0 or 1")
    a0 = math.fsum((w00 * x0, w01 * x1, b0))
    a1 = math.fsum((w10 * x0, w11 * x1, b1))
    h0, h1 = max(0.0, a0), max(0.0, a1)
    score = math.fsum((v0 * h0, v1 * h1, c))
    # Binary cross entropy without subtracting two large nearly equal values.
    loss = max(score, 0.0) if target == 0 else max(-score, 0.0)
    loss += math.log1p(math.exp(-abs(score)))
    # Avoid cancellation of sigmoid(score)-1 for a positive correct-class
    # score.
    delta = _sigmoid(score) if target == 0 else -_sigmoid(-score)
    q0 = delta * v0 if a0 > 0 else 0.0
    q1 = delta * v1 if a1 > 0 else 0.0
    gradient = (
        q0 * x0,
        q0 * x1,
        q1 * x0,
        q1 * x1,
        q0,
        q1,
        delta * h0,
        delta * h1,
        delta,
    )
    return loss, gradient


def step(parameters, gradient, rate):
    "Return fresh supported parameters; no mutation or gradient accumulation."
    p = vector(parameters, 9)
    # Gradients produced above may exceed the scalar-input support limit.
    # For an externally supplied update require the same finite bounded
    # contract.
    g = vector(gradient, 9)
    rate = number(rate)
    if rate <= 0:
        raise ValueError("rate must be positive")
    return vector([a - rate * b for a, b in zip(p, g)], 9)


def normalize(values, epsilon):
    """Within-vector population normalization, before learned gain/shift."""
    x = vector(values)
    epsilon = number(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    # Center relative to an input anchor: an identical vector stays exactly
    # zero
    # even when epsilon is tiny, instead of amplifying a rounded absolute mean.
    shifted = tuple(z - x[0] for z in x)
    mean_shift = math.fsum(shifted) / len(x)
    centered = tuple(z - mean_shift for z in shifted)
    variance = math.fsum(z * z for z in centered) / len(x)
    denominator = math.sqrt(variance + epsilon)
    return tuple(z / denominator for z in centered)


def finite_difference(parameters, x, target, increment=1e-6):
    """Central numerical check, meaningful only away from ReLU kinks."""
    p = vector(parameters, 9)
    increment = number(increment)
    if increment <= 0:
        raise ValueError("increment must be positive")
    result = []
    for j in range(9):
        plus, minus = list(p), list(p)
        plus[j] += increment
        minus[j] -= increment
        if plus[j] == p[j] or minus[j] == p[j]:
            raise ValueError("increment is lost to rounding")
        result.append(
            (
                loss_gradient(plus, x, target)[0]
                - loss_gradient(minus, x, target)[0]
            )
            / (2 * increment)
        )
    return tuple(result)


def demo():
    p = (2.0, -1.0, 1.0, 0.5, 0.0, 0.5, 1.0, -1.0, 0.0)
    x = (0.5, 0.0)
    loss, gradient = loss_gradient(p, x, 1)
    updated = step(p, gradient, 0.1)
    new_loss = loss_gradient(updated, x, 1)[0]
    numerical = finite_difference(p, x, 1)
    error = max(abs(a - b) for a, b in zip(gradient, numerical))
    assert error < 1e-8 and new_loss < loss
    assert p == (2.0, -1.0, 1.0, 0.5, 0.0, 0.5, 1.0, -1.0, 0.0)
    # Enumerate independent centered weights of variance 1/4 with x=(1,1,1,1).
    sums = [sum(w) for w in itertools.product((-0.5, 0.5), repeat=4)]
    variance = math.fsum(z * z for z in sums) / len(sums)
    assert variance == 1.0
    assert normalize([7.0, 7.0], 1e-5) == (0.0, 0.0)
    print(f"warning probability before: {_sigmoid(0.0):.6f}")
    print(f"loss before: {loss:.6f}")
    print("gradient:", ", ".join(f"{z:.6f}" for z in gradient))
    print(f"loss after one step: {new_loss:.6f}")
    print("central difference check: passed")
    print(f"input derivative at depth 20, w=0.5: {0.5**20:.12f}")
    print(f"shared-weight derivative at depth 20, w=0.5: {20 * 0.5**19:.12f}")
    print(f"residual multiplier when w=-1: {1.0 + (-1.0):.6f}")
    print(
        "population normalization:",
        ", ".join(f"{z:.6f}" for z in normalize([1.0, 3.0], 3.0)),
    )
    print(f"enumerated initialization variance: {variance:.6f}")
    print(f"full sample mean: {13 / 3:.6f}; equal-batch mean: {5.5:.6f}")


if __name__ == "__main__":
    demo()
