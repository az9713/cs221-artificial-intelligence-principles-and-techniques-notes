"""Original CS221 numerical teaching lab. Python 3.10+, standard library.
Run: python numerical_lab.py
No network, filesystem writes, external actions, or mutation of caller inputs.
"""
from __future__ import annotations

import copy
import json
import math
from numbers import Real
from fractions import Fraction
from collections.abc import Sequence


def _probabilities(values):
    """Validate the original real domain, then return fresh float rows."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("probabilities must be a nonempty batch of rows")
    width = None
    rows = []
    for row in values:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
            raise ValueError("each probability row must be nonempty")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("probability rows must have equal lengths")
        converted = []
        for p in row:
            if isinstance(p, bool) or not isinstance(p, Real):
                raise ValueError("probabilities must be real numbers, not booleans")
            if not 0 <= p <= 1:
                raise ValueError("probabilities must be finite and in [0, 1]")
            try:
                converted_p = float(p)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("probabilities must convert to finite floats") from exc
            if not math.isfinite(converted_p):
                raise ValueError("probabilities must convert to finite floats")
            converted.append(converted_p)
        rows.append(converted)
    return rows


def expected_losses(probabilities):
    """Return [batch][time][action] losses; action order is go, wait.

    The stipulated loss matrix has state rows clear/blocked and action columns
    go/wait: [[1,3],[11,3]]. Each result contracts the state probability axis.
    Returns rounded loss values. Action selection applies the derived threshold
    to converted probabilities instead of comparing these rounded outputs.
    """
    rows = _probabilities(probabilities)
    return [[[1.0 + 10.0 * p, 3.0] for p in row] for row in rows]


def choose_actions(probabilities):
    """Apply the derived threshold to validated binary floating-point inputs.

    Loss values are rounded; comparing them can create a false tie.
    The representable value immediately below 0.2 selects go; 0.2 selects wait.
    The policy acts on the values after conversion by _probabilities.
    """
    rows = _probabilities(probabilities)
    return [["go" if p < 0.2 else "wait" for p in row] for row in rows]


def storage_bytes(shape, itemsize):
    """Raw dense storage only; reject invalid dimensions and element sizes."""
    if isinstance(itemsize, bool) or not isinstance(itemsize, int) or itemsize <= 0:
        raise ValueError("itemsize must be a positive integer")
    dims = tuple(shape)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in dims):
        raise ValueError("shape dimensions must be nonnegative integers")
    return math.prod(dims) * itemsize


def split_features(features, groups):
    """Split one ordered feature vector into groups, with d = group*width+local."""
    if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
        raise ValueError("groups must be a positive integer")
    data = list(features)
    if not data or len(data) % groups:
        raise ValueError("nonempty feature count must be divisible by groups")
    width = len(data) // groups
    return [data[i * width:(i + 1) * width] for i in range(groups)]


def run_checks():
    """Behavior checks; optional NumPy checks are explicitly marked if skipped."""
    p = [[0.0, 0.1, 0.2, 0.4, 1.0], [0.9, 0.0, 0.2, 0.1, 0.5]]
    before = copy.deepcopy(p)
    expected = [[[1., 3.], [2., 3.], [3., 3.], [5., 3.], [11., 3.]],
                [[10., 3.], [1., 3.], [3., 3.], [2., 3.], [6., 3.]]]
    assert expected_losses(p) == expected
    assert choose_actions(p)[0] == ["go", "go", "wait", "wait", "wait"]
    assert p == before
    boundary_values = [0.0, math.nextafter(0.2, 0.0), 0.2,
                       math.nextafter(0.2, 1.0), 1.0]
    assert choose_actions([boundary_values]) == [["go", "go", "wait", "wait", "wait"]]
    assert choose_actions([[Fraction(1, 3)]]) == [["wait"]]
    # A changed robot cannot change another robot's output.
    changed = copy.deepcopy(p)
    changed[1] = [1.0] * 5
    assert expected_losses(changed)[0] == expected_losses(p)[0]
    output = expected_losses(p)
    output[0][0][0] = 999
    assert p == before and expected_losses(p) == expected
    rejected = 0
    for invalid in ([], [[]], [[0.1], [0.2, 0.3]], [[-0.1]], [[1.1]],
                    [[float("nan")]], [[float("inf")]], [[True]], [["0.2"]], None, [[10**400]], [[-(10**400)]],
                    [[Fraction(1) + Fraction(1, 10**100)]],
                    [[-Fraction(1, 10**400)]]):
        try:
            expected_losses(invalid)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("invalid probability input was accepted")
    assert rejected == 14
    assert storage_bytes((32, 224, 224, 3), 4) == 19267584
    assert storage_bytes((32, 224, 224, 3), 4) / 2**20 == 18.375
    assert storage_bytes((), 8) == 8
    assert storage_bytes((0, 4), 8) == 0
    assert split_features(range(8), 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert split_features(range(8), 2)[1][2] == 6
    assert sum(split_features(range(8), 2), []) == list(range(8))
    try:
        split_features(range(8), 3)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid feature factorization accepted")
    assert 4**10 == 1048576
    assert math.isclose((5-2)/(22-2), 0.15)
    assert 3 * 5 * 2 * 4 == 120
    # Reducing first imposes one constant action; selecting first permits adaptation.
    two_times = expected_losses([[0.1, 0.4]])[0]
    constant_action_means = [sum(x[a] for x in two_times) / 2 for a in range(2)]
    adaptive_mean = sum(min(x) for x in two_times) / 2
    assert constant_action_means == [3.5, 3.0]
    assert adaptive_mean == 2.5 < min(constant_action_means)
    # Worked two-key mask: zeroing a zero score excludes nothing.
    weights = [math.exp(0.0) / (2 * math.exp(0.0))] * 2
    assert sum(w * v for w, v in zip(weights, [2.0, 10.0])) == 6.0
    assert sum(w * v for w, v in zip([1.0, 0.0], [2.0, 10.0])) == 2.0
    report = {"standard_library": "passed", "invalid_probability_cases_rejected": rejected,
              "policy_boundary": "converted float below 0.2: go; at or above: wait",
              "policy_boundary_cases_checked": len(boundary_values),
              "numpy": "not installed; not run", "speed_benchmark": "not performed"}
    try:
        import numpy as np
    except ImportError:
        return report
    q = np.stack((1 - np.array(p), np.array(p)), axis=-1)
    costs = np.array([[1., 3.], [11., 3.]])
    np.testing.assert_allclose(q @ costs, expected)
    x = np.array([[[2., 1.], [0., 3.]]])
    w = np.diag([1., 2.])
    np.testing.assert_array_equal(x @ w, [[[2., 2.], [0., 6.]]])
    np.testing.assert_array_equal(np.einsum("btd,dk->btk", x, w), x @ w)
    u = np.array([[[1., 0.], [0., 1.]]])
    v = np.array([[[1., 1.], [2., 0.]]])
    np.testing.assert_array_equal(np.einsum("bid,bjd->bij", u, v),
                                  [[[1., 2.], [1., 0.]]])
    base = np.arange(6).reshape(2, 3)
    view = base[0]
    view[0] = -1
    assert base[0, 0] == -1 and np.shares_memory(base, view)
    independent = base.copy()
    independent[0, 0] = -99
    assert base[0, 0] == -1
    try:
        np.zeros((3, 5, 2)) + np.zeros((3,))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid broadcast accepted")
    elementwise = np.array([1., 4., 9.])
    np.testing.assert_array_equal(elementwise**2, [1., 16., 81.])
    np.testing.assert_array_equal(np.sqrt(elementwise), [1., 2., 3.])
    np.testing.assert_array_equal(elementwise + elementwise, [2., 8., 18.])
    np.testing.assert_array_equal(3 * elementwise, [3., 12., 27.])
    np.testing.assert_array_equal(np.tril(np.ones((3, 3))), [[1,0,0],[1,1,0],[1,1,1]])
    np.testing.assert_array_equal(np.triu(np.ones((3, 3))), [[1,1,1],[0,1,1],[0,0,1]])
    report["numpy"] = "passed"
    return report


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
