"""Exact finite parameter estimation and a tied binary-rating EM example.

Run with Python 3, standard library only. Importing does not print or write.
The bounded contract intentionally rejects floats and bools: numbers are
Python ints or Fractions, with at most 4096 numerator/denominator bits.
"""

from fractions import Fraction
from itertools import product


MAX_BITS = 4096
MAX_ROWS = 4096


def exact(value):
    """Validate a bounded nonnegative exact number, without coercing floats."""
    if type(value) not in (int, Fraction):
        raise ValueError("numbers must be ints or Fractions, not bools/floats")
    result = Fraction(value)
    if result < 0 or max(result.numerator.bit_length(),
                         result.denominator.bit_length()) > MAX_BITS:
        raise ValueError("negative number or exact-integer bit limit exceeded")
    return result


def row(values):
    """A normalized finite probability row; structural zeros are allowed."""
    if type(values) not in (tuple, list) or not 1 <= len(values) <= 8:
        raise ValueError("a probability row must have 1 through 8 entries")
    result = tuple(exact(v) for v in values)
    if sum(result) != 1:
        raise ValueError("a probability row must sum exactly to one")
    return result


def normalize(counts, smoothing=0):
    """Return a smoothed row, or None for an unidentified unsmoothed row."""
    if type(counts) not in (tuple, list) or not 1 <= len(counts) <= 8:
        raise ValueError("a count row must have 1 through 8 entries")
    lam = exact(smoothing)
    adjusted = tuple(exact(exact(c) + lam) for c in counts)
    total = exact(sum(adjusted))
    return tuple(exact(c / total) for c in adjusted) if total else None


def schema(specification):
    """Check (variable, cardinality, ordered parents, table name) records.

    Values are integer indices 0,...,cardinality-1. Parent names must precede
    their child. Shared tables need identical ordered parent cardinalities
    and child cardinality. This topological interface excludes cycles.
    """
    if (type(specification) not in (tuple, list)
            or not 1 <= len(specification) <= 16):
        raise ValueError("schema requires 1 through 16 nodes")
    known, signatures, records = {}, {}, []
    for item in specification:
        if type(item) not in (tuple, list) or len(item) != 4:
            raise ValueError("each node requires four fields")
        name, size, parents, table = item
        if (type(name) is not str or not name or name in known
                or type(table) is not str or not table):
            raise ValueError("unique nonempty node and nonempty table names")
        if type(size) is not int or not 1 <= size <= 8:
            raise ValueError("cardinality must be an integer from 1 to 8")
        if (type(parents) not in (tuple, list)
                or any(type(p) is not str for p in parents)
                or len(set(parents)) != len(parents)
                or any(p not in known for p in parents)):
            raise ValueError("distinct parents must precede their child")
        parents = tuple(parents)
        signature = (tuple(known[p] for p in parents), size)
        if table in signatures and signatures[table] != signature:
            raise ValueError("shared table signatures must agree")
        signatures[table] = signature
        known[name] = size
        records.append((name, size, parents, table))
    number_rows = sum(_row_count(s[0]) for s in signatures.values())
    if number_rows > MAX_ROWS:
        raise ValueError("schema exceeds 4096 distinct parameter rows")
    return tuple(records), known, signatures


def _row_count(sizes):
    result = 1
    for size in sizes:
        result *= size
    return result


def fit_complete(specification, assignments, smoothing=0):
    """Count declared factor occurrences; return (counts, parameter rows).

    Up to 256 complete assignments; every declared row/category is present
    in the result. None means the raw likelihood leaves that row unidentified.
    No missing keys, extra keys, bool-valued states or unknown states pass.
    """
    records, domains, signatures = schema(specification)
    lam = exact(smoothing)
    if type(assignments) not in (tuple, list) or len(assignments) > 256:
        raise ValueError("assignments must be a list/tuple of length <=256")
    counts = {}
    for table, (sizes, size) in signatures.items():
        for parent_values in product(*(range(s) for s in sizes)):
            counts[(table, parent_values)] = [0] * size
    for assignment in assignments:
        if type(assignment) is not dict or set(assignment) != set(domains):
            raise ValueError("assignment keys must match all declared nodes")
        for name, size in domains.items():
            value = assignment[name]
            if type(value) is not int or not 0 <= value < size:
                raise ValueError("state must be a declared integer index")
        for name, _, parents, table in records:
            parent_values = tuple(assignment[p] for p in parents)
            counts[(table, parent_values)][assignment[name]] += 1
    counts = {key: tuple(value) for key, value in counts.items()}
    return counts, {key: normalize(c, lam) for key, c in counts.items()}


def model(parameters):
    """Two mixing probabilities and two tied binary-rating kernels."""
    if type(parameters) not in (tuple, list) or len(parameters) != 2:
        raise ValueError("model is (mixing row, two component rows)")
    mixing, kernels = parameters
    mixing = row(mixing)
    if (len(mixing) != 2 or type(kernels) not in (tuple, list)
            or len(kernels) != 2):
        raise ValueError("exactly two components are required")
    kernels = tuple(row(k) for k in kernels)
    if any(len(k) != 2 for k in kernels):
        raise ValueError("each component has two rating probabilities")
    return mixing, kernels


def observations(data):
    """One through 256 ordered pairs of binary rating indices."""
    if type(data) not in (tuple, list) or not 1 <= len(data) <= 256:
        raise ValueError("data require 1 through 256 rating pairs")
    checked = []
    for pair in data:
        if (type(pair) not in (tuple, list) or len(pair) != 2
                or any(type(v) is not int or v not in (0, 1) for v in pair)):
            raise ValueError("ratings must be integer indices zero or one")
        checked.append(tuple(pair))
    return tuple(checked)


def _joint(parameters, pair):
    mixing, kernels = parameters
    a, b = pair
    return tuple(exact(mixing[g] * kernels[g][a] * kernels[g][b])
                 for g in range(2))


def posterior(parameters, pair):
    """Exact P(component | pair); zero evidence has no posterior."""
    parameters = model(parameters)
    pair = observations([pair])[0]
    joint = _joint(parameters, pair)
    evidence = exact(sum(joint))
    if not evidence:
        raise ValueError("zero evidence: posterior is undefined")
    return tuple(exact(v / evidence) for v in joint)


def likelihood(parameters, data):
    """Full observed-data likelihood product; zero likelihood is allowed."""
    parameters, data = model(parameters), observations(data)
    result = Fraction(1)
    for pair in data:
        result = exact(result * exact(sum(_joint(parameters, pair))))
    return result


def em_step(parameters, data):
    """One exact unsmoothed EM step and unused-component flags.

    A zero expected component mass makes its rating kernel unidentified;
    retain the old kernel as a valid maximizer, explicitly flagging this.
    """
    parameters, data = model(parameters), observations(data)
    masses = [Fraction(0), Fraction(0)]
    counts = [[Fraction(0), Fraction(0)] for _ in range(2)]
    for pair in data:
        weights = posterior(parameters, pair)
        for g, weight in enumerate(weights):
            masses[g] = exact(masses[g] + weight)
            for value in pair:
                counts[g][value] = exact(counts[g][value] + weight)
    mixing = tuple(exact(m / len(data)) for m in masses)
    unused = tuple(not m for m in masses)
    kernels = tuple(parameters[1][g] if unused[g] else normalize(counts[g])
                    for g in range(2))
    return model((mixing, kernels)), unused


def run_em(parameters, data, iterations):
    """Zero through five updates, with exact full-objective comparisons.

    A bit-limit violation raises ValueError rather than rounding an exact
    probability or pretending that unsupported arithmetic has converged.
    """
    if type(iterations) is not int or not 0 <= iterations <= 5:
        raise ValueError("iterations must be an integer from zero to five")
    parameters, data = model(parameters), observations(data)
    history = [likelihood(parameters, data)]
    for _ in range(iterations):
        parameters, _ = em_step(parameters, data)
        current = likelihood(parameters, data)
        if current < history[-1]:
            raise AssertionError("exact unsmoothed EM decreased likelihood")
        history.append(current)
    return parameters, tuple(history)


def main():
    spec = (("G", 2, (), "genre"),
            ("R1", 5, ("G",), "rating"),
            ("R2", 5, ("G",), "rating"))
    data = [dict(G=g, R1=a - 1, R2=b - 1) for g, a, b in
            [(0, 4, 5), (0, 4, 4), (0, 5, 3), (1, 1, 2), (1, 5, 4)]]
    counts, fitted = fit_complete(spec, data)
    print("pooled drama counts:", counts[("rating", (0,))])
    print("pooled drama row:", ", ".join(map(str, fitted[("rating", (0,))])))
    print("unseen row:", normalize([0, 0]))
    add_one = normalize([1, 0, 0, 1, 0], 1)
    print("add-one ratings:", ", ".join(map(str, add_one)))
    initial = ((Fraction(1, 2), Fraction(1, 2)),
               ((Fraction(3, 5), Fraction(2, 5)),
                (Fraction(2, 5), Fraction(3, 5))))
    pairs = [(0, 0), (1, 1)]
    first_weights = posterior(initial, pairs[0])
    print("first posterior:", ", ".join(map(str, first_weights)))
    updated, history = run_em(initial, pairs, 1)
    print("first component:", ", ".join(map(str, updated[1][0])))
    print("likelihood:", " -> ".join(map(str, history)))
    symmetric = ((Fraction(1, 2), Fraction(1, 2)),
                 ((Fraction(1, 2), Fraction(1, 2)),) * 2)
    fixed, _ = run_em(symmetric, pairs, 5)
    print("symmetric fixed point:", fixed == symmetric)
    print("opposite perturbation improves:",
          likelihood(initial, pairs) > likelihood(symmetric, pairs))
    unused_model = ((1, 0), ((Fraction(1, 2), Fraction(1, 2)), (1, 0)))
    _, unused = em_step(unused_model, [(0, 1)])
    print("unused component flags:", unused)
    impossible = ((1, 0), ((1, 0), (0, 1)))
    try:
        posterior(impossible, (1, 1))
    except ValueError as error:
        print("zero-evidence rejection:", error)


if __name__ == "__main__":
    main()
