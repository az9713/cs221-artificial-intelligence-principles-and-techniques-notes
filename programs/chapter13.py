"""Original finite Bayesian inference and Gibbs experiments; stdlib only."""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product, combinations
from math import lcm, prod
from random import Random
from collections.abc import Mapping


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low},{high}]")
    return value


def probability(value):
    if type(value) not in (int, F):
        raise ValueError("probabilities require int/Fraction, excluding bool")
    value = F(value)
    if not 0 <= value <= 1 or value.denominator > 10**9:
        raise ValueError("probability outside [0,1] or denominator budget")
    return value


def draw(row, rng):
    """Exact rational categorical draw with an explicit integer budget."""
    if not isinstance(rng, Random):
        raise ValueError("rng must be a local random.Random")
    if not isinstance(row, (tuple, list)) or not row:
        raise ValueError("nonempty probability row required")
    row = tuple(probability(v) for v in row)
    if sum(row) != 1:
        raise ValueError("probabilities must sum exactly to one")
    denominator = lcm(*(p.denominator for p in row))
    if denominator > 10**12:
        raise ValueError("categorical common denominator exceeds 10**12")
    pick = rng.randrange(denominator)
    for i, p in enumerate(row):
        pick -= p.numerator * (denominator // p.denominator)
        if pick < 0:
            return i
    raise RuntimeError("unreachable normalized categorical branch")


@dataclass(frozen=True, init=False)
class Network:
    """Immutable finite kernels in an explicitly supplied topological order."""
    names: tuple
    sizes: tuple
    parents: tuple
    tables: tuple

    def __init__(self, names, sizes, parents, tables):
        if not all(isinstance(v, (list, tuple))
                   for v in (names, sizes, parents, tables)):
            raise ValueError("network fields must be lists or tuples")
        names, sizes = tuple(names), tuple(sizes)
        if not 1 <= len(names) <= 8:
            raise ValueError("one to eight variables required")
        if not all(type(v) is str and v for v in names):
            raise ValueError("variable names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("distinct variable names required")
        if not len(sizes) == len(parents) == len(tables) == len(names):
            raise ValueError("network field lengths differ")
        sizes = tuple(integer(v, 1, 4, "domain size") for v in sizes)
        if prod(sizes) > 4096:
            raise ValueError("joint state budget exceeds 4096")
        pp, tt = [], []
        for i, (pa, table) in enumerate(zip(parents, tables)):
            if not isinstance(pa, (tuple, list)):
                raise ValueError("parent labels require tuple or list")
            if not all(type(v) is str and v in names[:i] for v in pa):
                raise ValueError("parents must precede child in given order")
            if len(set(pa)) != len(pa):
                raise ValueError("duplicate parent")
            indices = tuple(names.index(v) for v in pa)
            keys = tuple(product(*(range(sizes[j]) for j in indices)))
            if not isinstance(table, Mapping):
                raise ValueError("conditional table must be a mapping")
            if not all(type(k) is tuple and len(k) == len(indices)
                       and all(type(v) is int for v in k) for k in table):
                raise ValueError("parent keys must be exact integer tuples")
            if set(table) != set(keys):
                raise ValueError("table must contain every parent assignment")
            rows = []
            for key in keys:
                raw = table[key]
                if not isinstance(raw, (list, tuple)) or len(raw) != sizes[i]:
                    raise ValueError("conditional row has wrong domain size")
                row = tuple(probability(v) for v in raw)
                if sum(row) != 1:
                    raise ValueError("conditional row not normalized")
                rows.append((key, row))
            pp.append(indices)
            tt.append(tuple(rows))
        for name, value in zip(("names", "sizes", "parents", "tables"),
                               (names, sizes, tuple(pp), tuple(tt))):
            object.__setattr__(self, name, value)

    def assignment(self, values, complete=False):
        if not isinstance(values, Mapping):
            raise ValueError("assignment must be a mapping")
        if not all(type(k) is str and k in self.names for k in values):
            raise ValueError("unknown variable")
        if complete and set(values) != set(self.names):
            raise ValueError("complete assignment required")
        return {self.names.index(k): integer(v, 0,
                self.sizes[self.names.index(k)] - 1, "state value")
                for k, v in values.items()}

    def query(self, names, evidence):
        if not isinstance(names, (tuple, list)) or not names:
            raise ValueError("nonempty query tuple/list required")
        if not all(type(v) is str and v in self.names for v in names):
            raise ValueError("unknown query variable")
        if len(set(names)) != len(names):
            raise ValueError("duplicate query variable")
        indices = tuple(self.names.index(v) for v in names)
        if set(indices) & set(evidence):
            raise ValueError("query and evidence must be disjoint")
        return indices

    def states(self):
        return tuple(product(*(range(d) for d in self.sizes)))

    def factor(self, i, state):
        key = tuple(state[j] for j in self.parents[i])
        for k, row in self.tables[i]:
            if key == k:
                return row[state[i]]
        raise RuntimeError("unreachable checked parent key")

    def joint_tuple(self, state):
        if type(state) is not tuple or len(state) != len(self.names):
            raise ValueError("complete state tuple required")
        for i, value in enumerate(state):
            integer(value, 0, self.sizes[i] - 1, "state value")
        return prod((self.factor(i, state) for i in range(len(state))),
                    start=F(1))

    def support(self, evidence):
        evidence = self.assignment(evidence)
        weights = {s: self.joint_tuple(s) for s in self.states()
                   if all(s[i] == v for i, v in evidence.items())}
        weights = {s: p for s, p in weights.items() if p > 0}
        mass = sum(weights.values(), F(0))
        if mass == 0:
            raise ValueError("impossible evidence has zero probability")
        return mass, {s: p / mass for s, p in weights.items()}

    def posterior(self, query, evidence):
        indices = self.query(query, self.assignment(evidence))
        mass, target = self.support(evidence)
        result = {s: F(0) for s in
                  product(*(range(self.sizes[i]) for i in indices))}
        for state, weight in target.items():
            result[tuple(state[i] for i in indices)] += weight
        return mass, result

    def conditional(self, state, name):
        """Local factors cancel only on a supported complete assignment."""
        current_mass = self.joint_tuple(state)
        if type(name) is not str or name not in self.names:
            raise ValueError("unknown update variable")
        if current_mass == 0:
            raise ValueError("Gibbs initialization must have positive support")
        i = self.names.index(name)
        touched = [j for j in range(len(self.names))
                   if j == i or i in self.parents[j]]
        weights = []
        for value in range(self.sizes[i]):
            candidate = tuple(value if j == i else s
                              for j, s in enumerate(state))
            weights.append(prod((self.factor(j, candidate) for j in touched),
                                start=F(1)))
        mass = sum(weights, F(0))
        if mass <= 0:
            raise ValueError("unsupported conditional fiber")
        return tuple(p / mass for p in weights)

    def blanket(self, name):
        if type(name) is not str or name not in self.names:
            raise ValueError("unknown blanket variable")
        i = self.names.index(name)
        children = {j for j, pa in enumerate(self.parents) if i in pa}
        indices = set(self.parents[i]) | children
        for child in children:
            indices.update(self.parents[child])
        indices.discard(i)
        return tuple(self.names[j] for j in sorted(indices))


def network(value):
    if type(value) is not Network:
        raise ValueError("a validated Network is required")
    return value


def kernel(model, evidence, order):
    """Exact ordered coordinate/block transition; at most 64 support states."""
    model = network(model)
    ev = model.assignment(evidence)
    if not isinstance(order, (tuple, list)):
        raise ValueError("order must be tuple/list of nonempty blocks")
    blocks, seen = [], []
    for block in order:
        if not isinstance(block, (tuple, list)) or not block:
            raise ValueError("each update block must be nonempty")
        if not all(type(v) is str and v in model.names for v in block):
            raise ValueError("unknown block variable")
        indices = tuple(model.names.index(v) for v in block)
        if len(set(indices)) != len(indices) or set(indices) & set(ev):
            raise ValueError("duplicate/evidence update coordinate")
        blocks.append(set(indices))
        seen.extend(indices)
    if len(set(seen)) != len(seen):
        raise ValueError("each coordinate may appear at most once")
    _, target = model.support(evidence)
    states = tuple(target)
    if len(states) > 64:
        raise ValueError("transition matrix support exceeds 64 states")
    size = len(states)
    result = tuple(tuple(F(i == j) for j in range(size)) for i in range(size))
    for block in blocks:
        rows = []
        for x in states:
            fiber = [y for y in states
                     if all(x[j] == y[j] for j in range(len(x))
                            if j not in block)]
            mass = sum((target[y] for y in fiber), F(0))
            rows.append(tuple(target[y] / mass if y in fiber else F(0)
                              for y in states))
        result = multiply(result, tuple(rows))
    return states, tuple(target.values()), result


def multiply(left, right):
    """Internal exact square matrix multiplication on checked kernels."""
    n = len(left)
    return tuple(tuple(sum((left[i][k] * right[k][j] for k in range(n)), F(0))
                       for j in range(n)) for i in range(n))


def components(model, evidence):
    """Positive-support components under single-coordinate changes."""
    _, target = network(model).support(evidence)
    states = tuple(target)
    if len(states) > 256:
        raise ValueError("support graph budget exceeds 256 states")
    unseen, result = set(states), []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, found = [start], {start}
        while todo:
            x = todo.pop()
            neighbors = [y for y in unseen
                         if sum(a != b for a, b in zip(x, y)) == 1]
            unseen.difference_update(neighbors)
            found.update(neighbors)
            todo.extend(neighbors)
        result.append(tuple(sorted(found)))
    return tuple(result)


def gibbs(model, query, evidence, initial, sweeps, seed=0, burn=0):
    model = network(model)
    integer(sweeps, 1, 100000, "retained sweeps")
    integer(burn, 0, 100000 - sweeps, "burn sweeps")
    integer(seed, -(2**63), 2**63 - 1, "seed")
    ev = model.assignment(evidence)
    indices = model.query(query, ev)
    init = model.assignment(initial, complete=True)
    if any(init[i] != value for i, value in ev.items()):
        raise ValueError("initial assignment disagrees with evidence")
    state = tuple(init[i] for i in range(len(model.names)))
    if model.joint_tuple(state) == 0:
        raise ValueError("Gibbs initialization must have positive support")
    rng = Random(seed)
    counts = {s: 0 for s in
              product(*(range(model.sizes[i]) for i in indices))}
    for t in range(burn + sweeps):
        for i, name in enumerate(model.names):
            if i not in ev:
                value = draw(model.conditional(state, name), rng)
                state = tuple(value if j == i else s
                              for j, s in enumerate(state))
        if t >= burn:
            counts[tuple(state[i] for i in indices)] += 1
    return {q: F(n, sweeps) for q, n in counts.items()}


def rejection(model, query, evidence, proposals, seed=0):
    model = network(model)
    integer(proposals, 1, 100000, "proposal count")
    integer(seed, -(2**63), 2**63 - 1, "seed")
    ev = model.assignment(evidence)
    indices = model.query(query, ev)
    model.support(evidence)
    counts = {s: 0 for s in
              product(*(range(model.sizes[i]) for i in indices))}
    accepted, rng = 0, Random(seed)
    for _ in range(proposals):
        state = []
        for i in range(len(model.names)):
            key = tuple(state[j] for j in model.parents[i])
            row = dict(model.tables[i])[key]
            state.append(draw(row, rng))
        if all(state[i] == value for i, value in ev.items()):
            accepted += 1
            counts[tuple(state[i] for i in indices)] += 1
    estimate = ({q: F(n, accepted) for q, n in counts.items()}
                if accepted else None)
    return accepted, estimate


def separated(model, left, right, given=()):
    """Ancestor restriction, moralization, evidence removal, connectivity."""
    model = network(model)
    sets = []
    for raw in (left, right, given):
        if not isinstance(raw, (tuple, list)):
            raise ValueError("graph query sets must be tuples/lists")
        if not all(type(v) is str and v in model.names for v in raw):
            raise ValueError("unknown graph query variable")
        if len(set(raw)) != len(raw):
            raise ValueError("duplicate graph query variable")
        sets.append({model.names.index(v) for v in raw})
    a, b, s = sets
    if not a or not b or a & b or a & s or b & s:
        raise ValueError("nonempty disjoint queries/evidence required")
    ancestral = a | b | s
    todo = list(ancestral)
    while todo:
        i = todo.pop()
        for parent in model.parents[i]:
            if parent not in ancestral:
                ancestral.add(parent)
                todo.append(parent)
    adjacency = {i: set() for i in ancestral}
    for child in ancestral:
        pa = model.parents[child]
        for x, y in [(child, j) for j in pa] + list(combinations(pa, 2)):
            adjacency[x].add(y)
            adjacency[y].add(x)
    found, todo = set(a), list(a)
    while todo:
        i = todo.pop()
        for j in adjacency[i] - s - found:
            found.add(j)
            todo.append(j)
    return not bool(found & b)


def telephone(error=F(1, 5)):
    error = probability(error)
    return Network(("A", "B", "C"), (2, 2, 2), ((), ("A",), ("B",)),
                   ({(): (F(1, 2), F(1, 2))},
                    {(0,): (1 - error, error), (1,): (error, 1 - error)},
                    {(0,): (1 - error, error), (1,): (error, 1 - error)}))


def alarm():
    return Network(("B", "E", "A"), (2, 2, 2), ((), (), ("B", "E")),
                   ({(): (F(19, 20), F(1, 20))},
                    {(): (F(19, 20), F(1, 20))},
                    {be: (F(not any(be)), F(any(be)))
                     for be in product(range(2), repeat=2)}))


def locked():
    return Network(("A", "B"), (2, 2), ((), ("A",)),
                   ({(): (F(1, 2), F(1, 2))},
                    {(0,): (F(1), F(0)), (1,): (F(0), F(1))}))


def rare():
    return Network(("A", "B"), (2, 2), ((), ("A",)),
                   ({(): (F(1, 2), F(1, 2))},
                    {(0,): (F(9999, 10000), F(1, 10000)),
                     (1,): (F(4999, 5000), F(1, 5000))}))


def demo():
    tel = telephone()
    mass, post = tel.posterior(("A",), {"C": 1})
    print(f"telephone evidence={mass}, posterior A=1:{post[(1,)]}")
    print("B | A=1,C=1:", tel.conditional((1, 0, 1), "B"))
    states, pi, sweep = kernel(tel, {"C": 1}, (("A",), ("B",)))
    invariant = all(sum((pi[i] * sweep[i][j] for i in range(len(pi))), F(0))
                    == pi[j] for j in range(len(pi)))
    reversible = all(pi[i] * sweep[i][j] == pi[j] * sweep[j][i]
                     for i in range(len(pi)) for j in range(len(pi)))
    print(f"ordered sweep invariant={invariant}, reversible={reversible}")
    print("telephone Gibbs A=1, 2000 sweeps:",
          gibbs(tel, ("A",), {"C": 1}, {"A": 0, "B": 0, "C": 1},
                2000, seed=13)[(1,)])
    print("alarm evidence/posterior:", alarm().posterior(("B",), {"A": 1}))
    print("alarm support components:", components(alarm(), {"A": 1}))
    print("locked support components:", components(locked(), {}))
    locked_hist = gibbs(locked(), ("A",), {}, {"A": 0, "B": 0}, 100, seed=13)
    print("locked Gibbs A=1:", locked_hist[(1,)])
    print("rare evidence/posterior:", rare().posterior(("A",), {"B": 1}))
    print("rare rejection accepted, estimate:",
          rejection(rare(), ("A",), {"B": 1}, 10, seed=13))
    print("alarm parent independence: marginal=",
          separated(alarm(), ("B",), ("E",)), " given alarm=",
          separated(alarm(), ("B",), ("E",), ("A",)), sep="")
    print("Exact finite fixtures do not certify mixing or model accuracy.")


if __name__ == "__main__":
    demo()
