"""Original finite Bayesian-network enumeration and rejection diagnostics.

Standard library only. Model functions have no filesystem/network effects.
Running this file prints exact teaching fixtures as JSON. No clinical model.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import lcm
from random import Random
import json
import re

MAX_ASSIGNMENTS = 65536
MAX_PROPOSALS = 100000
MAX_BITS = 128
ZERO = Fraction(0)
ONE = Fraction(1)


def integer(value, low, high, label):
    """Validate a bounded built-in integer; booleans are not counts."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer in [{low}, {high}]')
    return value


def probability(value):
    """Accept exact built-in int/Fraction probabilities, never floats."""
    if type(value) not in (int, Fraction):
        raise ValueError('probability must be an exact int or Fraction')
    result = Fraction(value)
    if not ZERO <= result <= ONE:
        raise ValueError('probability is outside [0, 1]')
    if max(result.numerator.bit_length(),
           result.denominator.bit_length()) > MAX_BITS:
        raise ValueError('probability exceeds the 128-bit input budget')
    return result


def name(value):
    """Use short ASCII identifiers so coordinate meaning is explicit."""
    if type(value) is not str or not re.fullmatch(r'[A-Za-z]\w{0,31}',
                                                value, flags=re.ASCII):
        raise ValueError('variable name must be a short ASCII identifier')
    return value


def sequence(value, label):
    if type(value) not in (tuple, list):
        raise ValueError(f'{label} must be a tuple or list')
    return tuple(value)


@dataclass(frozen=True)
class Node:
    """One finite node: parent axes first, own value axis last."""

    label: str
    size: int
    parents: tuple
    rows: tuple

    def __post_init__(self):
        name(self.label)
        integer(self.size, 1, 8, 'domain size')
        parents = sequence(self.parents, 'parents')
        if len(parents) > 12 or len(set(map(name, parents))) != len(parents):
            raise ValueError('parents must be distinct variable names')
        rows = sequence(self.rows, 'rows')
        if not 1 <= len(rows) <= MAX_ASSIGNMENTS:
            raise ValueError('kernel row count exceeds the budget')
        frozen = []
        for row in rows:
            row = tuple(map(probability, sequence(row, 'kernel row')))
            if len(row) != self.size or sum(row, ZERO) != ONE:
                raise ValueError('every kernel row must normalize')
            denominator = lcm(*(q.denominator for q in row))
            if denominator.bit_length() > 256:
                raise ValueError('kernel sampling denominator exceeds budget')
            frozen.append(row)
        object.__setattr__(self, 'parents', parents)
        object.__setattr__(self, 'rows', tuple(frozen))


@dataclass(frozen=True)
class Network:
    """Nodes in supplied topological order; domains are range(node.size)."""

    nodes: tuple

    def __post_init__(self):
        nodes = sequence(self.nodes, 'nodes')
        if not 1 <= len(nodes) <= 12:
            raise ValueError('network must contain 1 to 12 nodes')
        preceding = {}
        assignments = 1
        for node in nodes:
            if type(node) is not Node or node.label in preceding:
                raise ValueError('nodes must be distinct validated Node values')
            if any(parent not in preceding for parent in node.parents):
                raise ValueError('parents must precede their child')
            row_count = 1
            for parent in node.parents:
                row_count *= preceding[parent]
            if len(node.rows) != row_count:
                raise ValueError('kernel rows do not match parent domains')
            preceding[node.label] = node.size
            assignments *= node.size
            if assignments > MAX_ASSIGNMENTS:
                raise ValueError('full assignment space exceeds budget')
        object.__setattr__(self, 'nodes', nodes)

    @property
    def labels(self):
        return tuple(node.label for node in self.nodes)

    def _row(self, node, assignment):
        index = 0
        sizes = {item.label: item.size for item in self.nodes}
        for parent in node.parents:
            index = index * sizes[parent] + assignment[parent]
        return node.rows[index]

    def _specification(self, query, evidence):
        query = sequence(query, 'query')
        if len(set(map(name, query))) != len(query):
            raise ValueError('query names must be distinct')
        if type(evidence) is not dict:
            raise ValueError('evidence must be a built-in dictionary')
        sizes = {node.label: node.size for node in self.nodes}
        if any(label not in sizes for label in query):
            raise ValueError('unknown query variable')
        for label, value in evidence.items():
            name(label)
            if label not in sizes:
                raise ValueError('unknown evidence variable')
            integer(value, 0, sizes[label] - 1, 'evidence value')
        if set(query) & set(evidence):
            raise ValueError('query and evidence must be disjoint')
        states = tuple(product(*(range(sizes[label]) for label in query)))
        return query, dict(evidence), states

    def joint(self):
        """Return a fresh complete exact table, including zero-mass states."""
        result = {}
        for state in product(*(range(node.size) for node in self.nodes)):
            assignment = dict(zip(self.labels, state))
            mass = ONE
            for node, value in zip(self.nodes, state):
                mass *= self._row(node, assignment)[value]
            result[state] = mass
        return result

    def infer(self, query, evidence):
        """Return (normalized query table, evidence mass); reject zero mass."""
        query, evidence, states = self._specification(query, evidence)
        weights = dict.fromkeys(states, ZERO)
        for state, mass in self.joint().items():
            assignment = dict(zip(self.labels, state))
            if all(assignment[label] == value
                   for label, value in evidence.items()):
                key = tuple(assignment[label] for label in query)
                weights[key] += mass
        normalizer = sum(weights.values(), ZERO)
        if normalizer == ZERO:
            raise ValueError('evidence has zero model probability')
        return {key: mass / normalizer for key, mass in weights.items()}, normalizer

    def sample(self, rng):
        """Generate one fresh assignment using a caller-owned Random stream."""
        if type(rng) is not Random:
            raise ValueError('rng must be a built-in Random instance')
        assignment = {}
        for node in self.nodes:
            row = self._row(node, assignment)
            denominator = lcm(*(q.denominator for q in row))
            draw = rng.randrange(denominator)
            cumulative = 0
            for value, q in enumerate(row):
                cumulative += q.numerator * (denominator // q.denominator)
                if draw < cumulative:
                    assignment[node.label] = value
                    break
        return assignment

    def reject(self, query, evidence, proposals, seed):
        """Seeded finite diagnostics; zero accepts give estimate=None."""
        query, evidence, states = self._specification(query, evidence)
        integer(proposals, 0, MAX_PROPOSALS, 'proposal count')
        integer(seed, 0, 2**32 - 1, 'seed')
        counts = dict.fromkeys(states, 0)
        rng = Random(seed)
        accepted = 0
        for _ in range(proposals):
            assignment = self.sample(rng)
            if all(assignment[label] == value
                   for label, value in evidence.items()):
                accepted += 1
                counts[tuple(assignment[label] for label in query)] += 1
        estimate = None if not accepted else {
            key: Fraction(count, accepted) for key, count in counts.items()}
        return {'proposed': proposals, 'accepted': accepted,
                'counts': counts, 'estimate': estimate}


def binary(label, parents, success_probabilities):
    """Create binary rows in lexicographic order of the parent axes."""
    successes = sequence(success_probabilities, 'success probabilities')
    return Node(label, 2, parents,
                tuple((ONE - probability(q), probability(q))
                      for q in successes))


def alarm(successes=(0, 1, 1, 1), root_probability=Fraction(1, 20)):
    """Stipulated B,E,A model; successes order is (00,01,10,11)."""
    return Network((
        binary('B', (), (root_probability,)),
        binary('E', (), (root_probability,)),
        binary('A', ('B', 'E'), successes),
    ))


def symptoms():
    """Stipulated toy C,D,H,I model, without clinical interpretation."""
    return Network((
        binary('C', (), (Fraction(1, 10),)),
        binary('D', (), (Fraction(1, 5),)),
        binary('H', ('C', 'D'),
               (Fraction(1, 10),) + (Fraction(9, 10),) * 3),
        binary('I', ('D',), (Fraction(1, 10), Fraction(9, 10))),
    ))


def tracking():
    """Exact five-step fair-bit H3/Y5 table; marginalize other noise bits."""
    masses = {(h, y): ZERO for h in range(4) for y in range(7)}
    for bits in product(range(2), repeat=6):
        h3 = sum(bits[:3])
        y5 = sum(bits)
        masses[h3, y5] += Fraction(1, 64)
    evidence = sum(q for (h, y), q in masses.items() if y == 2)
    return ({h: masses[h, 2] / evidence for h in range(4)}, evidence)


def demonstration():
    model = alarm()
    posterior, evidence = model.infer(('B',), {'A': 1})
    second, _ = model.infer(('B',), {'A': 1, 'E': 1})
    synergy = alarm((Fraction(1, 10), Fraction(1, 5),
                     Fraction(1, 5), Fraction(99, 100)), Fraction(1, 2))
    first_synergy, _ = synergy.infer(('B',), {'A': 1})
    second_synergy, _ = synergy.infer(('B',), {'A': 1, 'E': 1})
    cold, _ = symptoms().infer(('C',), {'H': 1})
    both, _ = symptoms().infer(('C',), {'H': 1, 'I': 1})
    track, track_evidence = tracking()
    sampled = model.reject(('B',), {'A': 1}, 1000, 3)
    unavailable = model.reject(('B',), {'A': 0, 'E': 1}, 20, 3)
    assert sum(model.joint().values(), ZERO) == ONE
    assert posterior[(1,)] == Fraction(20, 39)
    assert unavailable['estimate'] is None
    return {
        'alarm_evidence': str(evidence),
        'burglary_given_alarm': str(posterior[(1,)]),
        'burglary_given_alarm_earthquake': str(second[(1,)]),
        'synergy_posteriors': [str(first_synergy[(1,)]),
                               str(second_synergy[(1,)])],
        'cold_posteriors': [str(cold[(1,)]), str(both[(1,)])],
        'tracking_evidence': str(track_evidence),
        'tracking_posterior': [str(track[h]) for h in range(4)],
        'seeded_alarm': {
            'proposed': sampled['proposed'], 'accepted': sampled['accepted'],
            'burglary_estimate': str(sampled['estimate'][(1,)])},
        'zero_accept': {
            'proposed': unavailable['proposed'],
            'accepted': unavailable['accepted'],
            'estimate': unavailable['estimate']},
    }


if __name__ == '__main__':
    print(json.dumps(demonstration(), indent=2))
