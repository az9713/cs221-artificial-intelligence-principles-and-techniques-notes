"""Original finite-search lab. Standard library only; no model, network or eval.

Exact integer/Fraction costs; graphs are copied and frozen. Algorithms stop on
first arrival at a goal. Failure returns None; malformed inputs, cycles passed
to DAG solvers, or declared resource caps raise ValueError. Seeded sampling is
a reproducible finite trace, not evidence of statistical independence.
"""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import comb, exp, log1p
from random import Random
from types import MappingProxyType
import json


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def name(value, label):
    if type(value) is not str or not value or len(value) > 100:
        raise ValueError(f'{label} must be a nonempty string of at most100 characters')
    return value


@dataclass(frozen=True)
class Edge:
    action: str
    target: str
    cost: Fraction


class Problem:
    """Validated snapshot: dict[str, sequence[Edge]], start, iterable of goals."""
    __slots__ = ('graph', 'start', 'goals')

    def __init__(self, graph, start, goals):
        if type(graph) is not dict or not 1 <= len(graph) <= 10000:
            raise ValueError('graph must be a dict with1..10000 states')
        nodes = {name(s, 'state') for s in graph}
        start = name(start, 'start')
        if start not in nodes or type(goals) not in (tuple, list, set, frozenset):
            raise ValueError('start must be a state; goals must be a finite collection')
        frozen_goals = frozenset(name(g, 'goal') for g in goals)
        if not frozen_goals <= nodes:
            raise ValueError('goals must be graph states')
        copied = {}
        count = 0
        for state, edges in graph.items():
            if type(edges) not in (tuple, list):
                raise ValueError('adjacency must be tuple or list')
            actions = set()
            row = []
            for edge in edges:
                if type(edge) is not Edge:
                    raise ValueError('adjacency entries must be Edge objects')
                action, target = name(edge.action, 'action'), name(edge.target, 'target')
                if action in actions or target not in nodes:
                    raise ValueError('actions must be unique per state; targets must exist')
                if type(edge.cost) not in (int, Fraction):
                    raise ValueError('cost must be an integer or Fraction, excluding bool')
                cost = Fraction(edge.cost)
                if abs(cost) > 1000000 or cost.denominator > 1000000:
                    raise ValueError('cost magnitude and denominator must be at most1000000')
                row.append(Edge(action, target, cost))
                actions.add(action)
                count += 1
                if count > 100000:
                    raise ValueError('at most100000 edges supported')
            copied[state] = tuple(row)
        object.__setattr__(self, 'graph', MappingProxyType(copied))
        object.__setattr__(self, 'start', start)
        object.__setattr__(self, 'goals', frozen_goals)

    def __setattr__(self, key, value):
        raise AttributeError('Problem is immutable')


@dataclass(frozen=True)
class Solution:
    states: tuple
    actions: tuple
    cost: Fraction


def problem(value):
    if type(value) is not Problem:
        raise ValueError('a validated Problem is required')
    return value


def outgoing(p, state):
    return () if state in p.goals else p.graph[state]


def travel(n, tickets=None, forbid_consecutive=False):
    integer(n, 'n', 1, 200)
    if tickets is not None:
        integer(tickets, 'tickets', 0, 20)
    if type(forbid_consecutive) is not bool:
        raise ValueError('forbid_consecutive must be bool')
    def key(state):
        i, k, last = state
        return f'{i}|{k if k is not None else "u"}|{int(last)}'
    start = (1, tickets, False)
    pending, seen, graph, goals = deque([start]), {start}, {}, []
    while pending:
        state = pending.popleft()
        i, k, last = state
        successors = []
        if i == n:
            goals.append(key(state))
        else:
            successors.append(('walk', (i + 1, k, False), 1))
            if 2 * i <= n and (k is None or k > 0) and not last:
                successors.append(('tram', (2 * i, None if k is None else k - 1,
                                            forbid_consecutive), 2))
        graph[key(state)] = [Edge(a, key(t), c) for a, t, c in successors]
        for _, target, _ in successors:
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return Problem(graph, key(start), goals)


def dag_order(p):
    """Topological order of start-reachable states, goals treated as absorbing."""
    problem(p)
    reachable, discovered, stack = {p.start}, [], [p.start]
    while stack:
        state = stack.pop()
        discovered.append(state)
        for edge in outgoing(p, state):
            if edge.target not in reachable:
                reachable.add(edge.target)
                stack.append(edge.target)
    indegree = {s: 0 for s in discovered}
    for state in indegree:
        for edge in outgoing(p, state):
            indegree[edge.target] += 1
    ready = deque(s for s, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        state = ready.popleft()
        order.append(state)
        for edge in outgoing(p, state):
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                ready.append(edge.target)
    if len(order) != len(reachable):
        raise ValueError('DAG solver requires acyclic start-reachable nonterminal graph')
    return order


def exact_dag(p):
    order = dag_order(p)
    values, choices = {}, {}
    for state in reversed(order):
        best, choice = (Fraction(0), None) if state in p.goals else (None, None)
        for edge in outgoing(p, state):
            suffix = values[edge.target]
            if suffix is not None:
                candidate = edge.cost + suffix
                if best is None or candidate < best:
                    best, choice = candidate, edge
        values[state], choices[state] = best, choice
    if values[p.start] is None:
        return None, values
    states, actions, state = [p.start], [], p.start
    while state not in p.goals:
        edge = choices[state]
        actions.append(edge.action)
        state = edge.target
        states.append(state)
    return Solution(tuple(states), tuple(actions), values[p.start]), values


def exhaustive(p, max_visits=100000):
    dag_order(p)
    integer(max_visits, 'max_visits', 1, 1000000)
    stack = [Solution((p.start,), (), Fraction(0))]
    best, visits = None, 0
    while stack:
        path = stack.pop()
        visits += 1
        if visits > max_visits:
            raise ValueError('exhaustive visit budget exceeded')
        state = path.states[-1]
        if state in p.goals:
            if best is None or path.cost < best.cost:
                best = path
        else:
            for edge in reversed(outgoing(p, state)):
                stack.append(Solution(path.states + (edge.target,),
                                      path.actions + (edge.action,), path.cost + edge.cost))
    return best, visits


def beam(p, width, horizon):
    problem(p)
    integer(width, 'width', 1, 10000)
    integer(horizon, 'horizon', 0, 1000)
    candidates = [Solution((p.start,), (), Fraction(0))]
    for _ in range(horizon):
        pool = []
        for path in candidates:
            state = path.states[-1]
            if state in p.goals:
                pool.append(path)
            else:
                pool.extend(Solution(path.states + (e.target,), path.actions + (e.action,),
                                     path.cost + e.cost) for e in outgoing(p, state))
            if len(pool) > 200000:
                raise ValueError('beam candidate budget exceeded')
        candidates = sorted(pool, key=lambda path: path.cost)[:width]
        if not candidates or all(path.states[-1] in p.goals for path in candidates):
            break
    complete = [path for path in candidates if path.states[-1] in p.goals]
    return min(complete, key=lambda path: path.cost, default=None)


def best_sample(p, trials, horizon, seed=0):
    problem(p)
    integer(trials, 'trials', 1, 10000)
    integer(horizon, 'horizon', 0, 1000)
    integer(seed, 'seed', -(2 ** 63), 2 ** 63 - 1)
    if trials * max(horizon, 1) > 100000:
        raise ValueError('sampling action budget exceeded')
    rng, best, completed = Random(seed), None, 0
    for _ in range(trials):
        states, actions, cost, state = [p.start], [], Fraction(0), p.start
        for _ in range(horizon):
            if state in p.goals:
                break
            edges = outgoing(p, state)
            if not edges:
                break
            edge = rng.choice(edges)
            actions.append(edge.action)
            cost += edge.cost
            state = edge.target
            states.append(state)
        if state in p.goals:
            completed += 1
            candidate = Solution(tuple(states), tuple(actions), cost)
            if best is None or cost < best.cost:
                best = candidate
    return best, completed


def pass_at_k(total, correct, k):
    integer(total, 'total', 1, 100000)
    integer(correct, 'correct', 0, total)
    integer(k, 'k', 1, total)
    return 1 - Fraction(comb(total - correct, k), comb(total, k))


def safe_answer(text):
    """Hard verifier for the stipulated answer to3+7*3, not general reasoning."""
    if type(text) is not str or len(text) > 1000:
        raise ValueError('candidate must be a string of at most1000 characters')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (ValueError, RecursionError):
        return False
    return type(value) is dict and set(value) == {'answer'} and type(value['answer']) is int and value['answer'] == 24


def fixture(nonmonotonic=False):
    if not nonmonotonic:
        rows = {'s': [('a', 1), ('b', 2)], 'a': [('g', 100)], 'b': [('g', 0)], 'g': []}
    else:
        rows = {'s': [('a', 1), ('b', 2)], 'a': [('a1', 4), ('a2', Fraction(9, 2))],
                'b': [('b1', 0), ('b2', 0)], 'a1': [('g', 0)], 'a2': [('g', 0)],
                'b1': [('g', 100)], 'b2': [('g', 100)], 'g': []}
    return Problem({s: [Edge(f'to_{t}', t, c) for t, c in edges] for s, edges in rows.items()}, 's', ['g'])


def report():
    travel_rows = {}
    for n in (4, 10, 17):
        p = travel(n)
        exact, values = exact_dag(p)
        full, visits = exhaustive(p)
        assert exact.cost == full.cost
        travel_rows[str(n)] = {'cost': int(exact.cost), 'visits': visits, 'states': len(values),
                               'route': [s.split('|')[0] for s in exact.states]}
    sample, completed = best_sample(travel(17), 100, 20, 7)
    constrained, _ = exact_dag(travel(17, 2, True))
    widths = {'greedy_failure': [int(beam(fixture(), b, 2).cost) for b in (1, 2)],
              'nonmonotonic': [int(beam(fixture(True), b, 3).cost) for b in (1, 2)]}
    return {'travel': travel_rows, 'two_tickets_no_consecutive_cost': int(constrained.cost),
            'sampling_seed7': {'best_cost': int(sample.cost), 'complete_trials': completed},
            'beam_width1_width2': widths, 'rare_path_miss_1000': round(exp(1000 * log1p(-2 ** -20)), 12),
            'pool_pass_at_k_N5_C2': [str(pass_at_k(5, 2, k)) for k in range(1, 6)],
            'safe_answer': [safe_answer(t) for t in ('{"answer":24}', '{"answer":23}', '{"answer":true}',
                                                       '{"answer":24,"answer":23}', '__import__("os")')],
            'finite_bonus_scores': {'valid': 200 - 100, 'invalid': 1}}


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
