"""Exact finite-graph UCS/A* teaching fixtures; standard library only.

Public costs and potentials are built-in integers (booleans excluded). States
must have stable hash/equality; adjacency order defines FIFO priority ties.
Reopening mode deliberately audits admissibility with an exact reverse solve.
It is a correctness demonstration, not a competitive heuristic implementation.
No files, network, random state, or input objects are modified.
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count


@dataclass(frozen=True)
class Result:
    cost: int
    path: tuple
    removals: int  # valid removals, including the returned goal


def _integer(value, label, nonnegative=False):
    if type(value) is not int or (nonnegative and value < 0):
        raise ValueError(
            label
            + " must be a built-in "
            + ("nonnegative " if nonnegative else "")
            + "integer"
        )
    return value


def _graph(graph):
    if type(graph) is not dict or not graph:
        raise ValueError("graph must be a nonempty dict")
    copied = {}
    for state, edges in graph.items():
        if not isinstance(edges, (list, tuple)):
            raise ValueError("adjacency must be a list or tuple")
        outgoing = []
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError("each edge must be (target, cost)")
            target, cost = edge
            try:
                declared = target in graph
            except TypeError as exc:
                raise ValueError("targets must be hashable") from exc
            if not declared:
                raise ValueError("every target needs an adjacency entry")
            outgoing.append((target, _integer(cost, "cost", True)))
        copied[state] = tuple(outgoing)
    return copied


def _terminals(graph, goals):
    if not isinstance(goals, (list, tuple, set, frozenset)) or not goals:
        raise ValueError("goals must be a nonempty finite collection")
    try:
        terminals = frozenset(goals)
        if not terminals <= graph.keys():
            raise ValueError("goals must be declared states")
    except TypeError as exc:
        raise ValueError("goals must be hashable") from exc
    return terminals


def _run(graph, start, goals, potential, reopen):
    (
        'Internal engine. Call shortest_path for validated '
        'correctness guarantees.'
    )
    serial = count()
    queue = [(potential[start], next(serial), 0, start)]
    best, parent, closed = {start: 0}, {}, set()
    removals = 0
    while queue:
        _, _, distance, state = heappop(queue)
        if distance != best[state] or state in closed:
            continue
        closed.add(state)
        removals += 1
        if state in goals:
            route = [state]
            while state is not start and state != start:
                state = parent[state]
                route.append(state)
            return Result(distance, tuple(reversed(route)), removals)
        for target, cost in graph[state]:
            candidate = distance + cost
            if target in closed and not reopen:
                continue
            if target not in best or candidate < best[target]:
                best[target] = candidate
                parent[target] = state
                closed.discard(target)
                heappush(
                    queue,
                    (
                        candidate + potential[target],
                        next(serial),
                        candidate,
                        target,
                    ),
                )
    return None


def _future(graph, goals):
    reverse = {state: [] for state in graph}
    for state, edges in graph.items():
        for target, cost in edges:
            reverse[target].append((state, cost))
    serial = count()
    # Iterate graph order, not set order, so ties are deterministic.
    queue = [(0, next(serial), state) for state in graph if state in goals]
    best = {state: 0 for state in graph if state in goals}
    final = {}
    while queue:
        distance, _, state = heappop(queue)
        if state in final or distance != best[state]:
            continue
        final[state] = distance
        for target, cost in reverse[state]:
            candidate = distance + cost
            if target not in best or candidate < best[target]:
                best[target] = candidate
                heappush(queue, (candidate, next(serial), target))
    return final


def future_costs(graph, goals):
    """Exact distance to ANY goal; omit states unable to reach a goal."""
    copied = _graph(graph)
    return _future(copied, _terminals(copied, goals))


def shortest_path(graph, start, goals, heuristic=None, *, reopen=False):
    (
        'Return Result or None. Reject violated supported input '
        'hypotheses.\n\n    Default: check consistency, use '
        'permanent settlement. Reopening mode:\n    check '
        'admissibility using a full reverse solve, then allow '
        'repeat removals.\n    Reported removals exclude that '
        'audit and stale heap records. Integer costs\n    avoid '
        'floating-point comparisons; their arithmetic cost '
        'grows with bits.\n    '
    )
    copied = _graph(graph)
    try:
        present = start in copied
    except TypeError as exc:
        raise ValueError("start must be hashable") from exc
    if not present:
        raise ValueError("start must be a declared state")
    terminals = _terminals(copied, goals)
    if type(reopen) is not bool:
        raise ValueError("reopen must be boolean")
    if heuristic is None:
        potential = {state: 0 for state in copied}
    else:
        if type(heuristic) is not dict or heuristic.keys() != copied.keys():
            raise ValueError("heuristic must specify exactly all states")
        potential = {
            state: _integer(value, "heuristic")
            for state, value in heuristic.items()
        }
    if any(potential[goal] != 0 for goal in terminals):
        raise ValueError("every goal heuristic must be zero")
    if reopen:
        exact = _future(copied, terminals)
        if any(
            potential[state] > distance for state, distance in exact.items()
        ):
            raise ValueError("reopening audit requires admissibility")
    elif any(
        potential[state] > cost + potential[target]
        for state, edges in copied.items()
        for target, cost in edges
    ):
        raise ValueError("permanent settlement requires consistency")
    return _run(copied, start, terminals, potential, reopen)


def grid_problem(rows):
    """Rectangular ASCII grid: # wall, . free, exactly one S and one G."""
    if (
        not isinstance(rows, (list, tuple))
        or not rows
        or any(type(row) is not str for row in rows)
        or not rows[0]
        or any(len(row) != len(rows[0]) for row in rows)
        or any(ch not in ".#SG" for row in rows for ch in row)
    ):
        raise ValueError("invalid rectangular grid")
    starts = [
        (r, c)
        for r, row in enumerate(rows)
        for c, ch in enumerate(row)
        if ch == "S"
    ]
    goals = [
        (r, c)
        for r, row in enumerate(rows)
        for c, ch in enumerate(row)
        if ch == "G"
    ]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("grid needs exactly one S and one G")
    graph = {
        (r, c): []
        for r, row in enumerate(rows)
        for c, ch in enumerate(row)
        if ch != "#"
    }
    for r, c in graph:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            target = r + dr, c + dc
            if target in graph:
                graph[r, c].append((target, 1))
    gr, gc = goals[0]
    h = {(r, c): abs(r - gr) + abs(c - gc) for r, c in graph}
    return graph, starts[0], goals, h


def demo():
    diamond = {
        "A": [("B", 1), ("C", 100)],
        "B": [("A", 1), ("C", 1), ("D", 100)],
        "C": [("A", 100), ("B", 1), ("D", 1)],
        "D": [("B", 100), ("C", 1)],
    }
    answer = shortest_path(diamond, "A", ["D"])
    print("diamond:", answer.cost, "->".join(answer.path))
    graph, start, goals, h = grid_problem(
        ("S....", "###.#", ".....", ".####", "....G")
    )
    ucs = shortest_path(graph, start, goals)
    astar = shortest_path(graph, start, goals, h)
    exact = future_costs(graph, goals)
    print(
        f"grid: cost={astar.cost}, Manhattan={h[start]}, "
        f"UCS removals={ucs.removals}, A* removals={astar.removals}"
    )
    print(
        f"grid ranking: (2,4) h={h[2, 4]} true={exact[2, 4]}; "
        f"(3,0) h={h[3, 0]} true={exact[3, 0]}"
    )
    inconsistent = {
        "s": [("a", 3), ("b", 1)],
        "a": [("g", 3)],
        "b": [("a", 1), ("g", 100)],
        "g": [],
    }
    ih = {"s": 0, "a": 0, "b": 4, "g": 0}
    unsafe = _run(inconsistent, "s", {"g"}, ih, False)
    repaired = shortest_path(inconsistent, "s", ["g"], ih, reopen=True)
    print(
        f"inconsistent audit: unchecked closed={unsafe.cost}, "
        f"validated reopening={repaired.cost}"
    )
    zero = {"s": [("a", 0)], "a": [("s", 0), ("g", 1)], "g": []}
    print("finite zero cycle:", shortest_path(zero, "s", ["g"]).cost)
    print("unreachable:", shortest_path({"s": [], "g": []}, "s", ["g"]))


if __name__ == "__main__":
    demo()
