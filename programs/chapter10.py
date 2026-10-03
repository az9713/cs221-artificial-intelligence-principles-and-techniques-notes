'Original finite game calculations. Standard library; ' \
    'no file/network effects.\n\nExact model numbers are ' \
    'ints/Fractions (not bools or floats), magnitude ' \
    '<=1e100\nand denominator <=1e6. Finite acyclic labelled ' \
    'graph: <=4096 states, <=16\nactions/state and <=64 ' \
    'plies. State labels identify full information. Each\n' \
    'policy/known-chance row has the exact action keys and ' \
    'sums exactly to one.\n'

from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from random import Random

F = Fraction


def rational(x, name="number"):
    if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
        raise ValueError(name + " must be an int or Fraction, not bool/float")
    x = F(x)
    if abs(x) > 10**100 or x.denominator > 10**6:
        raise ValueError(name + " exceeds magnitude/denominator bound")
    return x


def count(x, name, lo, hi):
    if type(x) is not int or not lo <= x <= hi:
        raise ValueError(name + " is outside the integer bounds")
    return x


@dataclass(frozen=True)
class Node:
    label: str
    kind: str
    edges: tuple = ()  # (action, successor_label)
    utility: object = None
    probabilities: tuple = ()  # (action, exact probability), chance only


def distribution(edges, supplied):
    actions = tuple(a for a, _ in edges)
    if not isinstance(supplied, dict) or set(supplied) != set(actions):
        raise ValueError("distribution must have exactly the admissible keys")
    row = tuple((a, rational(supplied[a], "probability")) for a in actions)
    if any(p < 0 for _, p in row) or sum(p for _, p in row) != 1:
        raise ValueError("probability row must be nonnegative and sum to one")
    return row


@dataclass(frozen=True, init=False)
class Game:
    "Own immutable copies; reject malformed, cyclic or unreachable states."

    nodes: tuple
    root: str
    height: int

    def __init__(self, nodes, root):
        if not isinstance(nodes, (tuple, list)) or not 1 <= len(nodes) <= 4096:
            raise ValueError(
                "nodes must be a finite sequence of 1..4096 Nodes"
            )
        copied = {}
        for node in nodes:
            if (
                not isinstance(node, Node)
                or not isinstance(node.label, str)
                or not node.label
            ):
                raise ValueError("each Node needs a nonempty string label")
            if node.label in copied or node.kind not in (
                "leaf",
                "max",
                "min",
                "chance",
            ):
                raise ValueError("duplicate label or unsupported node kind")
            if not isinstance(node.edges, (tuple, list)) or not isinstance(
                node.probabilities, (tuple, list)
            ):
                raise ValueError(
                    "edges/probabilities must be finite sequences"
                )
            edges = []
            for edge in node.edges:
                if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                    raise ValueError("edge must be an action/successor pair")
                a, child = edge
                if (
                    not isinstance(a, str)
                    or not a
                    or not isinstance(child, str)
                    or not child
                ):
                    raise ValueError("edge labels must be nonempty strings")
                edges.append((a, child))
            edges = tuple(edges)
            if len({a for a, _ in edges}) != len(edges):
                raise ValueError("duplicate action")
            if node.kind == "leaf":
                if edges or node.probabilities or node.utility is None:
                    raise ValueError(
                        "leaf has utility and no edges/probabilities"
                    )
                copied[node.label] = Node(
                    node.label, "leaf", (), rational(node.utility), ()
                )
            else:
                if not 1 <= len(edges) <= 16 or node.utility is not None:
                    raise ValueError(
                        "nonterminal needs 1..16 edges and no utility"
                    )
                probs = ()
                if node.kind == "chance":
                    try:
                        pairs = tuple(
                            tuple(pair) for pair in node.probabilities
                        )
                        if any(len(pair) != 2 for pair in pairs) or len(
                            {pair[0] for pair in pairs}
                        ) != len(pairs):
                            raise ValueError(
                                ("invalid/duplicate chance action")
                            )
                        probs = distribution(edges, dict(pairs))
                    except (TypeError, KeyError) as exc:
                        raise ValueError(
                            "malformed chance distribution"
                        ) from exc
                elif node.probabilities:
                    raise ValueError("only chance nodes carry probabilities")
                copied[node.label] = Node(
                    node.label, node.kind, edges, None, probs
                )
        if not isinstance(root, str) or root not in copied:
            raise ValueError("root must identify a declared state")
        if any(
            child not in copied
            for node in copied.values()
            for _, child in node.edges
        ):
            raise ValueError("unknown successor")
        active, heights = set(), {}

        def height(s, depth=0):
            if depth > 64:
                raise ValueError("graph exceeds 64-ply bound")
            if s in active:
                raise ValueError(
                    "cyclic game; finite backward induction required"
                )
            if s in heights:
                return heights[s]
            active.add(s)
            h = (
                0
                if not copied[s].edges
                else 1 + max(height(c, depth + 1) for _, c in copied[s].edges)
            )
            active.remove(s)
            if h > 64:
                raise ValueError("graph exceeds 64-ply bound")
            heights[s] = h
            return h

        height(root)
        if set(heights) != set(copied):
            raise ValueError("unreachable states are not part of this Game")
        object.__setattr__(self, "nodes", tuple(copied.values()))
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "height", heights[root])

    def table(self):
        return {node.label: node for node in self.nodes}


def policy_rows(game, policy, kinds):
    if not isinstance(game, Game) or not isinstance(policy, dict):
        raise ValueError("Game and policy dictionary required")
    expected = {n.label for n in game.nodes if n.kind in kinds}
    if set(policy) != expected:
        raise ValueError(
            "policy must specify exactly the requested player states"
        )
    return {
        n.label: distribution(n.edges, policy[n.label])
        for n in game.nodes
        if n.label in expected
    }


@dataclass(frozen=True)
class Result:
    value: Fraction
    action: object
    leaves: int
    bound: str = "exact"
    trace: tuple = ()


def solve(game, mode="minimax", policy=None):
    "Memoized exact evaluation; leaves counts unique evaluated leaf states."
    if not isinstance(game, Game) or mode not in (
        "evaluate",
        "expectimax",
        "minimax",
    ):
        raise ValueError("invalid Game or solve mode")
    if mode == "minimax":
        if policy is not None:
            raise ValueError("minimax does not consume a fixed player policy")
        rows = {}
    else:
        rows = policy_rows(
            game, policy, ("max", "min") if mode == "evaluate" else ("min",)
        )
    table, memo, choices, leaves = game.table(), {}, {}, set()

    def value(s):
        if s in memo:
            return memo[s]
        n = table[s]
        if n.kind == "leaf":
            leaves.add(s)
            v = n.utility
        else:
            child = {a: value(c) for a, c in n.edges}
            if n.kind == "chance" or s in rows:
                probs = n.probabilities if n.kind == "chance" else rows[s]
                v = sum((p * child[a] for a, p in probs), F(0))
            else:
                choose = max if n.kind == "max" else min
                a = choose(child, key=child.get)
                choices[s] = a
                v = child[a]
        memo[s] = v
        return v

    v = value(game.root)
    return Result(v, choices.get(game.root), len(leaves))


def sample_value(game, policy, rng):
    "Exact rational action sampling with integer draws, denominator cap1e12."
    rows = policy_rows(game, policy, ("max", "min"))
    if not isinstance(rng, Random):
        raise ValueError("local random.Random required")
    table, s = game.table(), game.root
    while table[s].kind != "leaf":
        node = table[s]
        row = node.probabilities if node.kind == "chance" else rows[s]
        denominator = lcm(*(p.denominator for _, p in row))
        if denominator > 10**12:
            raise ValueError("exact integer sampling denominator exceeds1e12")
        draw = rng.randrange(denominator)
        cumulative = 0
        for a, p in row:
            cumulative += p.numerator * (denominator // p.denominator)
            if draw < cumulative:
                s = dict(node.edges)[a]
                break
    return table[s].utility


def alpha_beta(game, alpha=None, beta=None, order=None, max_calls=1000000):
    (
        'Fail-soft deterministic search; return '
        'exact/upper/lower plus trace.\n\n    None is an infinite '
        'window endpoint. A narrow-window return is not an\n    '
        'exact subtree value unless its status says exact. '
        'Leaves count visits,\n    including repeated '
        'occurrences of shared graph nodes; no transposition '
        'cache.\n    '
    )
    if not isinstance(game, Game) or any(
        n.kind == "chance" for n in game.nodes
    ):
        raise ValueError("alpha-beta requires a deterministic Game")
    max_calls = count(max_calls, "search-call budget", 1, 1000000)
    alpha = None if alpha is None else rational(alpha, "alpha")
    beta = None if beta is None else rational(beta, "beta")
    if alpha is not None and beta is not None and alpha >= beta:
        raise ValueError("alpha must be strictly below beta")
    table = game.table()
    if order is None:
        order = {}
    if not isinstance(order, dict) or any(
        s not in table or table[s].kind == "leaf" for s in order
    ):
        raise ValueError("order keys must identify nonterminal states")
    for s, actions in order.items():
        if (
            not isinstance(actions, (tuple, list))
            or any(not isinstance(a, str) for a in actions)
            or len(actions) != len(table[s].edges)
            or set(actions) != {a for a, _ in table[s].edges}
        ):
            raise ValueError(
                "order must be a permutation of admissible actions"
            )
    visits, trace, calls = [], [], [0]

    def search(s, a0, b0):
        calls[0] += 1
        if calls[0] > max_calls:
            raise RuntimeError("alpha-beta search-call budget exhausted")
        n = table[s]
        if n.kind == "leaf":
            visits.append(s)
            return n.utility, None, "exact"
        edges = dict(n.edges)
        actions = order.get(s, tuple(edges))
        best, chosen, a, b = None, None, a0, b0
        for index, action in enumerate(actions):
            v, _, _ = search(edges[action], a, b)
            if best is None or (v > best if n.kind == "max" else v < best):
                best, chosen = v, action
            if n.kind == "max":
                a = best if a is None else max(a, best)
            else:
                b = best if b is None else min(b, best)
            if a is not None and b is not None and a >= b:
                trace.append((s, "cutoff", tuple(actions[index + 1:]), best))
                break
        status = (
            ("upper")
            if a0 is not None and best <= a0
            else ("lower")
            if b0 is not None and best >= b0
            else ("exact")
        )
        trace.append((s, status, chosen, best))
        return best, chosen, status

    v, action, status = search(game.root, alpha, beta)
    return Result(v, action, len(visits), status, tuple(trace))


def depth_search(game, depth, evaluation):
    (
        'Terminal first, then heuristic cutoff; depth counts '
        'every traversed edge.'
    )
    if not isinstance(game, Game):
        raise ValueError("Game required")
    depth = count(depth, "depth", 0, 64)
    expected = {n.label for n in game.nodes if n.kind != "leaf"}
    if not isinstance(evaluation, dict) or set(evaluation) != expected:
        raise ValueError(
            "evaluation must specify every nonterminal state exactly"
        )
    evaluation = {s: rational(x, "evaluation") for s, x in evaluation.items()}
    table, memo, choices, leaves = game.table(), {}, {}, set()

    def value(s, d):
        if (s, d) in memo:
            return memo[s, d]
        n = table[s]
        if n.kind == "leaf":
            leaves.add(s)
            v = n.utility
        elif d == 0:
            v = evaluation[s]
        else:
            child = {a: value(c, d - 1) for a, c in n.edges}
            if n.kind == "chance":
                v = sum((p * child[a] for a, p in n.probabilities), F(0))
            else:
                a = (max if n.kind == "max" else min)(child, key=child.get)
                choices[s, d] = a
                v = child[a]
        memo[s, d] = v
        return v

    v = value(game.root, depth)
    return Result(
        v,
        choices.get((game.root, depth)),
        len(leaves),
        "heuristic" if depth < game.height else "exact",
    )


def bins(chance=False):
    nodes = []
    for label, numbers in [("A", (-50, 50)), ("B", (1, 3)), ("C", (-5, 15))]:
        nodes.append(
            Node(label, (
                'min'
            ), tuple((str(i), label + str(i)) for i in (1, 2)))
        )
        nodes.extend(
            Node(label + str(i), "leaf", utility=x)
            for i, x in enumerate(numbers, 1)
        )
    targets = ("A", "B", "C")
    if chance:
        for i, label in enumerate(targets):
            nodes.append(
                Node(
                    "coin" + label,
                    ("chance"),
                    ((("stay"), label), (("left"), targets[(i - 1) % 3])),
                    probabilities=((("stay"), F(1, 2)), (("left"), F(1, 2))),
                )
            )
        targets = tuple("coin" + s for s in targets)
    nodes.append(Node("root", "max", tuple(zip(("A", "B", "C"), targets))))
    return Game(nodes, "root")


def halving(n=11):
    n = count(n, "initial number", 0, 40)
    states = {}

    def add(k, player):
        s = f"{k}:{player}"
        if s in states:
            return s
        if k == 0:
            states[s] = Node(s, "leaf", utility=1 if player == "max" else -1)
        else:
            other = "min" if player == "max" else "max"
            states[s] = Node(
                s,
                player,
                (
                    (("decrement"), add(k - 1, other)),
                    (("half"), add(k // 2, other)),
                ),
            )
        return s

    root = add(n, "max")
    return Game(list(states.values()), root)


def pruning_fixture():
    leaves = [
        Node(str(x) + suffix, ("leaf"), utility=x)
        for x, suffix in [
            (9, ("L")),
            (7, ("L")),
            (6, ("M")),
            (3, ("M")),
            (4, ("M")),
            (7, ("M")),
            (9, ("M")),
            (8, ("R")),
            (3, ("R")),
        ]
    ]
    return Game(
        leaves
        + [
            Node(("left"), ("min"), ((("a"), ("9L")), (("b"), ("7L")))),
            Node(("m1"), ("max"), ((("a"), ("3M")), (("b"), ("4M")))),
            Node(("m2"), ("max"), ((("a"), ("7M")), (("b"), ("9M")))),
            Node(
                ("middle"),
                ("min"),
                ((("a"), ("6M")), (("b"), ("m1")), (("c"), ("m2"))),
            ),
            Node(("right"), ("min"), ((("a"), ("8R")), (("b"), ("3R")))),
            Node(
                ("root"),
                ("max"),
                ((("L"), ("left")), (("M"), ("middle")), (("R"), ("right"))),
            ),
        ],
        ("root"),
    )


def main():
    game = bins()
    fair = {s: {"1": F(1, 2), "2": F(1, 2)} for s in ("A", "B", "C")}
    robust = solve(game)
    response = solve(game, "expectimax", fair)
    print(
        f"bins: minimax={robust.value}, action={robust.action}; expectimax={
            response.value
        }, action={response.action}"
    )
    adversary = {s: {"1": 1, "2": 0} for s in ("A", "B", "C")}
    for agent in ("C", "B"):
        row = {"root": {a: int(a == agent) for a in ("A", "B", "C")}}
        print(
            f"faceoff {agent}: adversarial={
                solve(game, 'evaluate', dict(row, **adversary)).value
            }, fair={solve(game, 'evaluate', dict(row, **fair)).value}"
        )
    row = dict(fair, root={"A": 1, "B": 0, "C": 0})
    rng = Random(7)
    mean = (
        sum((sample_value(game, row, rng) for _ in range(1000)), F(0)) / 1000
    )
    print(
        f"fixed A/fair: exact={
            solve(game, 'evaluate', row).value
        }, seeded1000 mean={mean}"
    )
    print(
        "halving agent-turn n1..11:",
        tuple(int(solve(halving(n)).value) for n in range(1, 12)),
    )
    coin = solve(bins(True))
    print(
        f"chance before informed opponent: value={coin.value}, action={
            coin.action
        }"
    )
    tree = pruning_fixture()
    pruned = alpha_beta(tree)
    print(
        f"alpha-beta full window: value={pruned.value}, action={
            pruned.action
        }, visited leaves={pruned.leaves}/9, status={pruned.bound}"
    )
    middle_nodes = [
        n
        for n in tree.nodes
        if n.label in ("middle", "m1", "m2", "6M", "3M", "4M", "7M", "9M")
    ]
    middle = Game(middle_nodes, "middle")
    cut = alpha_beta(middle, alpha=7)
    print(
        f"cut middle subtree: returned={cut.value}, status={
            cut.bound
        }, exhaustive={solve(middle).value}"
    )
    shallow = Game(
        [
            Node("r", "max", (("A", "a"), ("B", "b"))),
            Node("a", "min", (("go", "aa"),)),
            Node("b", "min", (("go", "bb"),)),
            Node("aa", "max", (("finish", "u2"),)),
            Node("bb", "max", (("finish", "u1"),)),
            Node("u2", "leaf", utility=2),
            Node("u1", "leaf", utility=1),
        ],
        "r",
    )
    evaluation = {"r": 0, "a": 2, "b": 1, "aa": 0, "bb": 3}
    print(
        "depth cutoff:",
        tuple(
            (
                d,
                depth_search(shallow, d, evaluation).action,
                str(depth_search(shallow, d, evaluation).value),
            )
            for d in (1, 2, 3)
        ),
    )
    print(
        (
            "Finite fixtures do not verify an opponent model or a "
            "heuristic error bound."
        )
    )


if __name__ == "__main__":
    main()
