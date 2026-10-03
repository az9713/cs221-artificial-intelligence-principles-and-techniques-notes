'Original finite MDP examples for Chapter 7; Python ' \
    'standard library only.\n\nExact rational probabilities; ' \
    'binary64 rewards/iterates; bounded finite inputs.\n' \
    'Invalid inputs raise ValueError. Exhausted ' \
    'iteration/rollout budgets raise\nRuntimeError, never a ' \
    'silently completed episode. No file/network effects.\n'

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from math import fsum, isfinite, sqrt
from random import Random


def finite_real(value, name, bound=1e100):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be an int or float, excluding bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " must be representable in binary64") from exc
    if not isfinite(result) or abs(result) > bound:
        raise ValueError(name + " is outside the finite documented range")
    return result


def count(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def probability(value):
    if type(value) is int:
        value = Fraction(value)
    if not isinstance(value, Fraction) or not 0 < value <= 1:
        raise ValueError(
            ("outcome probabilities must be positive int/Fraction")
        )
    if value.denominator > 10**12:
        raise ValueError("probability denominator exceeds 10**12")
    return value


@dataclass(frozen=True)
class Outcome:
    probability: Fraction
    reward: float
    successor: str


@dataclass(frozen=True)
class Model:
    # rows: (state, ((action, (Outcome, ...)), ...)); terminal rows are empty.
    rows: tuple
    discount: float

    def __post_init__(self):
        gamma = finite_real(self.discount, "discount", 1)
        if gamma < 0:
            raise ValueError("discount must be in [0, 1]")
        if not isinstance(self.rows, (tuple, list)):
            raise ValueError("rows must be a tuple/list")
        count(len(self.rows), "state count", 1, 64)
        canonical, names = [], set()
        for row in self.rows:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("each row must contain state and actions")
            state, actions = row
            if not isinstance(state, str) or not state or state in names:
                raise ValueError(
                    "state names must be distinct nonempty strings"
                )
            names.add(state)
            if not isinstance(actions, (tuple, list)) or len(actions) > 16:
                raise ValueError(
                    "actions must be a tuple/list of at most 16 entries"
                )
            fixed_actions, action_names = [], set()
            for entry in actions:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise ValueError(
                        "action entries must contain name and outcomes"
                    )
                action, outcomes = entry
                if (
                    not isinstance(action, str)
                    or not action
                    or action in action_names
                ):
                    raise ValueError(
                        "action names must be distinct nonempty strings"
                    )
                action_names.add(action)
                if not isinstance(outcomes, (tuple, list)):
                    raise ValueError("outcomes must be a tuple/list")
                count(len(outcomes), "outcome count", 1, 64)
                fixed = []
                for outcome in outcomes:
                    if not isinstance(outcome, Outcome):
                        raise ValueError("each outcome must be an Outcome")
                    prob = probability(outcome.probability)
                    reward = finite_real(outcome.reward, "reward", 1e6)
                    if not isinstance(outcome.successor, str):
                        raise ValueError("successor must be a state name")
                    fixed.append(Outcome(prob, reward, outcome.successor))
                if sum((o.probability for o in fixed), Fraction()) != 1:
                    raise ValueError(
                        "each action's exact probability mass must be one"
                    )
                fixed_actions.append((action, tuple(fixed)))
            canonical.append((state, tuple(fixed_actions)))
        for _, actions in canonical:
            for _, outcomes in actions:
                if any(o.successor not in names for o in outcomes):
                    raise ValueError(
                        "every successor must be a declared state"
                    )
        object.__setattr__(self, "rows", tuple(canonical))
        object.__setattr__(self, "discount", gamma)

    def table(self):
        return {state: dict(actions) for state, actions in self.rows}


def values_for(model, values):
    table = model.table()
    if not isinstance(values, dict) or set(values) != set(table):
        raise ValueError("values must have exactly the model's state keys")
    fixed = {s: finite_real(values[s], "value") for s in table}
    if any(fixed[s] != 0 for s in table if not table[s]):
        raise ValueError("terminal continuation values must be zero")
    return fixed


def policy_for(model, policy):
    table = model.table()
    required = {s for s in table if table[s]}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("policy must contain exactly the nonterminal states")
    if any(
        not isinstance(policy[s], str) or policy[s] not in table[s]
        for s in required
    ):
        raise ValueError("policy selects an inadmissible action")
    return dict(policy)


def action_backup(model, state, action, values):
    fixed = values_for(model, values)
    table = model.table()
    if (
        not isinstance(state, str)
        or not isinstance(action, str)
        or state not in table
        or action not in table[state]
    ):
        raise ValueError("unknown state or inadmissible action")
    terms = [
        float(o.probability) * (o.reward + model.discount * fixed[o.successor])
        for o in table[state][action]
    ]
    return finite_real(fsum(terms), "backup")


def sweep(model, values, policy=None):
    fixed = values_for(model, values)
    selected = None if policy is None else policy_for(model, policy)
    updated, greedy = {}, {}
    for state, actions in model.rows:
        if not actions:
            updated[state] = 0.0
            continue
        pairs = [
            (a, action_backup(model, state, a, fixed)) for a, _ in actions
        ]
        action, score = max(pairs, key=lambda pair: pair[1])
        greedy[state] = action  # exact float ties retain declaration order
        updated[state] = (
            score if selected is None else dict(pairs)[selected[state]]
        )
    return updated, greedy


@dataclass(frozen=True)
class Iteration:
    values: tuple
    policy: tuple
    iterations: int
    residual: float
    error_bound: float


def iterate(model, policy=None, tolerance=1e-10, max_iterations=10000):
    if model.discount == 1:
        raise ValueError("generic iteration requires discount < 1")
    tolerance = finite_real(tolerance, "tolerance", 1e6)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    count(max_iterations, "max_iterations", 1, 10000)
    selected = None if policy is None else policy_for(model, policy)
    values = {s: 0.0 for s, _ in model.rows}
    for iteration in range(1, max_iterations + 1):
        values, _ = sweep(model, values, selected)
        checked, greedy = sweep(model, values, selected)
        residual = max(abs(checked[s] - values[s]) for s in values)
        bound = finite_real(residual / (1 - model.discount), "error bound")
        if bound <= tolerance:
            return Iteration(
                tuple(values.items()),
                tuple(greedy.items()),
                iteration,
                residual,
                bound,
            )
    raise RuntimeError(
        "iteration budget exhausted before numerical residual test"
    )


def tram_model(n=10, failure=Fraction(2, 5), discount=1):
    count(n, "n", 2, 50)
    if not isinstance(failure, Fraction) or not 0 <= failure < 1:
        raise ValueError("failure must be a Fraction in [0, 1)")
    if failure.denominator > 10**12:
        raise ValueError("failure denominator exceeds 10**12")
    rows = []
    for i in range(1, n + 1):
        actions = []
        if i < n:
            actions.append(("walk", (Outcome(Fraction(1), -1, str(i + 1)),)))
            if 2 * i <= n:
                outcomes = [Outcome(1 - failure, -2, str(2 * i))]
                if failure:
                    outcomes.append(Outcome(failure, -2, str(i)))
                actions.append(("tram", tuple(outcomes)))
        rows.append((str(i), tuple(actions)))
    return Model(tuple(rows), discount)


def exact_tram(n=10, failure=Fraction(2, 5), mode="optimal"):
    tram_model(n, failure)  # validate the same proper undiscounted model
    if mode not in ("walk", "tram", "optimal"):
        raise ValueError("mode must be walk, tram, or optimal")
    values, policy = {n: Fraction(0)}, {}
    for state in range(n - 1, 0, -1):
        choices = [("walk", -1 + values[state + 1])]
        if 2 * state <= n:
            choices.append(
                ("tram", -Fraction(2) / (1 - failure) + values[2 * state])
            )
        if mode == "walk":
            action, value = choices[0]
        elif mode == "tram":
            action, value = choices[-1]
        else:
            action, value = max(choices, key=lambda pair: pair[1])
        policy[state], values[state] = action, value
    return values, policy


def dice_model(discount=1):
    return Model(
        (
            (
                "in",
                (
                    ("quit", (Outcome(Fraction(1), 10, "end"),)),
                    (
                        "stay",
                        (
                            Outcome(Fraction(1, 3), 4, "end"),
                            Outcome(Fraction(2, 3), 4, "in"),
                        ),
                    ),
                ),
            ),
            ("end", ()),
        ),
        discount,
    )


def rollout(model, policy, start, rng, max_steps=10000):
    selected = policy_for(model, policy)
    table = model.table()
    if (
        not isinstance(start, str)
        or start not in table
        or not isinstance(rng, Random)
    ):
        raise ValueError(
            "start must be declared and rng must be random.Random"
        )
    count(max_steps, "max_steps", 1, 100000)
    state, weight, rewards = start, 1.0, []
    for _ in range(max_steps):
        if not table[state]:
            return finite_real(fsum(rewards), "return")
        draw, cumulative = rng.random(), 0.0
        outcomes = table[state][selected[state]]
        chosen = outcomes[-1]
        for outcome in outcomes:
            cumulative += float(outcome.probability)
            if draw < cumulative:
                chosen = outcome
                break
        rewards.append(weight * chosen.reward)
        weight *= model.discount
        state = chosen.successor
    if not table[state]:
        return finite_real(fsum(rewards), "return")
    raise RuntimeError("rollout budget exhausted before terminal arrival")


def mean_and_se(returns):
    if not isinstance(returns, (tuple, list)):
        raise ValueError("returns must be a tuple/list")
    count(len(returns), "return count", 2, 100000)
    xs = [finite_real(x, "return", 1e6) for x in returns]
    exact = [Fraction.from_float(x) for x in xs]
    exact_mean = sum(exact, Fraction()) / len(xs)
    mean = float(exact_mean)
    if exact_mean and mean == 0:
        raise ValueError("nonzero mean is below binary64 representability")
    squares = sum(((x - exact_mean) ** 2 for x in exact), Fraction())
    exact_se_squared = squares / (len(xs) * (len(xs) - 1))
    with localcontext() as context:
        context.prec = 80
        decimal_variance = Decimal(exact_se_squared.numerator) / Decimal(
            exact_se_squared.denominator
        )
        se = float(decimal_variance.sqrt())
    if exact_se_squared and se == 0:
        raise ValueError(
            "nonzero standard error is below binary64 representability"
        )
    return mean, se


def main():
    tram = tram_model()
    initial = {str(i): (-100.0 if i < 10 else 0.0) for i in range(1, 11)}
    print(
        f"first tram backup at state 5: {
            action_backup(tram, '5', 'tram', initial):.6f}"
    )
    for mode in ("walk", "tram", "optimal"):
        values, _ = exact_tram(mode=mode)
        print(f"{mode} start value: {values[1]} ({float(values[1]):.6f})")
    values, policy = exact_tram()
    print(
        "optimal actions:",
        " ".join(f"{s}:{policy[s]}" for s in sorted(policy)),
    )
    for gamma in (0, 0.5, 0.9, 1):
        stay = 4 / (1 - 2 * gamma / 3)
        print(f"dice discount {gamma}: stay={stay:.6f}, quit=10.000000")
    result = iterate(dice_model(0.5), {"in": "stay"})
    print(
        f"discounted evaluation: V={dict(result.values)['in']:.9f}, bound={
            result.error_bound:.3e}"
    )
    rng = Random(7)
    fixed_policy = {str(s): a for s, a in policy.items()}
    returns = [rollout(tram, fixed_policy, "1", rng) for _ in range(2000)]
    mean, se = mean_and_se(returns)
    print(
        f"2000 original optimal rollouts: mean={mean:.6f}, sample SE={se:.6f}"
    )
    print(f"analytic optimal return SD: {sqrt(40 / 9):.6f}")


if __name__ == "__main__":
    main()
