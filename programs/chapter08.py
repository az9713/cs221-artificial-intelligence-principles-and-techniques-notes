'Original tabular RL mechanics; standard library, no ' \
    'file/network effects.\n\nKnown finite state/action ' \
    'catalogue; unknown transition probabilities/rewards.\n' \
    'Invalid inputs and nonrepresentable nonzero arithmetic ' \
    'raise ValueError;\nunfinished episodes raise ' \
    'RuntimeError. Finite runs are not convergence proofs.\n'

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from random import Random


def finite_real(value, name, bound=1e100):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be an int/float excluding bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " must be binary64 representable") from exc
    if not isfinite(result) or abs(result) > bound:
        raise ValueError(name + " exceeds the finite documented range")
    return result


def rounded(exact, name):
    result = finite_real(float(exact), name)
    if exact and result == 0:
        raise ValueError(name + " is nonzero but rounds below binary64")
    return result


def unit(value, name):
    value = finite_real(value, name, 1)
    if value < 0:
        raise ValueError(name + " must lie in [0,1]")
    return value


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return value


def catalogue(actions):
    if not isinstance(actions, dict):
        raise ValueError("catalogue must be a dict")
    integer(len(actions), "state count", 1, 64)
    fixed = {}
    for state, row in actions.items():
        if not isinstance(state, str) or not state:
            raise ValueError("states must be nonempty strings")
        if not isinstance(row, (tuple, list)) or len(row) > 16:
            raise ValueError("action rows must be tuple/list of length <=16")
        if any(not isinstance(a, str) or not a for a in row) or len(
            set(row)
        ) != len(row):
            raise ValueError("action names must be distinct nonempty strings")
        fixed[state] = tuple(row)
    return fixed


def tram_actions(n=10):
    integer(n, "n", 2, 50)
    return {
        str(s): (
            () if s == n else (("walk", "tram") if 2 * s <= n else ("walk",))
        )
        for s in range(1, n + 1)
    }


@dataclass(frozen=True)
class Sample:
    state: str
    action: str
    reward: float
    next_state: str
    done: bool


def sample_for(actions, sample):
    fixed = catalogue(actions)
    if not isinstance(sample, Sample):
        raise ValueError("feedback must be a Sample")
    if (
        not isinstance(sample.state, str)
        or sample.state not in fixed
        or not isinstance(sample.action, str)
        or sample.action not in fixed[sample.state]
    ):
        raise ValueError(
            "feedback requires a declared nonterminal admissible action"
        )
    if (
        not isinstance(sample.next_state, str)
        or sample.next_state not in fixed
    ):
        raise ValueError("successor must be declared")
    if type(sample.done) is not bool or sample.done != (
        not fixed[sample.next_state]
    ):
        raise ValueError("done must agree with declared terminal successor")
    reward = finite_real(sample.reward, "reward")
    return Sample(
        sample.state, sample.action, reward, sample.next_state, sample.done
    )


def initial_q(actions):
    return {(s, a): 0.0 for s, row in catalogue(actions).items() for a in row}


def q_for(actions, q):
    expected = initial_q(actions)
    if not isinstance(q, dict) or set(q) != set(expected):
        raise ValueError(
            "Q must contain exactly all admissible state-action pairs"
        )
    return {pair: finite_real(q[pair], "Q value") for pair in expected}


def greedy(actions, q, state):
    fixed, q = catalogue(actions), q_for(actions, q)
    if not isinstance(state, str) or state not in fixed or not fixed[state]:
        raise ValueError("greedy requires a declared nonterminal state")
    return max(fixed[state], key=lambda action: q[(state, action)])


def epsilon_probabilities(actions, q, state, epsilon):
    fixed = catalogue(actions)
    best = greedy(fixed, q, state)
    epsilon = unit(epsilon, "epsilon")
    if epsilon and epsilon / len(fixed[state]) == 0:
        raise ValueError(
            "positive per-action exploration probability rounds to zero"
        )
    return {
        a: epsilon / len(fixed[state]) + (1 - epsilon if a == best else 0)
        for a in fixed[state]
    }


def choose(actions, q, state, epsilon, rng):
    if not isinstance(rng, Random):
        raise ValueError("rng must be random.Random")
    probs = epsilon_probabilities(actions, q, state, epsilon)
    draw, cumulative = rng.random(), 0.0
    for action, prob in probs.items():
        cumulative += prob
        if draw < cumulative:
            return action
    return next(reversed(probs))  # finite summation boundary only


class EmpiricalModel:
    def __init__(self, actions):
        self.actions = catalogue(actions)
        self.counts, self.rewards = {}, {}

    def add(self, sample):
        sample = sample_for(self.actions, sample)
        key = (sample.state, sample.action, sample.next_state)
        if sum(self.counts.values()) >= 1000000:
            raise ValueError("feedback budget exceeds 1000000")
        self.counts[key] = self.counts.get(key, 0) + 1
        self.rewards[key] = self.rewards.get(
            key, Fraction()
        ) + Fraction.from_float(sample.reward)

    def row(self, state, action):
        if (
            not isinstance(state, str)
            or state not in self.actions
            or not isinstance(action, str)
            or action not in self.actions[state]
        ):
            raise ValueError("row requires an admissible state-action pair")
        keys = [key for key in self.counts if key[:2] == (state, action)]
        total = sum(self.counts[key] for key in keys)
        if not total:
            raise ValueError(
                ("unobserved action has no empirical distribution")
            )
        return tuple(
            (
                key[2],
                Fraction(self.counts[key], total),
                self.rewards[key] / self.counts[key],
            )
            for key in keys
        )

    def coverage(self):
        return tuple(
            (s, a)
            for s, row in self.actions.items()
            for a in row
            if not any(key[:2] == (s, a) for key in self.counts)
        )


def plan_empirical(model, gamma=0.9, tolerance=1e-9, max_iterations=10000):
    if not isinstance(model, EmpiricalModel) or model.coverage():
        raise ValueError(
            "planning requires feedback for every admissible pair"
        )
    gamma = unit(gamma, "gamma")
    if gamma == 1:
        raise ValueError("generic empirical planning requires discount <1")
    tolerance = finite_real(tolerance, "tolerance", 1e6)
    if tolerance <= 0:
        raise ValueError("planning tolerance must be positive")
    integer(max_iterations, "max_iterations", 1, 10000)
    rows = {
        (s, a): model.row(s, a)
        for s, row in model.actions.items()
        for a in row
    }
    rate = Fraction.from_float(gamma)

    def backup(q):
        result = {}
        for pair, row in rows.items():
            target = Fraction()
            for successor, prob, reward in row:
                continuation = (
                    max(q[(successor, a)] for a in model.actions[successor])
                    if model.actions[successor]
                    else 0.0
                )
                target += prob * (
                    reward + rate * Fraction.from_float(continuation)
                )
            result[pair] = rounded(target, "empirical backup")
        return result

    q = initial_q(model.actions)
    if not q:
        return q, {}, 0, 0.0
    for iteration in range(1, max_iterations + 1):
        q = backup(q)
        checked = backup(q)
        residual = max(abs(checked[pair] - q[pair]) for pair in q)
        bound = finite_real(residual / (1 - gamma), (
            'numerical planning bound'
        ))
        if bound <= tolerance:
            policy = {
                s: greedy(model.actions, q, s)
                for s in model.actions
                if model.actions[s]
            }
            return q, policy, iteration, bound
    raise RuntimeError("empirical planning iteration budget exhausted")


def suffix_returns(actions, episode, gamma):
    gamma = unit(gamma, "gamma")
    if not isinstance(episode, (tuple, list)):
        raise ValueError("episode must be tuple/list")
    integer(len(episode), "episode length", 1, 10000)
    checked = [sample_for(actions, sample) for sample in episode]
    for i, sample in enumerate(checked):
        if sample.done != (i == len(checked) - 1):
            raise ValueError(
                (
                    "only the final feedback may be terminal, and it must "
                    "be terminal"
                )
            )
        if i and checked[i - 1].next_state != sample.state:
            raise ValueError("episode feedback must be contiguous")
    returns, continuation = [0.0] * len(checked), Fraction()
    discount = Fraction.from_float(gamma)
    for i in range(len(checked) - 1, -1, -1):
        continuation = (
            Fraction.from_float(checked[i].reward) + discount * continuation
        )
        returns[i] = rounded(continuation, "suffix return")
    return tuple(returns)


def monte_carlo(actions, episode, gamma, first_visit=False):
    if type(first_visit) is not bool:
        raise ValueError("first_visit must be bool")
    returns = suffix_returns(actions, episode, gamma)
    sums, counts = {}, {}
    for sample, value in zip(episode, returns):
        key = (sample.state, sample.action)
        if first_visit and key in counts:
            continue
        sums[key] = sums.get(key, Fraction()) + Fraction.from_float(value)
        counts[key] = counts.get(key, 0) + 1
    return {
        key: rounded(sums[key] / counts[key], "MC mean") for key in sums
    }, counts


def td_update(actions, q, sample, gamma, alpha, method, next_action=None):
    sample = sample_for(actions, sample)
    gamma, alpha = unit(gamma, "gamma"), unit(alpha, "alpha")
    q = q_for(actions, q)
    if method not in ("sarsa", "q_learning"):
        raise ValueError("method must be sarsa or q_learning")
    if sample.done:
        if next_action is not None:
            raise ValueError("terminal feedback has no next action")
        bootstrap = 0.0
    elif method == "sarsa":
        if (
            not isinstance(next_action, str)
            or next_action not in catalogue(actions)[sample.next_state]
        ):
            raise ValueError(
                "SARSA requires the actual admissible next action"
            )
        bootstrap = q[(sample.next_state, next_action)]
    else:
        if next_action is not None:
            raise ValueError(
                "Q-learning does not consume a sampled next action"
            )
        bootstrap = q[
            (sample.next_state, greedy(actions, q, sample.next_state))
        ]
    target_exact = Fraction.from_float(sample.reward) + Fraction.from_float(
        gamma
    ) * Fraction.from_float(bootstrap)
    target = rounded(target_exact, "TD target")
    pair = (sample.state, sample.action)
    rate = Fraction.from_float(alpha)
    new = rounded(
        (1 - rate) * Fraction.from_float(q[pair]) + rate * target_exact,
        "updated Q",
    )
    q[pair] = new
    return q, target


def tram_step(state, action, rng, n=10, failure=0.4):
    actions = tram_actions(n)
    failure = unit(failure, "failure")
    if failure == 1:
        raise ValueError(
            "this simulator requires positive tram success probability"
        )
    if (
        not isinstance(state, str)
        or state not in actions
        or not isinstance(action, str)
        or action not in actions[state]
        or not isinstance(rng, Random)
    ):
        raise ValueError(
            "step requires admissible state/action and random.Random"
        )
    s = int(state)
    successor = (
        s + 1 if action == (
            'walk'
        ) else (s if rng.random() < failure else 2 * s)
    )
    return Sample(
        state,
        action,
        -1.0 if action == "walk" else -2.0,
        str(successor),
        successor == n,
    )


def train(method, episodes=2000, seed=7, n=10, gamma=0.9, epsilon=0.2):
    if method not in ("sarsa", "q_learning"):
        raise ValueError("method must be sarsa or q_learning")
    integer(episodes, "episodes", 1, 10000)
    integer(seed, "seed", 0, 2**32 - 1)
    gamma, epsilon = unit(gamma, "gamma"), unit(epsilon, "epsilon")
    actions = tram_actions(n)
    q = initial_q(actions)
    visits = {pair: 0 for pair in q}
    rng = Random(seed)
    training_returns = []
    for _ in range(episodes):
        state = "1"
        action = choose(actions, q, state, epsilon, rng)
        episode = []
        for _ in range(10000):
            sample = tram_step(state, action, rng, n)
            episode.append(sample)
            next_action = (
                None
                if sample.done
                else choose(actions, q, sample.next_state, epsilon, rng)
            )
            pair = (state, action)
            visits[pair] += 1
            alpha = (visits[pair] + 1) ** (-0.7)
            q, _ = td_update(
                actions,
                q,
                sample,
                gamma,
                alpha,
                method,
                next_action if method == "sarsa" else None,
            )
            if sample.done:
                training_returns.append(
                    suffix_returns(actions, episode, gamma)[0]
                )
                break
            # The sampled SARSA target action is actually executed next.
            state, action = sample.next_state, next_action
        else:
            raise RuntimeError(
                "training rollout exhausted before terminal arrival"
            )
    return q, visits, tuple(training_returns)


def exact_policy_value(
    policy, n=10, gamma=Fraction(9, 10), failure=Fraction(2, 5)
):
    actions = tram_actions(n)
    if (
        not isinstance(gamma, Fraction)
        or not 0 <= gamma <= 1
        or gamma.denominator > 10**12
        or not isinstance(failure, Fraction)
        or not 0 <= failure < 1
        or failure.denominator > 10**12
    ):
        raise ValueError(
            (
                "oracle requires bounded rational discount and proper "
                "failure probability"
            )
        )
    required = {s for s in actions if actions[s]}
    if policy is not None and (
        not isinstance(policy, dict)
        or set(policy) != required
        or any(
            not isinstance(policy[s], str) or policy[s] not in actions[s]
            for s in required
        )
    ):
        raise ValueError(
            "oracle policy requires every nonterminal admissible action"
        )
    values = {n: Fraction()}
    for s in range(n - 1, 0, -1):
        choices = {"walk": -1 + gamma * values[s + 1]}
        if 2 * s <= n:
            choices["tram"] = (-2 + gamma * (1 - failure) * values[2 * s]) / (
                1 - gamma * failure
            )
        values[s] = (
            max(choices.values())
            if policy is None
            else choices[policy[str(s)]]
        )
    return values


def main():
    actions = tram_actions()
    model = EmpiricalModel(actions)
    for successor in ["2"] * 2 + ["4"] * 5:
        model.add(Sample("2", "tram", -2, successor, False))
    print(
        "estimated tram2 row:",
        " ".join(f"{s}:P={p},r={r}" for s, p, r in model.row("2", "tram")),
    )
    rewards = EmpiricalModel({"s": ("go",), "end": ()})
    for reward in [0, 4]:
        rewards.add(Sample("s", "go", reward, "end", True))
    print("stochastic reward mean:", rewards.row("s", "go")[0][2])
    short = {"1": ("walk",), "2": ("tram",), "4": ()}
    episode = [
        Sample("1", "walk", -1, "2", False),
        Sample("2", "tram", -2, ("2"), False),
        Sample(("2"), ("tram"), -2, ("4"), True),
    ]
    print(
        "complete artificial-terminal4 suffixes:",
        suffix_returns(short, episode, 1),
    )
    every, _ = monte_carlo(short, episode, 1)
    first, _ = monte_carlo(short, episode, 1, True)
    print(
        f"tram2 MC: every={every[(('2'), ('tram'))]:.6f}, first={
            first[(('2'), ('tram'))]:.6f}"
    )
    fixture = {"s": ("go",), "next": ("low", "high"), "end": ()}
    q = initial_q(fixture)
    q.update({("s", "go"): 3, ("next", "low"): 2, ("next", "high"): 8})
    sample = Sample("s", "go", -1, "next", False)
    for method in ["sarsa", "q_learning"]:
        updated, target = td_update(
            fixture,
            q,
            sample,
            0.9,
            0.5,
            method,
            ("low") if method == ("sarsa") else None,
        )
        print(f"{method} target={target:.6f}, new={updated[('s', 'go')]:.6f}")
    done = Sample("s", "go", -1, "end", True)
    updated, target = td_update(fixture, q, done, 0.9, 0.5, "sarsa")
    print(f"terminal target={target:.6f}, new={updated[('s', 'go')]:.6f}")
    print(
        "epsilon.2 with two actions:",
        epsilon_probabilities(fixture, q, "next", 0.2),
    )
    oracle = exact_policy_value(None)[1]
    print(f"discount.9 true optimum: {float(oracle):.6f}")
    fitted = EmpiricalModel(actions)
    rng = Random(19)
    empty_q = initial_q(actions)
    for _ in range(500):
        state = "1"
        for _ in range(10000):
            action = choose(actions, empty_q, state, 1, rng)
            feedback = tram_step(state, action, rng)
            fitted.add(feedback)
            if feedback.done:
                break
            state = feedback.next_state
        else:
            raise RuntimeError("model exploration rollout exhausted")
    _, frozen_policy, _, _ = plan_empirical(fitted)
    print(
        f"model-based 500 explore episodes: missing pairs={
            len(fitted.coverage())
        }, frozen V={float(exact_policy_value(frozen_policy)[1]):.6f}"
    )
    for method in ["sarsa", "q_learning"]:
        learned, visits, returns = train(method)
        policy = {
            s: greedy(actions, learned, s) for s in actions if actions[s]
        }
        frozen = exact_policy_value(policy)[1]
        print(
            f"{method} finite training: min visits={
                min(visits.values())
            }, frozen greedy V={float(frozen):.6f}"
        )
    print(
        (
            "Finite training/evaluation figures are fixtures, not "
            "an optimum guarantee."
        )
    )


if __name__ == "__main__":
    main()
