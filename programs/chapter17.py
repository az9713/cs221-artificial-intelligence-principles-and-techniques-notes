"""Original finite language-model fixtures; no neural training or GPU claims.

All mathematical probabilities use int/Fraction inputs and exact normalization.
The small fused logit loss and preference tilt use bounded binary64 arithmetic.
Importing defines objects only. Running prints a deterministic teaching report.
Functions never write files, use the network, mutate supplied containers, or
share a random generator with the caller except the explicit draw() argument.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import exp, fsum, isfinite, lcm, log
from random import Random

EOS = "EOS"
BOS = "BOS"
WORDS = (
    "The",
    "and",
    "crashed",
    "investors",
    "market",
    "stock",
    "lol",
    "panicked",
    "celebrated",
    "golfing",
)
PREFIX = ("The", "stock", "market", "crashed", "and", "investors")


def _sequence(value, name, maximum, allow_empty=True):
    if not isinstance(value, (list, tuple)):
        raise ValueError(name + " must be a list or tuple")
    result = tuple(value)
    if len(result) > maximum or (not allow_empty and not result):
        raise ValueError(name + " length outside the finite contract")
    return result


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name + " must be an integer in the stated interval")
    return value


def _rational(value, name, maximum=10**12):
    if type(value) is int:
        result = Fraction(value)
    elif type(value) is Fraction:
        result = value
    else:
        raise ValueError(name + " must be an int or Fraction, not float/bool")
    if abs(result.numerator) > maximum or result.denominator > maximum:
        raise ValueError(name + " exceeds the rational representation budget")
    return result


def _distribution(value):
    row = _sequence(value, "probability row", 33, allow_empty=False)
    row = tuple(_rational(p, "probability") for p in row)
    if any(p < 0 for p in row) or sum(row) != 1:
        raise ValueError(
            "probabilities must be nonnegative and sum exactly to one"
        )
    return row


def _float(value, name, bound=10000):
    if type(value) not in (int, float):
        raise ValueError(name + " must be int/float, not bool")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " conversion failed") from error
    if not isfinite(converted) or abs(converted) > bound:
        raise ValueError(name + " outside the finite numeric contract")
    return converted


def draw(probabilities, rng):
    (
        'One exact integer-interval categorical draw; adv'
        'ances only supplied RNG.'
    )
    row = _distribution(probabilities)
    if not isinstance(rng, Random):
        raise ValueError("rng must be random.Random")
    denominator = lcm(*(p.denominator for p in row))
    if denominator > 10**12:
        raise ValueError("common draw denominator exceeds 10**12")
    ticket = rng.randrange(denominator)
    cumulative = 0
    for index, probability in enumerate(row):
        cumulative += probability.numerator * (
            denominator // probability.denominator
        )
        if ticket < cumulative:
            return index
    raise AssertionError("validated probability intervals must cover ticket")


@dataclass(frozen=True)
class Bigram:
    """Immutable exact count table; fit documents or validate explicit counts.

    Input contexts are lexical vocabulary followed by input-only BOS. Outputs
    are lexical vocabulary followed by EOS. Equal last indices have different
    context/output meanings; public calls use token strings, not those indices.
    """

    words: tuple
    counts: tuple
    alpha: Fraction

    def __post_init__(self):
        words = _sequence(
            self.words, "lexical vocabulary", 32, allow_empty=False
        )
        if any(type(w) is not str or not w or w in (EOS, BOS) for w in words):
            raise ValueError(
                "words must be nonempty strings excluding BOS/EOS"
            )
        if len(set(words)) != len(words):
            raise ValueError("lexical vocabulary must be unique")
        alpha = _rational(self.alpha, "alpha", 10**6)
        if not 0 <= alpha <= 10**6:
            raise ValueError("alpha must be nonnegative and at most 10**6")
        rows = _sequence(self.counts, "counts", 33, allow_empty=False)
        if len(rows) != len(words) + 1:
            raise ValueError("one count row per lexical/BOS input context")
        copied = []
        for row in rows:
            row = _sequence(row, "count row", 33, allow_empty=False)
            if len(row) != len(words) + 1:
                raise ValueError("one output count per lexical/EOS token")
            copied.append(tuple(_integer(c, "count", 0, 32768) for c in row))
        if sum(map(sum, copied)) > 32768:
            raise ValueError("count table exceeds 32768 targets")
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "counts", tuple(copied))
        object.__setattr__(self, "alpha", alpha)

    @property
    def vocabulary(self):
        return self.words + (EOS,)

    def _prefix(self, tokens, maximum=4096):
        tokens = _sequence(tokens, "lexical sequence", maximum)
        if any(type(t) is not str or t not in self.words for t in tokens):
            raise ValueError(
                "prefix/document must contain lexical tokens only"
            )
        return tokens

    @classmethod
    def fit(cls, words, documents, alpha=Fraction(1)):
        """Counts each separate document from BOS through EOS exactly once."""
        words = _sequence(words, "lexical vocabulary", 32, allow_empty=False)
        empty = cls(
            words,
            tuple((0,) * (len(words) + 1) for _ in range(len(words) + 1)),
            alpha,
        )
        documents = _sequence(documents, "documents", 1024, allow_empty=False)
        rows = [[0] * (len(words) + 1) for _ in range(len(words) + 1)]
        target_count = 0
        for document in documents:
            document = empty._prefix(document, 128)
            target_count += len(document) + 1
            if target_count > 32768:
                raise ValueError("corpus exceeds 32768 valid targets")
            previous = len(words)
            for token in document + (EOS,):
                index = empty.vocabulary.index(token)
                rows[previous][index] += 1
                previous = index
        return cls(words, tuple(map(tuple, rows)), empty.alpha)

    def row(self, prefix=()):
        prefix = self._prefix(prefix)
        context = self.words.index(prefix[-1]) if prefix else len(self.words)
        counts = self.counts[context]
        total = sum(counts) + self.alpha * len(self.vocabulary)
        if total == 0:
            raise ValueError(
                "unseen unsmoothed context has no selected distribution"
            )
        return tuple((count + self.alpha) / total for count in counts)

    def completed_probability(self, document):
        """Probability of this lexical document followed immediately by EOS."""
        document = self._prefix(document)
        probability = Fraction(1)
        prefix = ()
        for token in document + (EOS,):
            probability *= self.row(prefix)[self.vocabulary.index(token)]
            if token != EOS:
                prefix += (token,)
        return probability

    def capped_mass(self, max_length):
        (
            'Exact mass terminating with <= max_length lexica'
            'l tokens, plus survival.\n\n        Every row enco'
            'untered by the finite propagation must be define'
            'd. Work\n        uses <=32 ordinary states and <='
            '129 propagation steps, not an exponential\n      '
            '  enumeration of strings. Censoring means no EOS'
            ' in max_length+1 predictions.\n        '
        )
        max_length = _integer(max_length, "max_length", 0, 128)
        initial = self.row(())
        active = initial[:-1]
        completed = initial[-1]
        for _ in range(max_length):
            following = [Fraction(0)] * len(self.words)
            for index, mass in enumerate(active):
                if mass == 0:
                    continue
                row = self.row((self.words[index],))
                completed += mass * row[-1]
                for next_index, probability in enumerate(row[:-1]):
                    following[next_index] += mass * probability
            active = tuple(following)
        return completed, sum(active)

    def generate(self, prefix=(), max_steps=32, seed=0, greedy=False):
        """Generate at most max_steps new predictions, counting EOS as a step.

        Returns (new lexical tokens, completed flag). A false flag means a
        censored prefix, never a fabricated completed sequence. First ID wins
        greedy ties. Seeded sampling advances a fresh private Random instance.
        """
        prefix = self._prefix(prefix)
        max_steps = _integer(max_steps, "max_steps", 0, 4096)
        _integer(seed, "seed", 0, 2**64 - 1)
        if type(greedy) is not bool:
            raise ValueError("greedy must be bool")
        if len(prefix) + max_steps > 4096:
            raise ValueError("combined prefix/prediction budget exceeds 4096")
        rng = Random(seed)
        generated = ()
        for _ in range(max_steps):
            row = self.row(prefix + generated)
            index = (
                max(range(len(row)), key=row.__getitem__)
                if greedy
                else draw(row, rng)
            )
            token = self.vocabulary[index]
            if token == EOS:
                return generated, True
            generated += (token,)
        return generated, False

    def corpus_nll(self, documents):
        (
            'Token mean teacher-forced NLL, including EOS, no'
            ' cross-document events.\n\n        Returns float n'
            'ats/target. A supported-format zero target proba'
            'bility\n        gives infinity; malformed input o'
            'r undefined row raises ValueError.\n        '
        )
        documents = _sequence(documents, "documents", 1024, allow_empty=False)
        total = 0
        terms = []
        for document in documents:
            document = self._prefix(document, 128)
            total += len(document) + 1
            if total > 32768:
                raise ValueError("corpus exceeds 32768 valid targets")
            prefix = ()
            for token in document + (EOS,):
                p = self.row(prefix)[self.vocabulary.index(token)]
                terms.append(
                    float("inf")
                    if not p
                    else log(p.denominator) - log(p.numerator)
                )
                if token != EOS:
                    prefix += (token,)
        return fsum(terms) / total


def fixed_horizon(initial, transition, length, greedy=False):
    (
        'Exact best sequence among V**length candidates, '
        'or local greedy sequence.\n\n    This experiment h'
        'as no EOS. At most 4096 candidates and 12 positi'
        'ons.\n    Ties follow lexicographic ID order, inc'
        'luding greedy row ties.\n    '
    )
    initial = _distribution(initial)
    rows = _sequence(transition, "transition", 33, allow_empty=False)
    rows = tuple(_distribution(row) for row in rows)
    if len(rows) != len(initial) or any(
        len(row) != len(initial) for row in rows
    ):
        raise ValueError("square transitions must match initial alphabet")
    length = _integer(length, "length", 1, 12)
    if type(greedy) is not bool or len(initial) ** length > 4096:
        raise ValueError(
            "invalid mode or fixed-horizon candidate budget exceeded"
        )

    def probability(sequence):
        result = initial[sequence[0]]
        for old, new in zip(sequence, sequence[1:]):
            result *= rows[old][new]
        return result

    if greedy:
        sequence = (max(range(len(initial)), key=initial.__getitem__),)
        for _ in range(length - 1):
            row = rows[sequence[-1]]
            sequence += (max(range(len(row)), key=row.__getitem__),)
        return sequence, probability(sequence)
    choices = product(range(len(initial)), repeat=length)
    best = max(choices, key=probability)
    return best, probability(best)


def logit_loss_gradient(logits, target):
    """Fused one-target categorical NLL/gradient, bounded binary64 scores."""
    logits = _sequence(logits, "logits", 33, allow_empty=False)
    logits = tuple(_float(z, "logit") for z in logits)
    target = _integer(target, "target", 0, len(logits) - 1)
    shifted = tuple(z - max(logits) for z in logits)
    weights = tuple(exp(z) for z in shifted)
    total = fsum(weights)
    probabilities = tuple(w / total for w in weights)
    loss = log(total) - shifted[target]
    gradient = tuple(p - int(k == target) for k, p in enumerate(probabilities))
    return loss, probabilities, gradient


def preference_tilt(reference, rewards, beta=1):
    """Finite KL-regularized ideal optimum, NOT a PPO/neural-policy optimizer.

    Reference zeros remain exactly zero. Float conversion/exp can underflow;
    positive components need not remain representably positive at extremes.
    """
    reference = _distribution(reference)
    rewards = _sequence(rewards, "rewards", 33, allow_empty=False)
    if len(rewards) != len(reference):
        raise ValueError("reward and reference dimensions differ")
    rewards = tuple(_float(r, "reward") for r in rewards)
    beta = _float(beta, "beta")
    if beta < 10**-6:
        raise ValueError("beta must be at least 10**-6 and at most 10000")
    scores = tuple(
        log(p.numerator) - log(p.denominator) + r / beta
        if p
        else float("-inf")
        for p, r in zip(reference, rewards)
    )
    largest = max(scores)
    weights = tuple(
        exp(s - largest) if p else 0.0 for s, p in zip(scores, reference)
    )
    total = fsum(weights)
    return tuple(w / total for w in weights)


def byte_encode(raw, merges=()):
    (
        'Ordered byte merges, left-to-right nonoverlap; n'
        'o special/pretokenizer rules.'
    )
    if type(raw) is not bytes or len(raw) > 4096:
        raise ValueError("raw must be bytes of length at most 4096")
    merges = _sequence(merges, "merges", 64)
    available = {bytes((i,)) for i in range(256)}
    validated = []
    for pair in merges:
        pair = _sequence(pair, "merge pair", 2, allow_empty=False)
        if len(pair) != 2 or any(
            type(x) is not bytes or x not in available for x in pair
        ):
            raise ValueError(
                "merge operands must already be available byte symbols"
            )
        combined = pair[0] + pair[1]
        if combined in available or len(combined) > 4096:
            raise ValueError("merge must create a new bounded byte symbol")
        available.add(combined)
        validated.append(pair)
    tokens = tuple(bytes((i,)) for i in raw)
    for left, right in validated:
        result = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i:i + 2] == (left, right):
                result.append(left + right)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        tokens = tuple(result)
    return tokens


def byte_decode(tokens):
    tokens = _sequence(tokens, "byte tokens", 4096)
    if any(type(t) is not bytes or not t for t in tokens):
        raise ValueError("byte tokens must be nonempty bytes")
    if sum(map(len, tokens)) > 4096:
        raise ValueError("decoded bytes exceed 4096")
    return b"".join(tokens)


def report():
    documents = (
        PREFIX + ("panicked",),
        PREFIX + ("panicked",),
        PREFIX + ("celebrated",),
    )
    model = Bigram.fit(WORDS, documents)
    row = model.row(PREFIX)
    print("prefix IDs:", tuple(model.vocabulary.index(w) for w in PREFIX))
    print("smoothed reaction/EOS:", row[7], row[8], row[10])
    print("token mean NLL: %.9f" % model.corpus_nll(documents))
    print("seeded completion:", model.generate(PREFIX, 12, 3))
    print("zero-budget generation:", model.generate(PREFIX, 0, 3))
    geometric = Bigram(("a",), ((3, 1), (3, 1)), Fraction(0))
    print("completed <=2, survival:", geometric.capped_mass(2))
    trapped = Bigram(("a",), ((1, 0), (1, 0)), Fraction(0))
    print(
        "never-EOS model:", trapped.capped_mass(2), trapped.generate((), 3, 0)
    )
    start = (Fraction(3, 5), Fraction(2, 5))
    transitions = (start, (Fraction(1), Fraction(0)))
    print(
        "greedy sequence/probability:",
        fixed_horizon(start, transitions, 2, True),
    )
    print("MAP sequence/probability:", fixed_horizon(start, transitions, 2))
    print("extreme target loss:", logit_loss_gradient((1000, -1000), 1)[0])
    tilt = preference_tilt((Fraction(2, 3), Fraction(1, 3)), (0, log(4)))
    print("preference tilt:", tuple(round(p, 9) for p in tilt))
    raw = b"ababa"
    pieces = byte_encode(raw, ((b"a", b"b"), (b"ab", b"a")))
    print("byte pieces/reconstruction:", pieces, byte_decode(pieces) == raw)
    print("70B BF16/ideal2bit bytes:", 70 * 10**9 * 2, 70 * 10**9 // 4)
    print("stipulated KV cache bytes:", 2 * 32 * 1 * 4096 * 8 * 128 * 2)
    print(
        (
            'Finite checks do not certify factual support, sa'
            'fety, or neural-scale behavior.'
        )
    )


if __name__ == "__main__":
    report()
