"""Original bounded numerical model for Chapter 3; Python standard library only.

Inputs are finite Python int/float scalars (not bool), converted to binary64.
Scores, weights and biases have magnitude at most 1e6. Features are normalized
nonnegative frequencies. Probabilities must sum to one within 1e-12; accepted
vectors are renormalized. Functions do not mutate inputs or perform I/O.
Invalid inputs raise ValueError. Direct cross entropy / KL return inf on a
target-positive, forecast-zero coordinate. The CLI prints a teaching report.
"""

import json
import math

LIMIT = 1_000_000
MASS_TOLERANCE = 1e-12
VOCABULARY = ("good", "bad", "not", "film", "<unk>")


def _number(value, bound=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite int/float, not bool")
    # Compare the original integer before conversion: arbitrarily large int
    # inputs can overflow float(), whereas this domain promises ValueError.
    if bound is not None and not -bound <= value <= bound:
        raise ValueError("number outside the declared magnitude bound")
    try:
        out = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("number cannot be represented as finite binary64") from exc
    if not math.isfinite(out):
        raise ValueError("number must be finite")
    return out


def _vector(values, bound=None):
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("expected a nonempty list or tuple")
    return tuple(_number(v, bound) for v in values)


def _distribution(values):
    out = _vector(values)
    if any(v < 0 or v > 1 for v in out):
        raise ValueError("probabilities must lie in [0,1]")
    total = math.fsum(out)
    if abs(total - 1.0) > MASS_TOLERANCE:
        raise ValueError("probability mass must sum to one within 1e-12")
    return tuple(v / total for v in out)


def _label(y):
    if isinstance(y, bool) or not isinstance(y, int) or y not in (-1, 1):
        raise ValueError("binary target must be integer -1 or +1")
    return y


def hard_label(score):
    """Positive precisely when score > 0; negative at a zero score."""
    return 1 if _number(score, LIMIT) > 0 else -1


def sigmoid(score):
    z = _number(score, LIMIT)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def softplus(value):
    u = _number(value, LIMIT)
    return max(u, 0.0) + math.log1p(math.exp(-abs(u)))


def binary_loss(score, y):
    z, label = _number(score, LIMIT), _label(y)
    return softplus(-label * z)


def binary_score_gradient(score, y):
    z, label = _number(score, LIMIT), _label(y)
    return -label * sigmoid(-label * z)


def _shifted_logits(logits):
    z = _vector(logits, LIMIT)
    if len(z) < 2:
        raise ValueError("categorical model requires at least two classes")
    pivot = max(range(len(z)), key=z.__getitem__)  # first maximum on ties
    shifted = tuple(v - z[pivot] for v in z)
    tail = math.fsum(math.exp(v) for j, v in enumerate(shifted) if j != pivot)
    return z, shifted, 1.0 + tail, math.log1p(tail)


def softmax(logits):
    _, u, denominator, _ = _shifted_logits(logits)
    return tuple(math.exp(v) / denominator for v in u)


def top_class(logits):
    z, _, _, _ = _shifted_logits(logits)
    return max(range(len(z)), key=z.__getitem__)


def categorical_loss(logits, target):
    """Fused soft-target cross entropy; no log of rounded probabilities."""
    _, u, _, correction = _shifted_logits(logits)
    t = _distribution(target)
    if len(t) != len(u):
        raise ValueError("target and logits dimensions must agree")
    return correction - math.fsum(tj * uj for tj, uj in zip(t, u))


def categorical_gradient(logits, target):
    p, t = softmax(logits), _distribution(target)
    if len(p) != len(t):
        raise ValueError("target and logits dimensions must agree")
    return tuple(pj - tj for pj, tj in zip(p, t))


def entropy(target):
    t = _distribution(target)
    return -math.fsum(v * math.log(v) for v in t if v > 0)


def cross_entropy(target, forecast):
    t, p = _distribution(target), _distribution(forecast)
    if len(t) != len(p):
        raise ValueError("probability dimensions must agree")
    if any(tj > 0 and pj == 0 for tj, pj in zip(t, p)):
        return math.inf
    return -math.fsum(tj * math.log(pj) for tj, pj in zip(t, p) if tj > 0)


def relative_entropy(target, forecast):
    t, p = _distribution(target), _distribution(forecast)
    if len(t) != len(p):
        raise ValueError("probability dimensions must agree")
    if any(tj > 0 and pj == 0 for tj, pj in zip(t, p)):
        return math.inf
    return math.fsum(tj * (math.log(tj) - math.log(pj))
                     for tj, pj in zip(t, p) if tj > 0)


def encode(text, vocabulary=VOCABULARY):
    if not isinstance(text, str) or not text.split():
        raise ValueError("review must be a nonempty string after whitespace splitting")
    if not isinstance(vocabulary, (tuple, list)) or not vocabulary:
        raise ValueError("vocabulary must be a nonempty list or tuple")
    if any(not isinstance(v, str) or not v or v != v.lower()
           or len(v.split()) != 1 or v.split()[0] != v for v in vocabulary):
        raise ValueError("vocabulary entries must be unique lowercase whitespace-free strings")
    if len(set(vocabulary)) != len(vocabulary) or "<unk>" not in vocabulary:
        raise ValueError("vocabulary must be unique and contain <unk>")
    lookup = {piece: j for j, piece in enumerate(vocabulary)}
    return tuple(lookup.get(piece, lookup["<unk>"]) for piece in text.lower().split())


def _indices(indices, size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("vocabulary size must be a positive integer")
    if not isinstance(indices, (tuple, list)) or not indices:
        raise ValueError("index sequence must be nonempty")
    if any(isinstance(j, bool) or not isinstance(j, int) or not 0 <= j < size
           for j in indices):
        raise ValueError("each index must be an integer in the vocabulary range")
    return tuple(indices)


def bag_of_words(indices, size):
    ids = _indices(indices, size)
    counts = [0] * size
    for j in ids:
        counts[j] += 1
    return tuple(count / len(ids) for count in counts)


def gather_mean(indices, weights):
    w = _vector(weights, LIMIT)
    ids = _indices(indices, len(w))
    # Divide before summing to avoid unnecessary accumulation overflow.
    return math.fsum(w[j] / len(ids) for j in ids)


def affine_score(features, weights, bias):
    x, w, b = _distribution(features), _vector(weights, LIMIT), _number(bias, LIMIT)
    if len(x) != len(w):
        raise ValueError("feature and weight dimensions must agree")
    z = math.fsum(xj * wj for xj, wj in zip(x, w)) + b
    return _number(z, LIMIT)


def binary_objective(features, labels, weights, bias):
    """Return unregularized MEAN loss, dJ/dw and dJ/db."""
    w, b = _vector(weights, LIMIT), _number(bias, LIMIT)
    if not isinstance(features, (tuple, list)) or not features:
        raise ValueError("training batch must be nonempty")
    if not isinstance(labels, (tuple, list)) or len(features) != len(labels):
        raise ValueError("one target is required per training row")
    x = tuple(_distribution(row) for row in features)
    if any(len(row) != len(w) for row in x):
        raise ValueError("training feature and weight dimensions must agree")
    y = tuple(_label(label) for label in labels)
    z = tuple(affine_score(row, w, b) for row in x)
    r = tuple(binary_score_gradient(zi, yi) for zi, yi in zip(z, y))
    n = len(x)
    loss = math.fsum(binary_loss(zi, yi) / n for zi, yi in zip(z, y))
    grad = tuple(math.fsum(ri * row[j] / n for ri, row in zip(r, x))
                 for j in range(len(w)))
    return loss, grad, math.fsum(ri / n for ri in r)


def train(features, labels, weights, bias, learning_rate=1.0, steps=20):
    eta = _number(learning_rate, LIMIT)
    if eta <= 0 or isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("learning rate must be positive; steps a nonnegative integer")
    w, b = _vector(weights, LIMIT), _number(bias, LIMIT)
    history = [binary_objective(features, labels, w, b)[0]]
    for _ in range(steps):
        _, dw, db = binary_objective(features, labels, w, b)
        w = tuple(_number(wj - eta * gj, LIMIT) for wj, gj in zip(w, dw))
        b = _number(b - eta * db, LIMIT)
        history.append(binary_objective(features, labels, w, b)[0])
    return w, b, tuple(history)


def report():
    features = tuple(bag_of_words(encode(text), len(VOCABULARY))
                     for text in ("good film", "bad film"))
    labels, zero = (1, -1), (0.0,) * len(VOCABULARY)
    first_w, first_b, first_history = train(features, labels, zero, 0, steps=1)
    final_w, final_b, history = train(features, labels, zero, 0, steps=20)
    t, p = (.5, .2, .3), (.1, .5, .4)
    left = bag_of_words(encode("not good but bad"), len(VOCABULARY))
    right = bag_of_words(encode("not bad but good"), len(VOCABULARY))
    return {
        "zero_score_negative_error": int(hard_label(0) != -1),
        "zero_score_nonpositive_margin_penalty": int((-1 * 0) <= 0),
        "initial_mean_loss": round(first_history[0], 12),
        "first_step_weights": list(first_w), "first_step_bias": first_b,
        "first_step_mean_loss": round(first_history[1], 12),
        "twenty_step_mean_loss": round(history[-1], 12),
        "twenty_step_losses_decrease": all(a > b for a, b in zip(history, history[1:])),
        "twenty_step_training_labels": [hard_label(affine_score(row, final_w, final_b))
                                       for row in features],
        "softmax_1_minus1_0": [round(v, 12) for v in softmax((1, -1, 0))],
        "shifted_softmax_matches": softmax((1, -1, 0)) == softmax((1000, 998, 999)),
        "soft_target_logit_loss": round(categorical_loss((1, -1, 0), t), 12),
        "soft_target_logit_gradient": [round(v, 12) for v in categorical_gradient((1, -1, 0), t)],
        "entropy": round(entropy(t), 12), "cross_entropy": round(cross_entropy(t, p), 12),
        "KL": round(relative_entropy(t, p), 12),
        "wrong_extreme_binary_loss": binary_loss(-1000, 1),
        "rounded_zero_forecast_fused_loss": categorical_loss((1000, -1000), (0, 1)),
        "top_class_tiny_positive": top_class((0, 1e-300)),
        "order_collision": left == right,
        "collision_minimum_mean_loss": round(binary_loss(0, 1), 12),
        "encoding_good_film_good": list(encode("good film good")),
        "gather_and_dot_agree": gather_mean(encode("good film good"), (1, -1, 0, 0, 0))
                                == affine_score(bag_of_words(encode("good film good"), 5), (1, -1, 0, 0, 0), 0),
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
