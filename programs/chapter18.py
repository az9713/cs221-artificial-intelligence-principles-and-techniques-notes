"""Exact finite audit fixtures; no empirical, legal or deployment \
certification.

Functions copy bounded list/tuple inputs. Exact ints and Fractions only;
booleans/floats are rejected as numbers. No files, network or randomness.
The CLI prints original examples. Stakeholder columns use stipulated common
utility units; risk probabilities and harm magnitudes are supplied assumptions.
"""
from fractions import Fraction as F


def rational(value):
    (
        'Accept an exact rational with at most 128 bits in each '
        'input component.'
    )
    if type(value) not in (int, F):
        raise ValueError('Use an exact int or Fraction, not bool/float.')
    value = F(value)
    if (value.numerator.bit_length() > 128 or
            value.denominator.bit_length() > 128):
        raise ValueError(
            'Input rational exceeds the 128-bit component budget.'
        )
    return value


def vector(values, maximum=16, lower=None, upper=None):
    (
        'Copy a nonempty finite numeric list/tuple under a '
        'declared length bound.'
    )
    if type(values) not in (tuple, list) or not 1 <= len(values) <= maximum:
        raise ValueError('Expected a nonempty bounded list or tuple.')
    copied = tuple(rational(v) for v in values)
    if any((lower is not None and v < lower) or
           (upper is not None and v > upper) for v in copied):
        raise ValueError('Numeric value outside the declared domain.')
    return copied


def distribution(weights):
    """At most sixteen nonnegative weights, summing exactly to one."""
    weights = vector(weights, lower=0, upper=1)
    if sum(weights) != 1:
        raise ValueError('Group/stakeholder weights must sum exactly to one.')
    return weights


def audit(losses, weights):
    (
        'Return assumed weighted and worst risks for up to '
        'sixteen known groups.'
    )
    losses = vector(losses, lower=0, upper=1)
    weights = distribution(weights)
    if len(losses) != len(weights):
        raise ValueError('One weight per supplied group is required.')
    worst = max(losses)
    return {
        'average': sum(w * r for w, r in zip(weights, losses)),
        'worst': worst,
        'worst_groups': tuple(i for i, r in enumerate(losses) if r == worst)
    }


def auc(positive, negative):
    (
        'Exact pairwise AUC, half credit for ties; at most 64 '
        'scores per class.'
    )
    positive = vector(positive, maximum=64)
    negative = vector(negative, maximum=64)
    pairs = len(positive) * len(negative)
    wins = sum(int(p > n) for p in positive for n in negative)
    ties = sum(int(p == n) for p in positive for n in negative)
    return F(2 * wins + ties, 2 * pairs)


def threshold_summary(positive, negative, threshold):
    (
        'Classify score >= threshold as positive; both true '
        'classes must exist.'
    )
    positive = vector(positive, maximum=64)
    negative = vector(negative, maximum=64)
    threshold = rational(threshold)
    tp = sum(p >= threshold for p in positive)
    tn = sum(n < threshold for n in negative)
    return {'true_positives': tp, 'true_negatives': tn,
            'sensitivity': F(tp, len(positive)),
            'specificity': F(tn, len(negative)),
            'accuracy': F(tp + tn, len(positive) + len(negative))}


def stakeholder_choice(utilities, weights, criterion='weighted'):
    (
        'Finite social choice under a supplied rule; first action'
        ' wins exact ties.\n'
        '\n'
        '    Rows are actions and columns are stakeholder '
        'utilities in common assumed\n'
        "    units. Weighted or maxmin aggregation is a caller's "
        'normative choice.\n'
        '    Both the matrix and stakeholder weights have at most'
        ' sixteen entries.\n'
        '    '
    )
    weights = distribution(weights)
    if (type(utilities) not in (list, tuple) or
            not 1 <= len(utilities) <= 16):
        raise ValueError('Expected one to sixteen action rows.')
    rows = tuple(vector(row) for row in utilities)
    if any(len(row) != len(weights) for row in rows):
        raise ValueError('Utility columns must match stakeholder weights.')
    if type(criterion) is not str:
        raise ValueError('Criterion must be a string.')
    if criterion == 'weighted':
        scores = tuple(
            sum(w * u for w, u in zip(weights, row)) for row in rows
        )
    elif criterion == 'maxmin':
        scores = tuple(min(row) for row in rows)
    else:
        raise ValueError('Criterion must be weighted or maxmin.')
    chosen = scores.index(max(scores))
    return {'chosen': chosen, 'scores': scores, 'utilities': rows[chosen]}


def proxy_selection(proxy, true_values):
    (
        'Compare finite proxy and stipulated true-value '
        'maximizers, first at ties.'
    )
    proxy = vector(proxy)
    true_values = vector(true_values)
    if len(proxy) != len(true_values):
        raise ValueError('Both value vectors must describe the same actions.')
    chosen = proxy.index(max(proxy))
    best = true_values.index(max(true_values))
    return {'proxy_action': chosen, 'true_value_action': best,
            'true_value_regret': true_values[best] - true_values[chosen]}


def suffix_probability(probabilities):
    """Product of up to 1000 supplied target-token conditionals along a prefix.

    These conditionals are along the specified target trajectory, under one
    declared decoding law. Their product is not a training-membership test.
    """
    probabilities = vector(probabilities, maximum=1000, lower=0, upper=1)
    result = F(1)
    for value in probabilities:
        result *= value
    return result


def at_least_once(probability, trials):
    """Exact success in 0..1000 independent identically distributed trials."""
    probability = rational(probability)
    if not 0 <= probability <= 1:
        raise ValueError('Probability must be in [0,1].')
    if type(trials) is not int or not 0 <= trials <= 1000:
        raise ValueError('Trials must be an integer in 0..1000.')
    return 1 - (1 - probability) ** trials


def marginal_risk(base_probability, new_probability,
                  base_severity=1, new_severity=1):
    """Difference of supplied expected harms in the same stated harm units."""
    bp, np = rational(base_probability), rational(new_probability)
    bs, ns = rational(base_severity), rational(new_severity)
    if not 0 <= bp <= 1 or not 0 <= np <= 1 or bs < 0 or ns < 0:
        raise ValueError(
            'Probability or nonnegative severity domain violated.'
        )
    return np * ns - bp * bs


def disclosure_score(indicators):
    (
        'Fraction of true Boolean disclosures; not '
        'truthfulness/safety scoring.'
    )
    if (type(indicators) is not dict
            or not 1 <= len(indicators) <= 128):
        raise ValueError('Expected one to 128 named Boolean indicators.')
    for name, value in indicators.items():
        if (type(name) is not str or not 1 <= len(name) <= 64 or
                not name.isascii() or not name.isidentifier() or
                type(value) is not bool):
            raise ValueError(
                'Indicator names and Boolean values are required.'
            )
    return F(sum(indicators.values()), len(indicators))


def demo():
    weights = [F(99, 100), F(1, 100)]
    for name, losses in [('A', [F(1, 100), F(4, 5)]),
                         ('B', [F(1, 10), F(1, 5)])]:
        row = audit(losses, weights)
        print(name, 'average:', row['average'], 'worst:', row['worst'])
    positive, negative = [F(1, 5), F(9, 10)], [F(1, 10), F(4, 5)]
    threshold = threshold_summary(positive, negative, F(1, 2))
    print('pooled AUC:', auc(positive, negative),
          'threshold accuracy:', threshold['accuracy'])
    proxy = proxy_selection([0, 1, 2, 3], [0, 1, F(4, 5), -1])
    print('proxy regret:', proxy['true_value_regret'])
    rows = [[4, 0], [0, 4], [3, 3]]
    print('weighted/maxmin choices:',
          stakeholder_choice(rows, [1, 0])['chosen'],
          stakeholder_choice(rows, [1, 0], 'maxmin')['chosen'])
    print('suffix .9^10:', suffix_probability([F(9, 10)] * 10))
    print('two independent fair attempts:', at_least_once(F(1, 2), 2))
    print('risk increments:',
          marginal_risk(F(1, 100), F(3, 100)),
          marginal_risk(F(1, 100), F(1, 200)))
    print('disclosure fraction:',
          disclosure_score({'data': True, 'labor': False}))
    print('Supplied finite assumptions; '
          'no empirical or deployment certification.')


if __name__ == '__main__':
    demo()
