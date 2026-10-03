"""Finite exact research-design examples, not model or career predictions.

Scalar inputs: int/Fraction, excluding bool/float, at most128bits per input
numerator/denominator. Intermediate exact arithmetic is not capped or rounded.
Functions preserve inputs; no I/O except demo. Invalid types raise TypeError;
invalid shapes/domains/resource budgets raise ValueError. Priors, utilities,
channels and intervention mechanisms are supplied rather than estimated.
"""
from fractions import Fraction as F


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, F)):
        raise TypeError('Use int or Fraction, not bool/float.')
    value = F(value)
    bits = max(abs(value.numerator).bit_length(),
               value.denominator.bit_length())
    if bits > 128:
        raise ValueError('Scalar input exceeds128bits.')
    return value


def _vector(values, limit=16, probability=False):
    if not isinstance(values, (list, tuple)):
        raise TypeError('Use list or tuple vectors.')
    if not 1 <= len(values) <= limit:
        raise ValueError('Vector size outside declared budget.')
    result = tuple(_number(v) for v in values)
    if probability and any(not 0 <= v <= 1 for v in result):
        raise ValueError('Probabilities must be in[0,1].')
    return result


def _distribution(values):
    result = _vector(values, probability=True)
    if sum(result) != 1:
        raise ValueError('Distribution must sum exactly to one.')
    return result


def _bits(values):
    if not isinstance(values, (list, tuple)):
        raise TypeError('Use list or tuple labels.')
    if not 1 <= len(values) <= 64:
        raise ValueError('Label count must be1..64.')
    if any(type(v) is not int for v in values):
        raise TypeError('Labels must be exact integers, not bool.')
    if any(v not in (0, 1) for v in values):
        raise ValueError('Labels must be0or1.')
    return tuple(values)


def binary_scores(probabilities, labels):
    """p=P(label1); decision is1iff p>1/2. Joint likelihood assumes product.

    Returns accuracy, mean Brier loss and product likelihood, not log loss.
    Zero likelihood is allowed and corresponds to infinite negative log loss.
    A supplied held-out sample is not checked for independence or leakage.
    """
    probabilities = _vector(probabilities, 64, probability=True)
    labels = _bits(labels)
    if len(probabilities) != len(labels):
        raise ValueError('Matched probability and label counts required.')
    n = len(labels)
    product = F(1)
    correct = 0
    squared = F(0)
    for p, y in zip(probabilities, labels):
        product *= p if y else 1-p
        correct += int(int(p > F(1, 2)) == y)
        squared += (p-y)**2
    return {'accuracy': F(correct, n), 'brier': squared/n,
            'likelihood': product}


def paired_changes(original, perturbed, labels):
    """Same ordered labeled cases, unchanged labels; no semantic check made."""
    original, perturbed, labels = map(_bits, (original, perturbed, labels))
    if len({len(original), len(perturbed), len(labels)}) != 1:
        raise ValueError('Matched paired case counts required.')
    n = len(labels)
    flips = sum(a != b for a, b in zip(original, perturbed))
    before = sum(a == y for a, y in zip(original, labels))
    after = sum(b == y for b, y in zip(perturbed, labels))
    return {'flip_rate': F(flips, n), 'accuracy_before': F(before, n),
            'accuracy_after': F(after, n)}


def experiment_value(prior, utilities, channel, cost=0):
    """Bayes decision: utilities[action][state], channel[state][outcome].

    Prior and each channel row normalized. At most16states/actions/outcomes.
    All original actions stay available after observation; cost in same units
    as utility, paid once regardless of outcome. First action wins exact ties.
    Zero-probability outcomes have None posterior/action and contribute zero.
    """
    prior = _distribution(prior)
    if not isinstance(utilities, (list, tuple)):
        raise TypeError('Use list or tuple utility rows.')
    if not 1 <= len(utilities) <= 16:
        raise ValueError('Action count must be1..16.')
    utilities = tuple(_vector(row) for row in utilities)
    if any(len(row) != len(prior) for row in utilities):
        raise ValueError('Each action needs one utility per state.')
    if not isinstance(channel, (list, tuple)):
        raise TypeError('Use list or tuple channel rows.')
    if len(channel) != len(prior):
        raise ValueError('Each state needs one channel row.')
    channel = tuple(_distribution(row) for row in channel)
    if len({len(row) for row in channel}) != 1:
        raise ValueError('Equal outcome counts required.')
    cost = _number(cost)
    if cost < 0:
        raise ValueError('Experiment cost must be nonnegative.')
    expected = tuple(sum(p*u for p, u in zip(prior, row))
                     for row in utilities)
    baseline = max(expected)
    before_action = expected.index(baseline)
    outcomes = []
    informed = F(0)
    for o in range(len(channel[0])):
        joint = tuple(p*row[o] for p, row in zip(prior, channel))
        mass = sum(joint)
        if mass == 0:
            outcomes.append({'mass': mass, 'posterior': None,
                             'action': None})
            continue
        posterior = tuple(p/mass for p in joint)
        values = tuple(sum(p*u for p, u in zip(posterior, row))
                       for row in utilities)
        best = max(values)
        action = values.index(best)
        informed += mass*best
        outcomes.append({'mass': mass, 'posterior': posterior,
                         'action': action})
    return {'baseline_action': before_action, 'baseline_value': baseline,
            'outcomes': tuple(outcomes), 'informed_value': informed,
            'information_value': informed-baseline,
            'net_information_value': informed-baseline-cost}


def toy_intervention(mechanism, forced_trace, latent_probability=F(1, 2)):
    """Two stipulated models: trace=U; answer=trace or answer=U.

    Force trace to0or1 while retaining U's distribution. This is a structural
    intervention in an explicit toy model, not an operation on a real LLM.
    """
    if not isinstance(mechanism, str):
        raise TypeError('Mechanism name must be a string.')
    if mechanism not in ('mediated', 'common-cause'):
        raise ValueError('Unknown mechanism.')
    if type(forced_trace) is not int:
        raise TypeError('Forced trace must be an integer.')
    if forced_trace not in (0, 1):
        raise ValueError('Forced trace must be0or1.')
    probability = _number(latent_probability)
    if not 0 <= probability <= 1:
        raise ValueError('Latent probability must be in[0,1].')
    return F(forced_trace) if mechanism == 'mediated' else probability


def demo():
    labels = [1, 0]
    a = binary_scores([F(3, 5), F(2, 5)], labels)
    b = binary_scores([F(99, 100), F(51, 100)], labels)
    print('accuracy A,B:', a['accuracy'], b['accuracy'])
    print('likelihood A,B:', a['likelihood'], b['likelihood'])
    print('Brier A,B:', a['brier'], b['brier'])
    result = experiment_value([F(1, 2), F(1, 2)],
                              [[2, 2], [0, 3]], [[1, 0], [0, 1]],
                              F(2, 5))
    print('baseline/informed:', result['baseline_value'],
          result['informed_value'])
    print('gross/net information:', result['information_value'],
          result['net_information_value'])
    print('forced trace0 probabilities:', toy_intervention('mediated', 0),
          toy_intervention('common-cause', 0))
    paired = paired_changes([1, 0, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0])
    print('paired flip rate:', paired['flip_rate'])
    print('Supplied finite research designs; no empirical certification.')


if __name__ == '__main__':
    demo()
