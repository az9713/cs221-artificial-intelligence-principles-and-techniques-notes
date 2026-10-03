"""Original exact policy-gradient laboratory; standard library only.

Inputs are built-in int/Fraction, excluding bool, with <=128-bit numerator
and denominator. Bernoulli policies have strict positive two-action support.
All scores differentiate an implicit logit, not its probability parameter.
No random training, files, network, global state or caller mutation occurs.
"""
from fractions import Fraction
from itertools import combinations, product
import json


def rational(value, label):
    if type(value) not in (int, Fraction):
        raise ValueError(label + ' must be an int or Fraction, excluding bool')
    value = Fraction(value)
    if (abs(value.numerator).bit_length() > 128
            or value.denominator.bit_length() > 128):
        raise ValueError(label + ' numerator/denominator exceed 128 bits')
    return value


def vector(values, label):
    if type(values) not in (list, tuple) or not 1 <= len(values) <= 64:
        raise ValueError(label + ' must be a list/tuple of 1..64 outcomes')
    return tuple(rational(x, label) for x in values)


def probability(p):
    p = rational(p, 'policy probability')
    if not 0 < p < 1:
        raise ValueError('policy probability must have strict support: 0<p<1')
    return p


def _moments(weights, values):
    """Internal exact arithmetic may grow beyond public-input bit bounds."""
    mean = sum((w*x for w, x in zip(weights, values)), Fraction(0))
    variance = sum((w*(x-mean)**2 for w, x in zip(weights, values)),
                   Fraction(0))
    return mean, variance


def moments(weights, values):
    """Exact normalized population mean and variance, not sample variance."""
    weights, values = vector(weights, 'weights'), vector(values, 'values')
    if len(weights) != len(values) or any(w < 0 for w in weights):
        raise ValueError('nonnegative weights need matching values')
    if sum(weights) != 1:
        raise ValueError('weights must sum exactly to one')
    return _moments(weights, values)


def score(p, action):
    """d log Bernoulli(A;p(theta))/d theta for p(theta)=sigmoid(theta)."""
    p = probability(p)
    if type(action) is not int or action not in (0, 1):
        raise ValueError('action must be built-in integer 0 or 1')
    return action-p


def bandit(p, rewards=(1, 3), baseline=0):
    p = probability(p)
    rewards = vector(rewards, 'rewards')
    baseline = rational(baseline, 'baseline')
    if len(rewards) != 2:
        raise ValueError('bandit requires two rewards')
    weights = (1-p, p)
    samples = tuple((rewards[a]-baseline)*score(p, a) for a in (0, 1))
    mean, variance = _moments(weights, samples)
    value, _ = _moments(weights, rewards)
    optimum_baseline = p*rewards[0]+(1-p)*rewards[1]
    return {'value': value, 'gradient': mean, 'variance': variance,
            'samples': samples, 'optimal_baseline': optimum_baseline}


def off_policy(p, behavior_p, rewards=(1, 3)):
    p, behavior_p = probability(p), probability(behavior_p)
    rewards = vector(rewards, 'rewards')
    if len(rewards) != 2:
        raise ValueError('two rewards required')
    target = (1-p, p)
    behavior = (1-behavior_p, behavior_p)
    scores = tuple((rewards[a]*score(p, a)) for a in (0, 1))
    # Products/ratios are internal exact arithmetic, not fresh public inputs.
    naive = sum(behavior[a]*scores[a] for a in (0, 1))
    corrected = sum(behavior[a]*(target[a]/behavior[a])*scores[a]
                    for a in (0, 1))
    return naive, corrected


def two_round(p, gamma=Fraction(1, 2)):
    """R1=2*A0, R2=1+2*A1, shared policy, independent decisions."""
    p, gamma = probability(p), rational(gamma, 'discount')
    if not 0 <= gamma <= 1:
        raise ValueError('discount must be in [0,1]')
    value = full = correct = missing = Fraction(0)
    for a0, a1 in product((0, 1), repeat=2):
        weight = (p if a0 else 1-p)*(p if a1 else 1-p)
        r1, r2 = 2*a0, 1+2*a1
        g0 = r1+gamma*r2
        u0, u1 = score(p, a0), score(p, a1)
        value += weight*g0
        full += weight*g0*(u0+u1)
        correct += weight*(u0*g0+gamma*u1*r2)
        missing += weight*(u0*g0+u1*r2)
    return {'value': value, 'full': full, 'return_to_go': correct,
            'missing_outer_discount': missing}


def four_point():
    weights = (Fraction(1, 4),)*4
    values = (-4, -6, 6, 8)
    single = moments(weights, values)
    iid_means = tuple(Fraction(x+y, 2) for x, y in product(values, repeat=2))
    distinct_means = tuple(Fraction(x+y, 2) for x, y in combinations(values, 2))
    adjusted = tuple(x-c for x, c in zip(values, (-6, -6, 6, 6)))
    return {'single': single,
            'iid_pair': moments((Fraction(1, 16),)*16, iid_means),
            'distinct_pair': moments((Fraction(1, 6),)*6, distinct_means),
            'control_variate': moments(weights, adjusted)}


def residual_gradients(x, y, reward=1, gamma=Fraction(1, 2)):
    """Full squared residual versus fixed-target semi-gradient in x,y."""
    x, y = rational(x, 'x'), rational(y, 'y')
    reward, gamma = rational(reward, 'reward'), rational(gamma, 'discount')
    if not 0 <= gamma <= 1:
        raise ValueError('discount must be in [0,1]')
    residual = x-reward-gamma*y
    return (2*residual, -2*gamma*residual), (2*residual, Fraction(0))


def stringify(value):
    if isinstance(value, dict):
        return {k: stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [stringify(v) for v in value]
    if type(value) is Fraction:
        return str(value)
    return value


def report():
    p = Fraction(1, 4)
    rows = {str(b): bandit(p, baseline=b)
            for b in (0, Fraction(5, 2), 100)}
    # Explicitly wrong action-dependent baseline and reversed critic cases.
    action_dependent = sum(w*(r-r)*score(p, a)
                           for a, w, r in ((0, 1-p, 1), (1, p, 3)))
    reversed_critic = sum(w*q*score(p, a)
                          for a, w, q in ((0, 1-p, 3), (1, p, 1)))
    compatible = sum(w*(2*score(p, a))*score(p, a)
                     for a, w in ((0, 1-p), (1, p)))
    return stringify({'bandit_baselines': rows,
                      'action_dependent_baseline_gradient': action_dependent,
                      'off_policy_naive_corrected': off_policy(p, 1-p),
                      'two_round': two_round(p),
                      'critic_reversed_compatible': (reversed_critic, compatible),
                      'four_point_mean_variance': four_point(),
                      'one_hot_selected_other': (2*Fraction(1, 10), Fraction(1, 10)),
                      'squared_full_detached_gradient': residual_gradients(0, 0)})


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
