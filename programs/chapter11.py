"""Exact finite TD and two-action game fixtures; Python standard library only.

Public numbers: built-in int or Fraction, excluding bool and subclasses;
numerator/denominator at most 128 bits. Inputs never mutated. Internal rational
arithmetic may grow. There is no training, randomness, network or file access.
"""
from fractions import Fraction


def rational(value):
    if type(value) not in (int, Fraction):
        raise ValueError('expected built-in int or Fraction, excluding bool')
    result = Fraction(value)
    size = max(abs(result.numerator).bit_length(),
               result.denominator.bit_length())
    if size > 128:
        raise ValueError(
            'public rational exceeds 128-bit numerator/denominator')
    return result


def probability(value):
    result = rational(value)
    if not 0 <= result <= 1:
        raise ValueError('probability or step must lie in [0,1]')
    return result


def vector(value):
    if type(value) not in (list, tuple) or not 1 <= len(value) <= 16:
        raise ValueError('vector must be list/tuple of length 1..16')
    return tuple(rational(x) for x in value)


def matrix(value):
    if type(value) not in (list, tuple) or len(value) != 2:
        raise ValueError('matrix must have exactly two rows')
    rows = tuple(vector(row) for row in value)
    if any(len(row) != 2 for row in rows):
        raise ValueError('matrix must be 2x2')
    return rows


def td_step(weights, features, next_features, reward, discount, step,
            terminal=False, residual=False):
    """Return(new_weights,target,delta) from one fixed recorded transition.

    terminal=True suppresses bootstrap. residual=True differentiates the
    whole sampled half-squared residual instead of freezing its target.
    All arguments, including unused terminal next_features, are validated.
    """
    w, phi, nxt = vector(weights), vector(features), vector(next_features)
    if not len(w) == len(phi) == len(nxt):
        raise ValueError('weights and feature vectors must have equal lengths')
    r = rational(reward)
    gamma = probability(discount)
    alpha = probability(step)
    if type(terminal) is not bool or type(residual) is not bool:
        raise ValueError('terminal and residual must be bool')
    current = sum(a*b for a, b in zip(w, phi))
    target = r if terminal else r + gamma*sum(a*b for a, b in zip(w, nxt))
    delta = target-current
    direction = tuple(a-gamma*b if residual and not terminal else a
                      for a, b in zip(phi, nxt))
    return tuple(a+alpha*delta*b for a, b in zip(w, direction)), target, delta


def _expect(m, p, q):
    return (p*q*m[0][0] + p*(1-q)*m[0][1]
            + (1-p)*q*m[1][0] + (1-p)*(1-q)*m[1][1])


def expected(payoffs, row_probability, column_probability):
    """A's payoff expectation for independent draws; pure endpoints allowed."""
    return _expect(matrix(payoffs), probability(row_probability),
                   probability(column_probability))


def joint_expected(payoffs, joint):
    """Expectation under an explicitly supplied joint law; not product play."""
    m, law = matrix(payoffs), matrix(joint)
    if any(x < 0 for row in law for x in row) or sum(map(sum, law)) != 1:
        raise ValueError(
            'joint probabilities must be nonnegative and sum to 1')
    return sum(m[i][j]*law[i][j] for i in range(2) for j in range(2))


def zero_sum(payoffs):
    """Return(p,q,value) from endpoints/intersections of response envelopes.

    Row player maximizes, column player minimizes the same matrix. Ties use
    the smallest probability. A two-action solver, not a general Nash solver.
    """
    m = matrix(payoffs)
    (a, b), (c, d) = m
    denominator = a-b-c+d
    ps, qs = {Fraction(0), Fraction(1)}, {Fraction(0), Fraction(1)}
    if denominator:
        p, q = (d-c)/denominator, (d-b)/denominator
        if 0 <= p <= 1:
            ps.add(p)
        if 0 <= q <= 1:
            qs.add(q)

    def lower(p):
        return min(c+(a-c)*p, d+(b-d)*p)

    def upper(q):
        return max(b+(a-b)*q, d+(c-d)*q)

    p = max(sorted(ps), key=lower)
    q = min(sorted(qs), key=upper)
    value = lower(p)
    assert value == upper(q), 'finite rational primal/dual disagreement'
    return p, q, value


def regrets(row_payoffs, column_payoffs, row_probability, column_probability):
    """Max gain from unilateral deviation for each player; both maximize."""
    a, b = matrix(row_payoffs), matrix(column_payoffs)
    p, q = probability(row_probability), probability(column_probability)
    row_value, col_value = _expect(a, p, q), _expect(b, p, q)
    row_best = max(q*a[i][0]+(1-q)*a[i][1] for i in range(2))
    col_best = max(p*b[0][j]+(1-p)*b[1][j] for j in range(2))
    return row_best-row_value, col_best-col_value


def pure_equilibria(row_payoffs, column_payoffs):
    """Return only pure Nash profiles as(row_index,column_index);0-based."""
    a, b = matrix(row_payoffs), matrix(column_payoffs)
    return tuple((i, j) for i in range(2) for j in range(2)
                 if a[i][j] == max(a[k][j] for k in range(2))
                 and b[i][j] == max(b[i][k] for k in range(2)))


def main():
    f = Fraction
    semi = td_step([2], [1], [3], 0, f(1, 2), f(1, 10))
    full = td_step([2], [1], [3], 0, f(1, 2), f(1, 10), residual=True)
    terminal = td_step([10], [1], [1], 2, f(9, 10), f(1, 10), terminal=True)
    print(f'TD target/delta: {semi[1]} / {semi[2]}')
    print(f'semi-gradient/full-residual weights: {semi[0][0]} / {full[0][0]}')
    unguarded = 2+f(9, 10)*10
    print(f'terminal target / unguarded target: {terminal[1]} / {unguarded}')
    m = [[2, -3], [-3, 4]]
    p, q, value = zero_sum(m)
    print(f'Morra row/column probabilities: {p} / {q}')
    print(f'Morra security value: {value}')
    uniform = expected(m, f(1, 2), f(1, 2))
    diagonal = joint_expected(m, [[f(1, 2), 0], [0, f(1, 2)]])
    print(f'uniform independent / diagonal joint: {uniform} / {diagonal}')
    revealed = p*min(m[0])+(1-p)*min(m[1])
    print(f'revealed-realization payoff: {revealed}')
    # Chance is independent, observed before choice in the first calculation.
    observed = f(1, 2)*max(2, 0)+f(1, 2)*max(0, 2)
    committed = max(f(1, 2)*2+f(1, 2)*0, f(1, 2)*0+f(1, 2)*2)
    print(f'observed-chance / precommitted action: {observed} / {committed}')
    a, b = [[-5, 0], [-10, -1]], [[-5, -10], [0, -1]]
    print(f'prisoner pure Nash profiles: {pure_equilibria(a, b)}')
    ra, rb = regrets(a, b, 0, 0)
    print(f'prisoner mutual-refusal regrets: {ra} / {rb}')
    print(f'cooperative pure Nash profiles: {pure_equilibria(m, m)}')
    ra, rb = regrets(m, m, p, q)
    cooperative = expected(m, p, q)
    print(f'cooperative interior regrets/payoff: {ra} / {rb} / {cooperative}')


if __name__ == '__main__':
    main()
