"""Exact finite supply-chain and productivity teaching models, not forecasts.

Public scalar inputs are int/Fraction (never bool/float), each numerator and
denominator at most128bits. Vectors are nonempty list/tuple, at most16 entries.
All effects are local calculations; inputs are preserved. No I/O except demo.
Invalid types raise TypeError; domain/shape/budget failures raise ValueError.
Caller supplies compatible units, quality constraints and economic assumptions.
"""
from fractions import Fraction as F


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, F)):
        raise TypeError('Use exact int or Fraction inputs.')
    value = F(value)
    bits = max(abs(value.numerator).bit_length(),
               value.denominator.bit_length())
    if bits > 128:
        raise ValueError('Exact scalar input exceeds128bits.')
    return value


def _vector(values, lower=None, upper=None):
    if not isinstance(values, (list, tuple)):
        raise TypeError('Use a list or tuple.')
    if not 1 <= len(values) <= 16:
        raise ValueError('Vector length must be1..16.')
    result = tuple(_number(value) for value in values)
    if lower is not None and any(value < lower for value in result):
        raise ValueError('Value below permitted domain.')
    if upper is not None and any(value > upper for value in result):
        raise ValueError('Value above permitted domain.')
    return result


def serial_capacity(capacities):
    """Serial chain without loss/inventory, rates in service units/time."""
    rates = _vector(capacities, 0)
    rate = min(rates)
    return {'throughput': rate,
            'bottlenecks': tuple(i for i, value in enumerate(rates)
                                 if value == rate)}


def serving_comparison(fixed_a, variable_a, fixed_b, variable_b, volume):
    """Same-period costs and matched service quality supplied by caller.

    crossover_volume is the real algebraic equality point, possibly negative;
    None denotes parallel cost lines. Feasible volume/quality is not inferred.
    """
    inputs = (fixed_a, variable_a, fixed_b, variable_b, volume)
    fa, va, fb, vb, n = tuple(_number(x) for x in inputs)
    if min(fa, va, fb, vb, n) < 0:
        raise ValueError('Costs and volume must be nonnegative.')
    a, b = fa + va*n, fb + vb*n
    return {'cost_a': a, 'cost_b': b, 'difference_a_minus_b': a-b,
            'crossover_volume': (fb-fa)/(va-vb) if va != vb else None}


def task_cost_savings(shares, exposure, adoption, savings):
    """Fixed cost partition; per-task exposure/adoption/savings in[0,1].

    Savings are net reductions on adopted exposed activity. This accounting
    excludes price adjustment, new tasks and general-equilibrium effects.
    """
    vectors = [_vector(x, 0, 1) for x in (shares, exposure, adoption, savings)]
    if len({len(x) for x in vectors}) != 1 or sum(vectors[0]) != 1:
        raise ValueError(
            'Equal vector lengths and normalized cost shares required.')
    contributions = tuple(w*e*a*s for w, e, a, s in zip(*vectors))
    return {'contributions': contributions, 'total': sum(contributions, F(0))}


def chain_value_added(sales, purchased_inputs):
    """Caller supplies complete same-period sales/intermediate-input accounts.

    Returns each value added and gross-sales/total-value-added ratios, not a
    certified GDP series or network productivity theorem. Negative value added
    is excluded in this model; total must be strictly positive.
    """
    sales = _vector(sales, 0)
    purchases = _vector(purchased_inputs, 0)
    if (len(sales) != len(purchases)
            or any(i > s for s, i in zip(sales, purchases))):
        raise ValueError(
            'Matched vectors and nonnegative value added required.')
    values = tuple(s-i for s, i in zip(sales, purchases))
    total = sum(values, F(0))
    if total <= 0:
        raise ValueError('Positive total value added required.')
    return {'value_added': values, 'total': total,
            'value_added_shares': tuple(v/total for v in values),
            'gross_output_ratios': tuple(s/total for s in sales)}


def budget_allocation(alpha, budget, capital_price, labor_price):
    """Interior optimum of A*K^alpha*L^(1-alpha), A>0, linear input budget.

    All prices and budget positive; alpha in(0,1). Fractional inputs and no
    minimum or capacity constraints. Output is the exact allocation.
    """
    inputs = (alpha, budget, capital_price, labor_price)
    a, b, pk, pl = tuple(_number(x) for x in inputs)
    if not 0 < a < 1 or min(b, pk, pl) <= 0:
        raise ValueError('Interior share and positive budget/prices required.')
    return {'capital': a*b/pk, 'labor': (1-a)*b/pl,
            'capital_spending': a*b, 'labor_spending': (1-a)*b}


def production_power(capital, labor, numerator, denominator):
    """Exact (Y/A)^q=K^p*L^(q-p), 1<=p<q<=16; no root rounding."""
    k, labor_value = _number(capital), _number(labor)
    if any(isinstance(x, bool) or not isinstance(x, int)
           for x in (numerator, denominator)):
        raise TypeError('Exponent integers required.')
    if min(k, labor_value) <= 0 or not 1 <= numerator < denominator <= 16:
        raise ValueError('Positive inputs and1<=p<q<=16 required.')
    return k**numerator*labor_value**(denominator-numerator)


def compound_multiplier(rate, periods):
    """Constant per-period fractional rate, rate>-1, periods integer0..1000."""
    rate = _number(rate)
    if isinstance(periods, bool) or not isinstance(periods, int):
        raise TypeError('Integer period count required.')
    if rate <= -1 or not 0 <= periods <= 1000:
        raise ValueError('Rate>-1 and periods0..1000 required.')
    return (1+rate)**periods


def measurement_wedge(unmeasured_investment, uncredited_services):
    """Supplied same-unit additive illustration services-investment per period.

    This teaching wedge differs from the published growth-accounting formula.
    """
    investments = _vector(unmeasured_investment, 0)
    services = _vector(uncredited_services, 0)
    if len(investments) != len(services):
        raise ValueError('Same period counts required.')
    return tuple(s-i for i, s in zip(investments, services))


def demo():
    print('serial throughput:', serial_capacity([50, 80, 100])['throughput'])
    costs = serving_comparison(100, F(1, 100), 0, F(3, 100), 10000)
    print('serving costs:', costs['cost_a'], costs['cost_b'])
    print('crossover volume:', costs['crossover_volume'])
    tasks = task_cost_savings([1], [F(1, 5)], [F(23, 100)], [F(18, 125)])
    print('linear task savings:', tasks['total'])
    accounts = chain_value_added([40, 100], [0, 40])
    print('value added:', accounts['total'])
    print('gross ratios sum:', sum(accounts['gross_output_ratios']))
    allocation = budget_allocation(F(1, 4), 4, 1, 1)
    print('optimal K,L:', allocation['capital'], allocation['labor'])
    print('unequal/equal fourth powers:',
          production_power(1, 3, 1, 4), production_power(2, 2, 1, 4))
    print('measurement wedge:', measurement_wedge([3, 2, 1, 1], [0, 1, 2, 3]))
    print('Supplied finite accounting; no empirical or economic forecast.')


if __name__ == '__main__':
    demo()
