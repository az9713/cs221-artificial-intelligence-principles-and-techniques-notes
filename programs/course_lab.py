"""Original cumulative CS221 lab models; exact rational teaching inputs.

Python3.10+, standard library; no network, file writes or input mutation.
This candidate awaits independent scientific review and reader integration.
"""
from fractions import Fraction as F
from itertools import product


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be a built-in integer in [{low},{high}]')
    return value


def learning(blocked_percent):
    """Exact expected-loss contraction; percentage input0..100, tie chooses wait."""
    p=F(integer(blocked_percent,'blocked_percent',0,100),100)
    go=1+10*p;wait=F(3)
    return {'blocked_probability':p,'go_loss':go,'wait_loss':wait,
            'action':'go' if go<wait else 'wait'}


def planning(failure_percent,discount_percent):
    """Compare repeated two-minute attempts to three one-minute walking steps.

    Retry failure returns to the same state; success terminates. Rewards count
    time with a negative sign. Discounted value is an objective, not expected
    elapsed time unless discount=1 and termination is integrable.
    Discount applies once per decision transition, not per minute; a retry
    is one two-minute action and walking is three one-minute actions.
    """
    p=F(integer(failure_percent,'failure_percent',0,100),100)
    g=F(integer(discount_percent,'discount_percent',0,100),100)
    walk=-(1+g+g*g)
    if p==g==1:
        return {'walk_value':walk,'retry_value':None,'retry_expected_minutes':None,
                'choice':'walk','boundary':'retry never terminates; undiscounted value diverges'}
    retry=-F(2)/(1-g*p)
    return {'walk_value':walk,'retry_value':retry,
            'retry_expected_minutes':None if p==1 else F(2)/(1-p),
            'choice':'retry' if retry>walk else 'walk',
            'boundary':'discounted continuing return; no arrival' if p==1 else 'terminating retry model'}


def td_feedback(terminal):
    """One fixed sampled transition: old3,reward-1,gamma9/10,alpha1/2.

    Nonterminal next-action values2/8; SARSA uses the already chosen low action.
    Terminal continuation is zero even if a stored next value is stale.
    """
    if type(terminal) is not bool:raise ValueError('terminal must be bool')
    sarsa=-F(1)+(0 if terminal else F(9,10)*2)
    q=-F(1)+(0 if terminal else F(9,10)*8)
    return {'sarsa_target':sarsa,'q_target':q,'sarsa_new':F(3,2)+sarsa/2,
            'q_new':F(3,2)+q/2}


def games(a_twelfths,b_twelfths):
    """Independent hidden draws in one-shot zero-sum Morra; A maximizes.

    Probability of first action is each slider integer divided by12. Matrix
    rowA,columnB is[[2,-3],[-3,4]]. Revealed actions have different information.
    """
    p=F(integer(a_twelfths,'a_twelfths',0,12),12)
    q=F(integer(b_twelfths,'b_twelfths',0,12),12)
    against_first=2*p-3*(1-p)
    against_second=-3*p+4*(1-p)
    first_against_b=2*q-3*(1-q)
    second_against_b=-3*q+4*(1-q)
    return {'payoff':q*against_first+(1-q)*against_second,
            'a_security':min(against_first,against_second),
            'b_upper_bound':max(first_against_b,second_against_b),
            'equilibrium_probability':F(7,12),'zero_sum_value':-F(1,12),
            'revealed_action_payoff':-F(3)}


def inference(prior_percent,sensitivity_percent,false_positive_percent):
    """Binary cause and positive test; finite stipulated conditional table.

    This is a probability example, not a clinical diagnostic recommendation.
    Zero-probability evidence raises ValueError instead of inventing a posterior.
    """
    p=F(integer(prior_percent,'prior_percent',0,100),100)
    s=F(integer(sensitivity_percent,'sensitivity_percent',0,100),100)
    f=F(integer(false_positive_percent,'false_positive_percent',0,100),100)
    evidence=p*s+(1-p)*f
    if not evidence:raise ValueError('positive evidence has zero probability')
    return {'evidence_probability':evidence,'posterior':p*s/evidence}


def disconnected_gibbs(initial_bit):
    """Joint support00/11 each1/2; single-coordinate Gibbs cannot switch.

    Return the state after either coordinate update. This two-point example
    exposes lack of irreducibility; it is not a general sampler implementation.
    """
    bit=integer(initial_bit,'initial_bit',0,1)
    return {'initial':(bit,bit),'after_first':(bit,bit),'after_second':(bit,bit),
            'target_mass_each':F(1,2)}


def logic(p_bit,q_bit):
    """Truth evaluation versus entailment in a finite propositional model."""
    p=bool(integer(p_bit,'p_bit',0,1));q=bool(integer(q_bit,'q_bit',0,1))
    models=[(a,b) for a,b in product([False,True],repeat=2) if (a or b) and not(a and b)]
    return {'satisfies_exclusive_or':(p or q) and not(p and q),'model_count':len(models),
            'entails_p':all(a for a,b in models),'entails_p_or_q':all(a or b for a,b in models)}


def finite_quantifiers():
    """Enumerate interpretations of two predicates on a two-object universe.

    Premises∀x(H(x)→M(x)) and H(a) entail M(a) in this finite universe.
    This computation is not a completeness proof for unrestricted first-order
    logic, and missing entries do not establish false facts in an open world.
    """
    models=[(ha,hb,ma,mb) for ha,hb,ma,mb in product([False,True],repeat=4)
            if (not ha or ma) and (not hb or mb) and ha]
    return {'premise_models':len(models),'entails_m_a':all(m[2] for m in models),
            'entails_m_b':all(m[3] for m in models)}


def society(majority_percent,ten_year_basis_points):
    """Two subgroup-loss vectors and a stipulated ten-year level change.

    A:[1%,80%], B:[10%,20%]. Percentages are chosen teaching assumptions,
    not observed performance or population shares. Growth input0..1000basis
    points means0..10% total over ten years; no forecast is asserted.
    """
    w=F(integer(majority_percent,'majority_percent',0,100),100)
    change=F(integer(ten_year_basis_points,'ten_year_basis_points',0,1000),10000)
    return {'average_a':w*F(1,100)+(1-w)*F(4,5),
            'average_b':w*F(1,10)+(1-w)*F(1,5),
            'worst_a':F(4,5),'worst_b':F(1,5),
            'ten_year_level_factor':1+change,
            'incorrect_annual_factor_after_ten_years':(1+change)**10}


def information_value(experiment_cost_tenths):
    """Perfect-information value in a stipulated two-action/two-world decision.

    Safe payoff2; risky payoff0 or3 with equal probability. Original actions
    remain available after learning. Cost0..10 in tenths; tie declines experiment.
    Human careers need their own preferences, obligations and opportunities.
    """
    cost=F(integer(experiment_cost_tenths,'experiment_cost_tenths',0,100),10)
    prior=F(2);informed=F(5,2)
    return {'before_information':prior,'after_information_before_cost':informed,
            'value_of_information':informed-prior,'after_cost':informed-cost,
            'experiment_worth_cost':informed-cost>prior}
