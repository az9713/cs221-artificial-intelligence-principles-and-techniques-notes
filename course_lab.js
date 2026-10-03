'use strict';
// Original browser approximations of the exact Python teaching models.
// Scientific review, browser parity and public integration remain separate.
(() => {
  function integer(v,name,lo,hi) {
    if (!Number.isInteger(v)||v<lo||v>hi) throw new Error(`${name} must be an integer in [${lo},${hi}]`);
    return v;
  }
  const models = {
    learning(n) {
      const p=integer(n,'blocked_percent',0,100)/100, go=1+10*p;
      return {blocked_probability:p,go_loss:go,wait_loss:3,action:n<20?'go':'wait'};
    },
    planning(n,d) {
      const p=integer(n,'failure_percent',0,100)/100,g=integer(d,'discount_percent',0,100)/100;
      const walk=-(1+g+g*g);
      if (n===100&&d===100) return {walk_value:walk,retry_value:null,retry_expected_minutes:null,choice:'walk',boundary:'retry never terminates; undiscounted value diverges'};
      const retry=-2/(1-g*p);
      return {walk_value:walk,retry_value:retry,retry_expected_minutes:n===100?null:2/(1-p),
        choice:retry>walk?'retry':'walk',boundary:n===100?'discounted continuing return; no arrival':'terminating retry model'};
    },
    td_feedback(terminal) {
      if (typeof terminal!=='boolean') throw new Error('terminal must be bool');
      const s=-1+(terminal?0:.9*2),q=-1+(terminal?0:.9*8);
      return {sarsa_target:s,q_target:q,sarsa_new:1.5+s/2,q_new:1.5+q/2};
    },
    games(a,b) {
      const p=integer(a,'a_twelfths',0,12)/12,q=integer(b,'b_twelfths',0,12)/12;
      const one=2*p-3*(1-p),two=-3*p+4*(1-p);
      return {payoff:q*one+(1-q)*two,a_security:Math.min(one,two),
        b_upper_bound:Math.max(2*q-3*(1-q),-3*q+4*(1-q)),
        equilibrium_probability:7/12,zero_sum_value:-1/12,revealed_action_payoff:-3};
    },
    inference(n,sn,fn) {
      const p=integer(n,'prior_percent',0,100)/100,s=integer(sn,'sensitivity_percent',0,100)/100,f=integer(fn,'false_positive_percent',0,100)/100;
      const z=p*s+(1-p)*f;
      if (z===0) throw new Error('positive evidence has zero probability');
      return {evidence_probability:z,posterior:p*s/z};
    },
    disconnected_gibbs(bit) {
      integer(bit,'initial_bit',0,1);
      return {initial:[bit,bit],after_first:[bit,bit],after_second:[bit,bit],target_mass_each:.5};
    },
    logic(a,b) {
      const p=!!integer(a,'p_bit',0,1),q=!!integer(b,'q_bit',0,1);
      return {satisfies_exclusive_or:(p||q)&&!(p&&q),model_count:2,entails_p:false,entails_p_or_q:true};
    },
    finite_quantifiers() {
      let count=0,maAll=true,mbAll=true;
      for(let bits=0;bits<16;bits++) {
        const ha=!!(bits&1),hb=!!(bits&2),ma=!!(bits&4),mb=!!(bits&8);
        if((!ha||ma)&&(!hb||mb)&&ha){count++;maAll=maAll&&ma;mbAll=mbAll&&mb;}
      }
      return {premise_models:count,entails_m_a:maAll,entails_m_b:mbAll};
    },
    society(n,bp) {
      const w=integer(n,'majority_percent',0,100)/100,c=integer(bp,'ten_year_basis_points',0,1000)/10000;
      return {average_a:w*.01+(1-w)*.8,average_b:w*.1+(1-w)*.2,worst_a:.8,worst_b:.2,
        ten_year_level_factor:1+c,incorrect_annual_factor_after_ten_years:(1+c)**10};
    },
    information_value(n) {
      const cost=integer(n,'experiment_cost_tenths',0,100)/10;
      return {before_information:2,after_information_before_cost:2.5,value_of_information:.5,
        after_cost:2.5-cost,experiment_worth_cost:n<5};
    }
  };
  window.CS221Lab=Object.freeze(models);
  const labels={
    blocked_probability:'Probability of blockage',go_loss:'Expected loss of going',wait_loss:'Expected loss of waiting',action:'Action with lower loss (wait at a tie)',
    walk_value:'Discounted walking return',retry_value:'Discounted retry return',retry_expected_minutes:'Expected minutes until retry succeeds',choice:'Preferred discounted return (walk at a tie)',boundary:'Model boundary',
    payoff:'Expected payoff to A',a_security:'Worst payoff to A against a pure best reply',b_upper_bound:'Best payoff to A against B’s chosen mixture',equilibrium_probability:'First-action probability at equilibrium',zero_sum_value:'Zero-sum equilibrium payoff to A',revealed_action_payoff:'Payoff if A’s sampled action is revealed',
    evidence_probability:'Probability of a positive observation',posterior:'Probability of the cause given a positive observation',
    satisfies_exclusive_or:'Chosen assignment satisfies the knowledge base',model_count:'Number of satisfying assignments',entails_p:'Knowledge base entails P',entails_p_or_q:'Knowledge base entails P or Q',
    average_a:'Average loss of A',average_b:'Average loss of B',worst_a:'Worst subgroup loss of A',worst_b:'Worst subgroup loss of B',ten_year_level_factor:'Correct total level factor after ten years',incorrect_annual_factor_after_ten_years:'Factor from incorrectly applying that change every year',
    before_information:'Best expected payoff before information',after_information_before_cost:'Best expected payoff with information before cost',value_of_information:'Value of information before cost',after_cost:'Expected payoff with information after cost',experiment_worth_cost:'Experiment improves expected payoff after cost'
  };
  function pretty(v) {
    if(v===null)return 'undefined / no finite terminating value';
    if(typeof v==='number')return Number.isInteger(v)?String(v):v.toPrecision(8);
    if(Array.isArray(v))return '('+v.join(', ')+')';
    return String(v);
  }
function refresh(panel) {
  const output = panel.querySelector('[data-output]');
  try {
    const values = Array.from(panel.querySelectorAll('input'), input => {
      const value = input.valueAsNumber;
      if (!Number.isFinite(value)) {
        throw new Error('Enter a number in every input.');
      }
      return value;
    });
    const result = models[panel.dataset.model](...values);
    output.replaceChildren();
    for (const [name, value] of Object.entries(result)) {
      const paragraph = document.createElement('p');
      paragraph.textContent = (labels[name] || name.replaceAll('_', ' ')) +
        ': ' + pretty(value);
      output.append(paragraph);
    }
    output.removeAttribute('data-error');
  } catch (error) {
    output.textContent = error.message;
    output.setAttribute('data-error', 'true');
  }
}
  document.addEventListener('DOMContentLoaded',()=>{
    document.querySelectorAll('[data-model]').forEach(panel=>{
      panel.querySelectorAll('input').forEach(input=>input.addEventListener('input',()=>refresh(panel)));
      refresh(panel);
    });
  });
})();
