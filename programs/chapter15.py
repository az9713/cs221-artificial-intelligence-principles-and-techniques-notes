"""Original finite propositional reasoning: exact worlds and explicit limits.

Python 3.10+, standard library. Library calls have no external effects. The
command-line example writes stdout only. Bounds: eight atoms, 64 KB formulas,
128 expanded nodes per formula, depth 32, and 200000 MP rule checks.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import re


def name(value):
    if (
        type(value) is not str
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", value) is None
    ):
        raise ValueError(
            "Atom names must be ASCII identifiers of length 1..32"
        )
    return value


@dataclass(frozen=True)
class Formula:
    op: str
    args: tuple

    def __post_init__(self):
        validate_formula(self)


def validate_formula(formula):
    (
        'Validate the complete expanded tree, including shared-s'
        'ubtree budgets.'
    )
    count = 0
    atoms = set()

    def visit(node, depth):
        nonlocal count
        count += 1
        if count > 128 or depth > 32:
            raise ValueError("Formula exceeds node/depth budget")
        if (
            type(node) is not Formula
            or type(node.op) is not str
            or type(node.args) is not tuple
        ):
            raise ValueError("Use an exact Formula with a tuple of operands")
        if node.op == "atom" and len(node.args) == 1:
            atoms.add(name(node.args[0]))
        elif (
            node.op == "constant"
            and len(node.args) == 1
            and type(node.args[0]) is bool
        ):
            pass
        elif node.op == "not" and len(node.args) == 1:
            visit(node.args[0], depth + 1)
        elif (
            node.op in ("and", "or", "implies", "iff") and len(node.args) == 2
        ):
            for child in node.args:
                visit(child, depth + 1)
        else:
            raise ValueError("Unsupported operator or operand arity")

    visit(formula, 1)
    return frozenset(atoms)


def atom(identifier):
    return Formula("atom", (identifier,))


def neg(formula):
    return Formula("not", (formula,))


def binary(op, left, right):
    if op not in ("and", "or", "implies", "iff"):
        raise ValueError("Unsupported binary operator")
    return Formula(op, (left, right))


TRUE = Formula("constant", (True,))
FALSE = Formula("constant", (False,))


def vocabulary(symbols):
    if type(symbols) is not tuple or len(symbols) > 8:
        raise ValueError("Vocabulary must be a tuple of at most eight atoms")
    for symbol in symbols:
        name(symbol)
    if len(set(symbols)) != len(symbols):
        raise ValueError("Vocabulary names must be unique")
    return symbols


def _interpret(formula, world):
    op, args = formula.op, formula.args
    if op == "atom":
        return world[args[0]]
    if op == "constant":
        return args[0]
    if op == "not":
        return not _interpret(args[0], world)
    left, right = (_interpret(child, world) for child in args)
    if op == "and":
        return left and right
    if op == "or":
        return left or right
    if op == "implies":
        return not left or right
    return left == right


def interpret(formula, symbols, world):
    """Require a total, exact-Boolean world over the supplied vocabulary."""
    vocabulary(symbols)
    if not validate_formula(formula) <= set(symbols):
        raise ValueError("Formula uses an undeclared atom")
    if type(world) is not dict or set(world) != set(symbols):
        raise ValueError(
            "World must be a dict with exactly the vocabulary keys"
        )
    if any(type(value) is not bool for value in world.values()):
        raise ValueError("World values must be exact booleans")
    return _interpret(formula, world)


@dataclass(frozen=True)
class Knowledge:
    symbols: tuple
    formulas: tuple = ()

    def __post_init__(self):
        vocabulary(self.symbols)
        if type(self.formulas) is not tuple or len(self.formulas) > 64:
            raise ValueError(
                "Knowledge must contain a tuple of at most 64 formulas"
            )
        for formula in self.formulas:
            if not validate_formula(formula) <= set(self.symbols):
                raise ValueError("Knowledge uses an undeclared atom")


def validate_query(kb, query):
    if type(kb) is not Knowledge:
        raise ValueError("Use an exact Knowledge instance")
    kb.__post_init__()
    if not validate_formula(query) <= set(kb.symbols):
        raise ValueError("Query uses an undeclared atom")


def worlds(symbols):
    (
        'Immutable tuples in False-before-True product order; ze'
        'ro atoms has one world.'
    )
    vocabulary(symbols)
    return tuple(product((False, True), repeat=len(symbols)))


def models(kb):
    validate_query(kb, TRUE)
    return tuple(
        values
        for values in worlds(kb.symbols)
        if all(
            _interpret(f, dict(zip(kb.symbols, values))) for f in kb.formulas
        )
    )


@dataclass(frozen=True)
class Answer:
    status: str
    positive: tuple | None
    negative: tuple | None
    model_count: int


def ask(kb, query):
    (
        'Return a logical status and first counter/support witne'
        'sses; never mutate KB.'
    )
    validate_query(kb, query)
    positive = negative = None
    admissible = models(kb)
    for values in admissible:
        value = _interpret(query, dict(zip(kb.symbols, values)))
        if value and positive is None:
            positive = values
        if not value and negative is None:
            negative = values
    if not admissible:
        status = "inconsistent"
    elif negative is None:
        status = "entailed"
    elif positive is None:
        status = "contradicted"
    else:
        status = "contingent"
    return Answer(status, positive, negative, len(admissible))


def tell(kb, formula):
    """Return a new immutable KB only for consistent informative additions."""
    answer = ask(kb, formula)
    if answer.status == "inconsistent":
        raise ValueError("Tell requires a consistent current knowledge base")
    if answer.status == "entailed":
        return kb, "already-known"
    if answer.status == "contradicted":
        return kb, "rejected"
    return Knowledge(kb.symbols, kb.formulas + (formula,)), "learned"


def classify_sat_answers(positive, negative):
    (
        'Classify two genuine SAT decisions; unknown is computat'
        'ional, not contingent.\n\n    Positive refers to KB and q'
        'uery; negative to KB and negated query. This\n    helper'
        " does not run a solver or validate a claimed solver's c"
        'ertificates.\n    '
    )
    allowed = ("sat", "unsat", "unknown")
    if (
        type(positive) is not str
        or type(negative) is not str
        or positive not in allowed
        or negative not in allowed
    ):
        raise ValueError("SAT responses must be sat, unsat or unknown")
    if "unknown" in (positive, negative):
        return "solver-unknown"
    return {
        ("sat", "sat"): "contingent",
        ("sat", "unsat"): "entailed",
        ("unsat", "sat"): "contradicted",
        ("unsat", "unsat"): "inconsistent",
    }[(positive, negative)]


def condition(kb, query, masses):
    (
        'Exact formula-event posterior and evidence mass in the '
        'worlds() order.'
    )
    validate_query(kb, query)
    assignments = worlds(kb.symbols)
    if type(masses) is not tuple or len(masses) != len(assignments):
        raise ValueError("Mass tuple must have one entry per ordered world")
    converted = []
    for value in masses:
        if type(value) not in (int, Fraction):
            raise ValueError("Use built-in integers or exact Fraction masses")
        value = Fraction(value)
        if (
            value < 0
            or value.numerator.bit_length() > 128
            or value.denominator.bit_length() > 128
        ):
            raise ValueError(
                "Mass must be nonnegative and within 128-bit fraction bounds"
            )
        converted.append(value)
    if sum(converted) != 1:
        raise ValueError("Masses must sum exactly to one")
    denominator = numerator = Fraction(0)
    for values, mass in zip(assignments, converted):
        world = dict(zip(kb.symbols, values))
        if all(_interpret(f, world) for f in kb.formulas):
            denominator += mass
            if _interpret(query, world):
                numerator += mass
    if denominator == 0:
        raise ValueError("Evidence has zero probability; no ratio conditional")
    return numerator / denominator, denominator


def mp_closure(kb):
    (
        'Saturate syntactically present modus-ponens premises; n'
        'ot a complete calculus.\n\n    Conclusions stay in the in'
        "itial formulas' finite subformula universe.\n    The ret"
        'urned KB retains the 64-formula bound; overflow raises '
        'ValueError\n    without modifying the input. At most 200'
        '000 active rule checks are made.\n    '
    )
    validate_query(kb, TRUE)
    known = list(dict.fromkeys(kb.formulas))
    seen = set(known)
    trace = []
    checks = 0
    changed = True
    while changed:
        changed = False
        for rule in tuple(known):
            checks += 1
            if checks > 200000:
                raise ValueError("MP rule-check budget exceeded")
            if (
                rule.op == "implies"
                and rule.args[0] in seen
                and rule.args[1] not in seen
            ):
                if len(known) == 64:
                    raise ValueError(
                        "MP closure exceeds the 64-formula output budget"
                    )
                conclusion = rule.args[1]
                known.append(conclusion)
                seen.add(conclusion)
                trace.append((rule.args[0], rule, conclusion))
                changed = True
    return Knowledge(kb.symbols, tuple(known)), tuple(trace)


def horn_closure(symbols, facts, rules):
    """Saturate finite positive definite-Horn rules, including empty bodies.

    Each rule is (antecedent-name tuple, consequent name). At most eight atoms
    and 64 rules are allowed. This rule matcher checks all body atoms; it is
    distinct from the general formula MP-only implementation above.
    """
    vocabulary(symbols)
    if type(facts) is not tuple or len(facts) > len(symbols):
        raise ValueError("Facts must be a tuple within the vocabulary size")
    for fact in facts:
        name(fact)
    if len(set(facts)) != len(facts) or not set(facts) <= set(symbols):
        raise ValueError("Facts must be unique declared atoms")
    if type(rules) is not tuple or len(rules) > 64:
        raise ValueError("Use a tuple of at most 64 Horn rules")
    for rule in rules:
        if type(rule) is not tuple or len(rule) != 2:
            raise ValueError("Horn rules are (body tuple, head name)")
        body, head = rule
        if type(body) is not tuple or len(body) > len(symbols):
            raise ValueError("Horn body must be a bounded tuple of names")
        name(head)
        for identifier in body:
            name(identifier)
        if len(set(body)) != len(body) or not (set(body) | {head}) <= set(
            symbols
        ):
            raise ValueError("Horn body/head must use unique declared atoms")
    known = set(facts)
    trace = []
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            if set(body) <= known and head not in known:
                known.add(head)
                trace.append((body, head))
                changed = True
    return tuple(
        identifier for identifier in symbols if identifier in known
    ), tuple(trace)


def main():
    t, c, o = map(atom, ("T", "C", "O"))
    rules = (binary("implies", t, c), binary("implies", c, o))
    kb = Knowledge(("T", "C", "O"), (t,) + rules)
    print("permission:", ask(kb, o).status, "models:", models(kb))
    unchanged, outcome = tell(kb, neg(c))
    print("tell not C:", outcome, "unchanged:", unchanged is kb)
    open_only = Knowledge(kb.symbols, (o,) + rules)
    answer = ask(open_only, t)
    print(
        "observed permission:",
        answer.status,
        "positive:",
        answer.positive,
        "negative:",
        answer.negative,
    )
    closure, trace = mp_closure(kb)
    shortcut = binary("implies", t, o)
    print(
        "MP added:",
        len(trace),
        "shortcut entailed:",
        ask(kb, shortcut).status,
        "shortcut generated:",
        shortcut in closure.formulas,
    )
    horn, horn_trace = horn_closure(
        kb.symbols, ("T",), ((("T",), "C"), (("T", "C"), "O"))
    )
    print("Horn closure:", horn, "new atoms:", len(horn_trace))
    r, w = map(atom, ("Rain", "Wet"))
    weather = Knowledge(("Rain", "Wet"), (w,))
    masses = (
        Fraction(1, 2),
        Fraction(1, 10),
        Fraction(1, 10),
        Fraction(3, 10),
    )
    posterior, evidence = condition(weather, r, masses)
    print("rain given wet:", posterior, "evidence:", evidence)
    empty = Knowledge(("Rain", "Wet"))
    event = binary("and", neg(r), neg(w))
    p, _ = condition(empty, event, (1, 0, 0, 0))
    print("zero-support gap:", p, "logical status:", ask(empty, event).status)
    inconsistent = Knowledge(("Rain", "Wet"), (r, neg(r)))
    print("inconsistent:", ask(inconsistent, w).status)
    print("solver response:", classify_sat_answers("unknown", "sat"))
    print(
        "Finite witnesses certify stated assumptions; "
        "they do not verify facts about the world."
    )


if __name__ == "__main__":
    main()
