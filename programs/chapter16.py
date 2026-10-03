"""Original bounded finite first-order semantics and positive Horn laboratory.

ASTs are tuples: ('v', name), ('c', name), ('f', name, args),
('p', name, args), ('eq', left, right), ('not', formula),
('and'/'or'/'implies', left, right), ('all'/'exists', name, body).
All public computations validate complete inputs, return new values, and
perform no I/O. Only main prints. Bounds are resource contracts, not logic.
"""
from itertools import product
import re


def _name(value):
    if type(value) is not str or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]{0,23}", value):
        raise ValueError("Names must be ASCII identifiers of length 1..24.")
    return value


def _integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Integer outside its declared range.")
    return value


def _sequence(value, limit):
    if type(value) is not tuple or len(value) > limit:
        raise ValueError("Expected a tuple within its length bound.")
    return value


def _dictionary(value, limit):
    if type(value) is not dict or len(value) > limit:
        raise ValueError("Expected a plain dictionary within its size bound.")
    return value


def make_structure(size, constants, functions, predicates):
    """Return an immutable canonical structure on objects 0..size-1.

    constants: name -> object. functions: name -> (arity, total table).
    predicates: name -> (arity, set/frozenset of true object tuples).
    size 1..8, at most 12 symbols per category, function arity 1..3,
    predicate arity 0..3. Every omitted predicate tuple is false.
    """
    _integer(size, 1, 8)
    for mapping in (constants, functions, predicates):
        _dictionary(mapping, 12)
        for name in mapping:
            _name(name)
    if (set(constants) & set(functions) or set(constants) & set(predicates)
            or set(functions) & set(predicates)):
        raise ValueError("Symbol categories must be disjoint.")
    for obj in constants.values():
        _integer(obj, 0, size - 1)
    function_rows = []
    for name, record in sorted(functions.items()):
        _sequence(record, 2)
        if len(record) != 2:
            raise ValueError("Function record needs arity and table.")
        arity, table = record
        _integer(arity, 1, 3)
        _dictionary(table, 512)
        for key, obj in table.items():
            _sequence(key, 3)
            if len(key) != arity:
                raise ValueError("Function input arity mismatch.")
            for value in key:
                _integer(value, 0, size - 1)
            _integer(obj, 0, size - 1)
        if set(table) != set(product(range(size), repeat=arity)):
            raise ValueError("Function table is not total.")
        function_rows.append((name, arity, tuple(sorted(table.items()))))
    predicate_rows = []
    for name, record in sorted(predicates.items()):
        _sequence(record, 2)
        if len(record) != 2:
            raise ValueError("Predicate record needs arity and extension.")
        arity, extension = record
        _integer(arity, 0, 3)
        if type(extension) not in (set, frozenset) or len(extension) > 512:
            raise ValueError("Expected a bounded set of object tuples.")
        for key in extension:
            _sequence(key, 3)
            if len(key) != arity:
                raise ValueError("Predicate tuple arity mismatch.")
            for value in key:
                _integer(value, 0, size - 1)
        predicate_rows.append((name, arity, frozenset(extension)))
    return (size, tuple(sorted(constants.items())),
            tuple(function_rows), tuple(predicate_rows))


def _pairs(rows):
    _sequence(rows, 512)
    result = {}
    for row in rows:
        _sequence(row, 2)
        if len(row) != 2:
            raise ValueError("Malformed canonical table row.")
        key, value = row
        if type(key) is str:
            _name(key)
        elif type(key) is tuple:
            for obj in key:
                _integer(obj, 0, 7)
        else:
            raise ValueError("Malformed canonical table key.")
        if key in result:
            raise ValueError("Duplicate canonical table key.")
        result[key] = value
    return result


def _tables(world):
    _sequence(world, 4)
    if len(world) != 4:
        raise ValueError("Malformed structure.")
    size, constants, functions, predicates = world
    constants = _pairs(constants)
    function_map, predicate_map = {}, {}
    for rows, target, function in (
            (functions, function_map, True),
            (predicates, predicate_map, False)):
        _sequence(rows, 12)
        for row in rows:
            _sequence(row, 3)
            if len(row) != 3:
                raise ValueError("Malformed symbol row.")
            name, arity, table = row
            _name(name)
            if name in target:
                raise ValueError("Duplicate symbol row.")
            if not function and type(table) is not frozenset:
                raise ValueError("Canonical predicate extension is immutable.")
            target[name] = (arity, _pairs(table) if function else table)
    canonical = make_structure(size, constants, function_map, predicate_map)
    if canonical != world:
        raise ValueError("Structure must have the factory's canonical form.")
    return size, constants, function_map, predicate_map


def _shape(node, category="formula", count=None, depth=0):
    count = [0] if count is None else count
    count[0] += 1
    if count[0] > 512 or depth > 16:
        raise ValueError("AST exceeds 512 nodes or depth 16.")
    if type(node) is not tuple or not node or type(node[0]) is not str:
        raise ValueError("Malformed AST node.")
    tag = node[0]
    if tag in ("v", "c") and category in ("term", "expression"):
        if len(node) != 2:
            raise ValueError("Variable/constant node needs a name.")
        _name(node[1])
    elif ((tag == "f" and category in ("term", "expression"))
          or (tag == "p" and category in ("formula", "expression"))):
        if len(node) != 3:
            raise ValueError("Application needs name and argument tuple.")
        _name(node[1])
        _sequence(node[2], 3)
        if tag == "f" and not node[2]:
            raise ValueError("Functions have positive arity; use a constant.")
        for child in node[2]:
            _shape(child, "term", count, depth + 1)
    elif category == "formula" and tag == "eq" and len(node) == 3:
        for child in node[1:]:
            _shape(child, "term", count, depth + 1)
    elif category == "formula" and tag == "not" and len(node) == 2:
        _shape(node[1], "formula", count, depth + 1)
    elif (category == "formula" and tag in ("and", "or", "implies")
          and len(node) == 3):
        for child in node[1:]:
            _shape(child, "formula", count, depth + 1)
    elif (category == "formula" and tag in ("all", "exists")
          and len(node) == 3):
        _name(node[1])
        _shape(node[2], "formula", count, depth + 1)
    else:
        raise ValueError("AST category, tag or node length mismatch.")


def _variables(node):
    tag = node[0]
    if tag == "v":
        return {node[1]}
    if tag == "c":
        return set()
    if tag in ("p", "f"):
        children = node[2]
    elif tag in ("all", "exists"):
        return _variables(node[2]) - {node[1]}
    else:
        children = node[1:]
    return set().union(*(_variables(child) for child in children))


def evaluate(formula, world, environment=None):
    """Exact Boolean truth in one finite structure, not satisfiability.

    A supplied environment is a plain dict of at most 16 variable/object
    bindings. All free variables need bindings, even in a skipped branch.
    More than 200,000 truth/term visits raises a resource ValueError.
    """
    _shape(formula)
    size, constants, functions, predicates = _tables(world)
    environment = {} if environment is None else environment
    _dictionary(environment, 16)
    for name, obj in environment.items():
        _name(name)
        _integer(obj, 0, size - 1)
    if not _variables(formula) <= set(environment):
        raise ValueError("A free variable lacks an assignment.")

    def signature(node):
        tag = node[0]
        if tag == "c":
            if node[1] not in constants:
                raise ValueError("Undeclared constant.")
        elif tag in ("p", "f"):
            table = predicates if tag == "p" else functions
            if node[1] not in table or table[node[1]][0] != len(node[2]):
                raise ValueError("Undeclared symbol or arity mismatch.")
            for child in node[2]:
                signature(child)
        elif tag in ("all", "exists"):
            signature(node[2])
        elif tag != "v":
            for child in node[1:]:
                signature(child)

    signature(formula)

    visits = [0]

    def visit():
        visits[0] += 1
        if visits[0] > 200000:
            raise ValueError("Finite interpretation visit budget exceeded.")

    def term(node, env):
        visit()
        if node[0] == "v":
            return env[node[1]]
        if node[0] == "c":
            return constants[node[1]]
        return functions[node[1]][1][tuple(term(t, env) for t in node[2])]

    def truth(node, env):
        visit()
        tag = node[0]
        if tag == "p":
            arguments = tuple(term(t, env) for t in node[2])
            return arguments in predicates[node[1]][1]
        if tag == "eq":
            return term(node[1], env) == term(node[2], env)
        if tag == "not":
            return not truth(node[1], env)
        if tag == "and":
            return truth(node[1], env) and truth(node[2], env)
        if tag == "or":
            return truth(node[1], env) or truth(node[2], env)
        if tag == "implies":
            return not truth(node[1], env) or truth(node[2], env)
        values = (truth(node[2], env | {node[1]: obj}) for obj in range(size))
        return all(values) if tag == "all" else any(values)

    return truth(formula, dict(environment))


def substitute(expression, substitution):
    """One simultaneous substitution on a term or atom, not binders."""
    _shape(expression, "expression")
    _dictionary(substitution, 16)
    for name, term in substitution.items():
        _name(name)
        _shape(term, "term")

    def visit(node):
        if node[0] == "v":
            return substitution.get(node[1], node)
        if node[0] == "c":
            return node
        return (node[0], node[1], tuple(visit(t) for t in node[2]))

    result = visit(expression)
    _shape(result, "expression")
    return result


def unify(left, right):
    """Return a solved idempotent dict, or None for no finite-tree unifier.

    Both inputs must be terms or both atoms, each within AST bounds, with
    at most 16 distinct variables together. 10,000 internal term visits
    and AST expansion bounds raise ValueError rather than logical failure.
    """
    for expression in (left, right):
        _shape(expression, "expression")
    if (left[0] == "p") != (right[0] == "p"):
        raise ValueError("Unification categories differ.")
    if len(_variables(left) | _variables(right)) > 16:
        raise ValueError("At most 16 variables in a unification problem.")
    bindings, pending, visits = {}, [(left, right)], [0]

    def resolve(node):
        visits[0] += 1
        if visits[0] > 10000:
            raise ValueError("Unification term-visit budget exceeded.")
        if node[0] == "v" and node[1] in bindings:
            return resolve(bindings[node[1]])
        if node[0] in ("v", "c"):
            return node
        result = (node[0], node[1], tuple(resolve(t) for t in node[2]))
        _shape(result, "expression")
        return result

    while pending:
        a, b = (resolve(t) for t in pending.pop())
        if a == b:
            continue
        if b[0] == "v" and a[0] != "v":
            a, b = b, a
        if a[0] == "v":
            if a[1] in _variables(b):
                return None
            bindings[a[1]] = b
        elif a[:2] != b[:2] or len(a[2]) != len(b[2]):
            return None
        else:
            pending.extend(zip(a[2], b[2]))
    solved = {name: resolve(term) for name, term in sorted(bindings.items())}
    if substitute(left, solved) != substitute(right, solved):
        raise AssertionError("Internal unification certificate failed.")
    return solved


def horn_closure(constants, signatures, facts, rules):
    """Positive function-free/equality-free closure and growing-round layers.

    At most 8 distinct constants, 8 predicates of arity 0..3, 4096 atoms,
    16 rules, 6 premises/rule and 4 variables/rule. Facts are a tuple/set/
    frozenset of ground atoms; rules are tuple (premise_tuple, head) pairs.
    Up to 200,000 ground rule assignments are checked. Bounds raise
    ValueError. The returned frozenset/layers never encode logical negation.
    """
    _sequence(constants, 8)
    unique_constants = set(_name(c) for c in constants)
    if not constants or len(unique_constants) != len(constants):
        raise ValueError("Constants must be nonempty and distinct.")
    _dictionary(signatures, 8)
    for name, arity in signatures.items():
        _name(name)
        _integer(arity, 0, 3)
    if set(constants) & set(signatures):
        raise ValueError("Constant and predicate names must differ.")
    if type(facts) not in (tuple, set, frozenset) or len(facts) > 4096:
        raise ValueError("Malformed bounded fact collection.")
    _sequence(rules, 16)

    def atom(node, ground):
        _shape(node, "formula")
        if (node[0] != "p" or node[1] not in signatures
                or len(node[2]) != signatures[node[1]]):
            raise ValueError("Horn predicate/signature mismatch.")
        for term in node[2]:
            if term[0] == "c" and term[1] in constants:
                continue
            if term[0] == "v" and not ground:
                continue
            raise ValueError("Horn arguments must be constants or variables.")

    for fact in facts:
        atom(fact, True)
    prepared = []
    for rule in rules:
        _sequence(rule, 2)
        if len(rule) != 2:
            raise ValueError("Rule needs premises and head.")
        premises, head = rule
        _sequence(premises, 6)
        for node in premises + (head,):
            atom(node, False)
        variables = set().union(*(_variables(n) for n in premises + (head,)))
        if len(variables) > 4:
            raise ValueError("Rule has more than four variables.")
        prepared.append((premises, head, tuple(sorted(variables))))
    current, layers, visits = set(facts), [], 0
    while True:
        additions = set()
        for premises, head, variables in prepared:
            for values in product(constants, repeat=len(variables)):
                visits += 1
                if visits > 200000:
                    raise ValueError("Horn grounding budget exceeded.")
                assignment = dict(zip(variables, (("c", c) for c in values)))
                if all(substitute(p, assignment) in current for p in premises):
                    additions.add(substitute(head, assignment))
        additions -= current
        if not additions:
            return frozenset(current), tuple(layers)
        layers.append(frozenset(additions))
        current |= additions


def main():
    def v(name):
        return "v", name

    def c(name):
        return "c", name

    def p(name, *args):
        return "p", name, args

    x, y, z = v("x"), v("y"), v("z")
    names = ("anna", "bea", "intro", "advanced", "logic", "safety")
    constants = dict(zip(names, range(6))) | {"anna_record": 0}
    relations = {
        "Staff": (1, {(0,), (1,)}),
        "Course": (1, {(2,), (3,)}),
        "Concept": (1, {(4,), (5,)}),
        "Permit": (1, {(0,)}),
        "Attends": (2, {(0, 2), (1, 3)}),
        "Covers": (2, {(2, 4), (2, 5), (3, 4)}),
        "Qualified": (2, {(0, 4), (0, 5), (1, 4)}),
    }
    mentor = {(i,): 1 if i in (0, 1) else i for i in range(6)}
    world = make_structure(6, constants, {"mentor": (1, mentor)}, relations)
    individual = ("all", "x", ("implies", p("Staff", x),
                  ("exists", "y",
                   ("and", p("Course", y), p("Attends", x, y)))))
    common = ("exists", "y", ("and", p("Course", y),
              ("all", "x", ("implies", p("Staff", x), p("Attends", x, y)))))
    print("Individual/common course:", evaluate(individual, world),
          evaluate(common, world))
    alias = ("eq", c("anna"), c("anna_record"))
    print("Alias equality:", evaluate(alias, world))
    print("Simultaneous x:", substitute(x, {"x": y, "y": c("anna")}))
    print("Repeated-variable mismatch:",
          unify(p("R", x, x), p("R", c("anna"), c("bea"))))
    print("Occurs-check cycle:", unify(x, ("f", "mentor", (x,))))
    signatures = {name: row[0] for name, row in relations.items()}
    facts = tuple(p(name, *(c(names[i]) for i in key))
                  for name, (arity, extension) in relations.items()
                  if name not in ("Qualified", "Permit")
                  for key in sorted(extension))
    training = ((p("Staff", x), p("Attends", x, y), p("Course", y),
                 p("Covers", y, z), p("Concept", z)), p("Qualified", x, z))
    permission = ((p("Qualified", x, c("logic")),
                   p("Qualified", x, c("safety"))), p("Permit", x))
    closure, layers = horn_closure(
        names, signatures, facts, (training, permission))
    print("Horn initial/final/layer sizes:", len(facts), len(closure),
          tuple(len(layer) for layer in layers))
    print("Derived Permit(anna)/Permit(bea):",
          p("Permit", c("anna")) in closure,
          p("Permit", c("bea")) in closure)
    print("Ground-atom bound:",
          sum(len(names) ** r for r in signatures.values()))


if __name__ == "__main__":
    main()
