from functools import singledispatch

from pymimir import Atom, Literal, Object, Predicate


@singledispatch
def node_of(item, *args, **kwargs) -> str | None:
    raise NotImplementedError("node_of is not implemented for {}".format(type(item)))


@node_of.register
def atom_node(
    atom: Atom,
    pos: int | None = None,
    *args,
    as_predicate: bool = False,
    **kwargs,
) -> str | None:
    if as_predicate:
        return node_of(atom.predicate, is_goal=False, is_negated=False)
    return f"{atom.get_name()}:{pos}"


@node_of.register
def predicate_node(
    predicate: Predicate, *, is_goal: bool, is_negated: bool, **kwargs
) -> str | None:
    negation = "~" if is_negated else ""
    suffix = "_g" if is_goal else ""
    return f"{negation}{predicate.name}{suffix}"


@node_of.register
def literal_node(
    literal: Literal,
    pos: int | None = None,
    *,
    as_predicate: bool = False,
    **kwargs,
) -> str | None:
    if as_predicate:
        return node_of(literal.atom.predicate, is_goal=True, is_negated=literal.negated)
    negation = "~" if literal.negated else ""
    pos_string = f":{pos}" if pos is not None else ""
    return f"{negation}{literal.atom.get_name()}_g{pos_string}"


@node_of.register
def object_node(obj: Object, *args, **kwargs) -> str | None:
    return obj.name


@node_of.register
def none_node(none: None, *args, **kwargs) -> str | None:
    return None
