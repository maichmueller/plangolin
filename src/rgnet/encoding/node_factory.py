from __future__ import annotations

from functools import singledispatch

from pymimir import Atom, Literal, Object, Predicate

NEGATION_PREFIX = "~"
GOAL_SUFFIX = "_g"

Node = str


@singledispatch
def node_of(item, *args, **kwargs) -> Node | None:
    raise NotImplementedError("node_of is not implemented for {}".format(type(item)))


@node_of.register
def atom_node(
    atom: Atom,
    pos: int | None = None,
    *args,
    as_predicate: bool = False,
    **kwargs,
) -> Node | None:
    if as_predicate:
        return node_of(atom.predicate, is_goal=False, is_negated=False)
    return f"{atom.get_name()}:{pos}"


@node_of.register
def predicate_node(
    predicate: Predicate, *, is_goal: bool = False, is_negated: bool = False, **kwargs
) -> Node | None:
    prefix = NEGATION_PREFIX if is_negated else ""
    suffix = GOAL_SUFFIX if is_goal else ""
    return f"{prefix}{predicate.name}{suffix}"


@node_of.register
def literal_node(
    literal: Literal,
    pos: int | None = None,
    *,
    as_predicate: bool = False,
    **kwargs,
) -> Node | None:
    if as_predicate:
        # by default, we assume that literals are goal atoms
        return node_of(
            literal.atom.predicate,
            is_goal=kwargs.get("is_goal", True),
            is_negated=literal.negated,
        )
    prefix = NEGATION_PREFIX if literal.negated else ""
    pos_string = f":{pos}" if pos is not None else ""
    return f"{prefix}{literal.atom.get_name()}{GOAL_SUFFIX}{pos_string}"


@node_of.register
def object_node(obj: Object, *args, **kwargs) -> Node | None:
    return obj.name


@node_of.register
def none_node(none: None, *args, **kwargs) -> Node | None:
    return None
