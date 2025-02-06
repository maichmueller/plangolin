from __future__ import annotations

from functools import cache, singledispatchmethod

from xmimir import (
    Atom,
    GroundAtom,
    GroundLiteral,
    Literal,
    Object,
    Predicate,
    XAtom,
    XLiteral,
    XPredicate,
)

Node = str


class NodeFactory:
    def __init__(self, negation_prefix: str = "not ", goal_suffix: str = "_g"):
        self.negation_prefix = negation_prefix
        self.goal_suffix = goal_suffix

    def __hash__(self):
        return hash((self.negation_prefix, self.goal_suffix))

    def __eq__(self, other):
        if not isinstance(other, NodeFactory):
            return NotImplemented
        return (
            self.negation_prefix == other.negation_prefix
            and self.goal_suffix == other.goal_suffix
        )

    @singledispatchmethod
    def __call__(self, item, *args, **kwargs) -> Node | None:
        raise NotImplementedError(
            "__call__ is not implemented for type {}".format(type(item))
        )

    @cache
    @__call__.register
    def atom_node(
        self,
        atom: XAtom,
        pos: int | None = None,
        *args,
        **kwargs,
    ) -> Node | None:
        if pos is None:
            return str(atom)
        return f"{atom}:{pos}"

    @cache
    @__call__.register
    def predicate_node(
        self,
        predicate: XPredicate,
        *,
        is_goal: bool = False,
        is_negated: bool = False,
        **kwargs,
    ) -> Node | None:
        prefix = self.negation_prefix if is_negated else ""
        suffix = self.goal_suffix if is_goal else ""
        return f"{prefix}{predicate.name}{suffix}"

    @cache
    @__call__.register
    def literal_node(
        self,
        literal: XLiteral,
        pos: int | None = None,
        *args,
        **kwargs,
    ) -> Node | None:
        pos_string = f":{pos}" if pos is not None else ""
        return f"{literal.atom}{self.goal_suffix}{pos_string}"

    @cache
    @__call__.register
    def object_node(self, obj: Object, *args, **kwargs) -> Node | None:
        return obj.get_name()

    @cache
    @__call__.register
    def none_node(self, none: None, *args, **kwargs) -> Node | None:
        return str(None)
