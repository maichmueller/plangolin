from typing import Union

# import the necessary classes:
# import pymimir.advanced.formalism as formalism  # for when migrating to new pymimir-API
import pymimir as formalism

# Define the type hint for the Atom class:
Atom = Union[
    formalism.StaticAtom,
    formalism.FluentAtom,
    formalism.DerivedAtom,
]
GroundAtom = Union[
    formalism.StaticGroundAtom,
    formalism.FluentGroundAtom,
    formalism.DerivedGroundAtom,
]
Literal = Union[
    formalism.StaticLiteral,
    formalism.FluentLiteral,
    formalism.DerivedLiteral,
]
GroundLiteral = Union[
    formalism.StaticGroundLiteral,
    formalism.FluentGroundLiteral,
    formalism.DerivedGroundLiteral,
]
Predicate = Union[
    formalism.StaticPredicate,
    formalism.FluentPredicate,
    formalism.DerivedPredicate,
]

Object = formalism.Object
Domain = formalism.Domain
Problem = formalism.Problem

__all__ = [
    "Atom",
    "GroundAtom",
    "Literal",
    "GroundLiteral",
    "Predicate",
    "Object",
    "Domain",
    "Problem",
]
