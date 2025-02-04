from __future__ import annotations

import itertools
import typing
import warnings
from collections import namedtuple
from enum import Enum
from functools import cache, singledispatchmethod
from typing import Any, Dict, Iterable, Iterator, NamedTuple, Optional

import numpy as np

from xmimir import Domain, XAtom, XDomain, XLiteral, XPredicate, XProblem


class FeatureKey(NamedTuple):
    name: str
    position: int | None
    is_goal: bool
    is_negated: bool


class FeatureMode(Enum):
    # mere integer naming the category
    categorical = 0
    # one-hot vector encoding
    one_hot = 1
    # combinatorial encoding of features in a vector of user-determined length
    combinatorial = 2
    # structural encoding of the features (1st layer: predicate, 2nd layer: pos, 3rd layer: goal, 4th layer: negated)
    structural = 3


class FeatureMap:

    @cache
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.needs_initialization = True
        return obj

    def __init__(
        self,
        domain: XDomain | None = None,
        predicates: tuple[XPredicate] | None = None,
        mode: FeatureMode = FeatureMode.categorical,
        enc_len: Optional[int] = None,
        ignore_arg_position: bool = False,
    ):
        if self.needs_initialization:
            if not isinstance(mode, FeatureMode):
                raise ValueError(f"{mode = } not instance of {FeatureMode}.")
            self._predicates = domain.predicates if domain else predicates
            if not self._predicates:
                raise ValueError(
                    f"No predicates given. Passed arguments for domain and predicates:\n{repr(domain)}\n{repr(predicates)}"
                )
            self._predicate_map = {p: i for i, p in enumerate(self._predicates)}
            self._mode = mode
            self._enc_len = enc_len
            self._ignore_arg_position = ignore_arg_position
            self._lookup = self._build(mode, enc_len)
            # mark this object fully initialized
            self.needs_initialization = False

    @property
    def ignore_arg_position(self):
        return self._ignore_arg_position

    @property
    def predicates(self):
        return self._predicates

    @property
    def predicate_map(self):
        return self._predicate_map

    @property
    def mode(self):
        return self._mode

    @property
    def encoding_len(self):
        return self._enc_len

    def _key_gen(self) -> Iterator[FeatureKey]:
        for pred in self._predicates:
            arity = pred.arity
            for pos in (
                range(0, max(1, arity)) if not self._ignore_arg_position else (None,)
            ):
                for is_goal, is_negated in (
                    (False, False),
                    (True, False),
                    (True, True),
                ):
                    yield FeatureKey(
                        pred.name, pos if arity > 0 else None, is_goal, is_negated
                    )

    def _build(self, mode: FeatureMode, encoding_len: Optional[int] = None):
        self._mode = mode
        match mode:
            case FeatureMode.categorical:
                none_feature = 0
                feature_iter = itertools.count(start=1)
            case FeatureMode.one_hot:
                encoding_len = sum(
                    3 * max(1, pred.arity * (not self._ignore_arg_position))
                    for pred in self._predicates
                )
                none_feature = np.zeros(encoding_len, dtype=np.int8)
                feature_iter = np.eye(encoding_len, dtype=np.int8)
                # make read-only views
                none_feature.flags.writeable = False
                feature_iter.flags.writeable = False
            case FeatureMode.combinatorial:
                required_nr_states = sum(
                    3 * max(1, pred.arity) for pred in self._predicates
                )
                if encoding_len is None:
                    encoding_len, _ = divmod(required_nr_states, 2)
                else:
                    # each position is a 0 or 1, encoding_len many positions -> 2^enc_len many different states possible,
                    # only (0,0,...,0,0) is reserved as the null encoding (hence, -1)
                    max_encoded_values = 2**encoding_len - 1
                    if max_encoded_values < required_nr_states:
                        raise ValueError(
                            f"Given {encoding_len=} cannot represent all the necessary "
                            f"encoding states ({required_nr_states})"
                        )
                    if encoding_len >= required_nr_states:
                        warnings.warn(
                            f"{encoding_len=} is no less than {required_nr_states=}. Will revert to one-hot encoding."
                        )
                        return self._build(
                            FeatureMode.combinatorial, divmod(required_nr_states, 2)[0]
                        )

                def feature_vector_gen():
                    unit_matrix = np.eye(encoding_len, dtype=np.int8)
                    # make read-only views
                    unit_matrix.flags.writeable = False
                    for idx_comb in itertools.chain(
                        itertools.combinations(range(encoding_len), n_combs)
                        for n_combs in range(1, encoding_len + 1)
                    ):
                        # sum up the unit vectors of the given indices to create another feature vector
                        yield sum(unit_matrix[i] for i in idx_comb)

                none_feature = np.zeros(encoding_len, dtype=np.int8)
                # make read-only views
                none_feature.flags.writeable = False
                feature_iter = feature_vector_gen()
            case _:
                raise ValueError(f"Unsupported feature mode {mode}")

        key_iter = self._key_gen()
        featuremap: Dict[Optional[FeatureKey], np.ndarray] = dict(
            zip(key_iter, feature_iter)
        )
        featuremap[None] = none_feature

        return featuremap

    def __eq__(self, other: FeatureMap):
        return self._mode == other._mode and self._enc_len == other._enc_len

    @singledispatchmethod
    def __call__(self, _: Any, *args, **kwargs) -> np.ndarray:
        return self._lookup[None]

    @__call__.register
    def _(self, atom: XAtom, pos: int | None = None, *args, **kwargs):
        if not self._ignore_arg_position and pos is None and atom.predicate.arity > 0:
            raise ValueError(
                f"atom {atom} has arity {atom.predicate.arity} > 0, but given pos is None"
            )
        return self._lookup[
            FeatureKey(atom.predicate.name, self._adapt_pos(pos), False, False)
        ]

    @__call__.register
    def _(
        self,
        literal: XLiteral,
        pos: int | None = None,
        *args,
        is_goal: bool = False,
        **kwargs,
    ):
        atom = literal.atom
        if not self._ignore_arg_position and pos is None and atom.predicate.arity > 0:
            raise ValueError(
                f"atom {atom} has arity {atom.predicate.arity} > 0, but given pos is None"
            )
        return self._lookup[
            FeatureKey(
                atom.predicate.name,
                self._adapt_pos(pos),
                is_goal,
                literal.is_negated,
            )
        ]

    def _adapt_pos(self, pos: int | None) -> int | None:
        if pos is None or self._ignore_arg_position:
            return None
        return pos
