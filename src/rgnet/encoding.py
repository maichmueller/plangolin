import itertools
from functools import singledispatchmethod
from typing import Dict

import networkx as nx

from pymimir import State, Problem


class ColorGraphEncoder:
    def __init__(self, problem: Problem):
        feature_map: Dict[str | None, int] = {None: 0}
        for object_ in problem.objects:
            feature_map["o_" + object_.name] = 0
        for i, typ in enumerate(problem.domain.types):
            feature_map["t_" + typ.name] = 1 + i
        for pred in problem.domain.predicates:
            a, b, c = 0, 0, 0
            for a in range(pred.arity):
                feature_map["p_{predicate.name}:{i}"] = 1 + i + a
            for b in range(pred.arity):
                feature_map["p_{predicate.name}_g:{i}"] = 1 + i + a + b
            for c in range(pred.arity):
                feature_map["~p_{predicate.name}_g:{i}"] = 1 + i + a + b + c
        self._color_feature_map = feature_map
        self._problem = problem

    def encode(self, state: State):
        graph = nx.Graph(state=state)

        curr_predicate_node = 0

        # Add vertices
        for obj in self._problem.objects:
            graph.add_node(
                self._color_feature_map[f"o_{obj.name}"],
                color=self._color_feature_map[f"t_{obj.type.name}"],
                info=obj.type.name,
            )

            # Add atom edges
            for atom, prefix, suffix in itertools.chain(
                # the regular atoms of the state first
                zip(
                    filter(lambda a: a.predicate.name != "=", state.get_atoms()),
                    itertools.repeat(""),
                    itertools.repeat(""),
                ),
                # then the goal atoms
                (
                    (literal.atom, "~" if literal.negated else "", "_g")
                    for literal in self._problem.goal
                ),
            ):
                prev_node_id = None
                for pos, atom_obj in enumerate(atom.terms):
                    object_node = self._color_feature_map["o_" + atom_obj.name]
                    node_str = f"{prefix}p_{atom.predicate.name}{suffix}:{pos}"
                    # Add predicate node
                    graph.add_node(
                        curr_predicate_node,
                        color=self._color_feature_map[node_str],
                        info=node_str,
                    )
                    # Connect predicate node to object node
                    graph.add_edge(object_node, curr_predicate_node)
                    graph.add_edge(curr_predicate_node, object_node)
                    if prev_node_id is not None:
                        # connect with previous positional node
                        graph.add_edge(prev_node_id, curr_predicate_node)
                        graph.add_edge(curr_predicate_node, prev_node_id)

                    prev_node_id = curr_predicate_node
                    curr_predicate_node += 1

        assert not nx.is_directed(graph)
        return graph
