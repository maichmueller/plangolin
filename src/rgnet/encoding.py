import itertools
from functools import singledispatchmethod
from typing import Dict

import networkx as nx

from pymimir import State, Problem


class ColorGraphEncoder:
    def __init__(self, problem: Problem):
        feature_map: Dict[str | None, int] = {None: 0}
        i = None  # error trigger incase there are no types in the problem
        for i, typ in enumerate(problem.domain.types):
            feature_map["t_" + typ.name] = i
        for pred in problem.domain.predicates:
            offset = i + 1
            for pos in range(pred.arity):
                feature_map[f"p_{pred.name}:{pos}"] = offset + pos
                offset += 1
            for pos in range(pred.arity):
                feature_map[f"p_{pred.name}_g:{pos}"] = offset + pos
                offset += 1
            for pos in range(pred.arity):
                feature_map[f"~p_{pred.name}_g:{pos}"] = offset + pos
        self._color_feature_map = feature_map
        self._problem = problem

    def encode(self, state: State):
        graph = nx.Graph(state=state)
        for obj in self._problem.objects:
            graph.add_node(
                obj.name,
                color=self._color_feature_map[f"t_{obj.type.name}"],
                info=obj.type.name,
            )

        state_atoms = zip(
            filter(lambda a: a.predicate.name != "=", state.get_atoms()),
            itertools.repeat(""),
            itertools.repeat(""),
        )
        goal_atoms = (
            (literal.atom, "~" if literal.negated else "", "_g")
            for literal in self._problem.goal
        )
        for atom, prefix, suffix in itertools.chain(state_atoms, goal_atoms):
            pred_name = atom.predicate.name
            prev_predicate_node = None
            for pos, obj in enumerate(atom.terms):
                object_node = obj.name
                predicate_node = f"{prefix}p_{pred_name}{suffix}:{pos}"

                graph.add_node(
                    predicate_node,
                    color=self._color_feature_map[predicate_node],
                )
                # Connect predicate node to object node
                graph.add_edge(object_node, predicate_node)
                if pos > 0:
                    # connect preceding positional node with the current one
                    graph.add_edge(prev_predicate_node, predicate_node)

                prev_predicate_node = predicate_node
        return graph
