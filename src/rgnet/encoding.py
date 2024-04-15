import itertools
from typing import Dict, Optional

import networkx as nx
import torch_geometric as pyg
from pymimir import State, Problem, Domain
from torch_geometric.data import Data


class ColorGraphEncoder:
    def __init__(self, domain: Domain):
        feature_map: Dict[str | None, int] = {None: 0}
        i = None  # error trigger incase there are no types in the problem
        for i, typ in enumerate(domain.types):
            feature_map["t_" + typ.name] = i
        for pred in domain.predicates:
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
        self._domain = domain

    def encode(self, state: State, problem: Optional[Problem] = None):
        problem = problem if problem is not None else state.get_problem()
        assert problem is not None, "Problem was neither given nor part of the state"
        graph = nx.Graph(state=state)
        for obj in problem.objects:
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
            for literal in problem.goal
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

    def encoding_to_pyg_data(self, state: State, **kwargs) -> Data:
        graph = self.encode(state, **kwargs)
        # In the pyg.utils.from_networkx the graph is converted to a DiGraph
        # In this process it has to be pickled, which is not defined for pymimir.State
        del graph.graph["state"]
        # Every node has to have the same features
        for node, attr in graph.nodes.data():
            if "info" in attr:
                del attr["info"]
        data: Data = pyg.utils.from_networkx(graph)
        # We want floating poit features
        data.x = data["color"].float()
        # Ensure that every node has a vector of features even though its size is one
        data.x = data.x.view(-1, 1)
        return data
