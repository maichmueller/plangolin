import itertools
from typing import Dict, Optional

import networkx as nx
import torch_geometric as pyg
from pymimir import State, Problem, Domain
from torch_geometric.data import Data


class ColorGraphEncoder:
    """
    A state encoder into an associated colored state-graph for a specified domain.

    Each object will receive its own node, each predicate might receive its own node (if configured to do so),
    each atom will receive multiple nodes which
    """

    def __init__(self, domain: Domain, add_predicate_nodes: bool = True):
        """
        Initialize the color graph encoder

        Parameters
        ----------
        domain: pymimir.Domain, the domain over which instance-states will be encoded
        add_predicate_nodes: bool, whether to add summarising predicate nodes to the graph.
            Predicate nodes will connect with respective pos-0-atom nodes.
        """
        feature_map: Dict[str | None, int] = {None: 0}
        i = None  # error trigger incase there are no types in the problem
        for i, typ in enumerate(domain.types):
            feature_map["t_" + typ.name] = i
        offset = i + 1
        for pred in domain.predicates:
            if pred.arity == 0:
                feature_map[f"p_{pred.name}"] = offset
                feature_map[f"p_{pred.name}_g"] = offset + 1
                feature_map[f"~p_{pred.name}_g"] = offset + 2
                offset += 3
            else:
                for pos in range(pred.arity):
                    feature_map[f"p_{pred.name}:{pos}"] = offset
                    feature_map[f"p_{pred.name}_g:{pos}"] = offset + 1
                    feature_map[f"~p_{pred.name}_g:{pos}"] = offset + 2
                    offset += 3

        self.add_predicate_nodes = add_predicate_nodes
        self._feature_map = feature_map
        self._domain = domain

    @property
    def domain(self):
        return self._domain

    @property
    def feature_mapping(self):
        return self._feature_map

    def encode(self, state: State, problem: Optional[Problem] = None):
        problem = problem if problem is not None else state.get_problem()
        assert problem is not None, "Problem was neither given nor part of the state"
        graph = nx.Graph(state=state)

        for pred in self.domain.predicates:
            graph.add_node(
                pred.name,
                feature=self._feature_map[None],
            )

        for obj in problem.objects:
            graph.add_node(
                obj.name,
                feature=self._feature_map[f"t_{obj.type.name}"],
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
            obj_names = ",".join(obj.name for obj in atom.terms)

            if atom.predicate.arity == 0:
                atom_node = f"{prefix}{pred_name}({obj_names}){suffix}"
                graph.add_node(
                    atom_node,
                    color=self._feature_map[f"{prefix}p_{pred_name}{suffix}"],
                )
            for pos, obj in enumerate(atom.terms):
                object_node = obj.name
                feature = self._feature_map[f"{prefix}p_{pred_name}{suffix}:{pos}"]
                atom_node = f"{prefix}{pred_name}({obj_names}){suffix}:{pos}"

                graph.add_node(
                    atom_node,
                    feature=feature,
                )
                # Connect atom node to object node
                graph.add_edge(object_node, atom_node)
                if pos > 0:
                    # connect preceding positional node with the current one
                    graph.add_edge(prev_predicate_node, atom_node)
                elif self.add_predicate_nodes:
                    # pos 0-node gets the connection to the predicate summarising node
                    graph.add_edge(atom.predicate.name, atom_node)

                prev_predicate_node = atom_node
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
        # We want floating point features
        data.x = data["feature"].float()
        # Ensure that every node has a vector of features even though its size is one
        data.x = data.x.view(-1, 1)
        return data
