import itertools
from collections import defaultdict
from typing import Dict, List

import networkx as nx
import pymimir as mi
import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType

from rgnet.encoding.encoder_base import StateEncoderBase


class HeteroEncoding(StateEncoderBase):

    def __init__(
        self,
        domain: mi.Domain,
        hidden_size: int,  # TODO Decouple hidden_size from encoding
        obj_name: str = "obj",
        goal_suffix: str = "_g",
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.obj_name: str = obj_name
        self.arity_by_pred = {pred.name: pred.arity for pred in domain.predicates}
        self.goal_suffix = goal_suffix
        for pred_name in list(self.arity_by_pred.keys()):
            self.arity_by_pred[self.goal_pred(pred_name)] = self.arity_by_pred[
                pred_name
            ]

    def goal_pred(self, pred_name: str) -> str:
        return f"{pred_name}{self.goal_suffix}"

    def encode(self, state: mi.State) -> nx.Graph:
        # Build hetero graph from state
        # One node for each object
        # One node for each atom
        # Edge label = position in atom
        problem = state.get_problem()
        graph = nx.Graph()

        for obj in problem.objects:
            graph.add_node(obj.name, type=self.obj_name)

        state_atoms = zip(
            filter(lambda a: a.predicate.name != "=", state.get_atoms()),
            itertools.repeat(""),
            itertools.repeat(False),
        )
        goal_atoms = (
            (literal.atom, "~" if literal.negated else "", True)
            for literal in problem.goal
        )

        for atom, prefix, is_goal in itertools.chain(state_atoms, goal_atoms):

            if atom.predicate.arity == 0:
                continue

            predicate: mi.Predicate = atom.predicate
            pred_name: str = predicate.name

            node_type = self.goal_pred(pred_name) if is_goal else pred_name

            obj_names = ",".join(obj.name for obj in atom.terms)

            atom_node = f"{prefix}{node_type}({obj_names})"
            graph.add_node(
                atom_node,
                type=node_type,
            )

            for pos, obj in enumerate(atom.terms):
                # Connect predicate node to object node
                graph.add_edge(obj.name, atom_node, position=str(pos))
        return graph

    def to_pyg_data(self, graph: nx.Graph) -> HeteroData:

        nodes_by_type: Dict[str, List[str]] = defaultdict(list)
        for key, value in nx.get_node_attributes(graph, "type").items():
            nodes_by_type[value].append(key)

        nidx_by_type = {
            ntype: {node: i for i, node in enumerate(nodes)}
            for ntype, nodes in nodes_by_type.items()
        }

        data = HeteroData()
        # Create x_dict the feature matrix for each node type
        for node_type, nodes_of_type in nodes_by_type.items():
            hidden = (
                self.hidden_size
                if node_type == self.obj_name
                else self.hidden_size * self.arity_by_pred[node_type]
            )
            data[node_type].x = torch.zeros(
                (len(nodes_of_type), hidden), dtype=torch.float32
            )

        # Group edges by src, position, dst
        edge_dict: Dict[EdgeType, List[torch.Tensor]] = {}
        for src, dst, attr in graph.edges.data():
            src_type: str = graph.nodes[src]["type"]
            dst_type: str = graph.nodes[dst]["type"]
            pos: str = attr["position"]
            edge_type = (src_type, pos, dst_type)
            reverse_edge = (dst_type, pos, src_type)
            if edge_type not in edge_dict:
                edge_dict[edge_type] = []
                edge_dict[reverse_edge] = []  # reverse edge

            edge_dict[edge_type].append(
                torch.tensor([nidx_by_type[src_type][src], nidx_by_type[dst_type][dst]])
            )
            edge_dict[reverse_edge].append(
                torch.tensor([nidx_by_type[dst_type][dst], nidx_by_type[src_type][src]])
            )
        # Stack grouped edge_indices and add to data
        for (src, rel, dst), value in edge_dict.items():
            # HeteroData want str,str,str as edge keys
            data[src, rel, dst].edge_index = torch.stack(value, dim=1)

        return data
