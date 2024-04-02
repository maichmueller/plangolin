import itertools
from functools import singledispatchmethod
from typing import Dict

import networkx as nx

from pymimir import State, Problem


class NameToIndexMap:
    def __init__(self):
        self.table: Dict[str | None, int] = {None: 0}

    def __getitem__(self, string: str):
        if string not in self.table:
            self.table[string] = len(self.table)
        return self.table[string]

    def size(self):
        return len(self.table)


class ColorGraphEncoder:
    def __init__(self, problem: Problem):
        index_map = NameToIndexMap()
        for i, elem in enumerate(
            itertools.chain(
                ("o_" + object_.name for object_ in problem.objects),
                ("t_" + type_.name for type_ in problem.domain.types),
                (
                    f"p_{predicate.name}:{i}"
                    for predicate in problem.domain.predicates
                    for i in range(predicate.arity)
                ),
                (
                    f"p_{predicate.name}_g:{i}"
                    for predicate in problem.domain.predicates
                    for i in range(predicate.arity)
                ),
                (
                    f"~p_{predicate.name}_g:{i}"
                    for predicate in problem.domain.predicates
                    for i in range(predicate.arity)
                ),
            ),
        ):
            index_map.table[elem] = i
        self._index_map = index_map
        self._problem = problem

    def encode(self, state: State):
        graph = nx.Graph(state=state)

        curr_predicate_node = self._index_map.size()

        # Add vertices
        for obj in self._problem.objects:
            graph.add_node(
                self._index_map[f"o_{obj.name}"],
                color=self._index_map[f"t_{obj.type.name}"],
                info=obj.type.name,
            )

            # Add atom edges
            for atom, prefix, suffix in itertools.chain(
                zip(
                    filter(lambda a: a.predicate.name != "=", state.get_atoms()),
                    itertools.repeat(""),
                    itertools.repeat(""),
                ),
                (
                    (literal.atom, "~" if literal.negated else "", "_g")
                    for literal in self._problem.goal
                ),
            ):
                prev_node_id = None
                for pos, atom_obj in enumerate(atom.terms):
                    object_node = self._index_map["o_" + atom_obj.name]
                    node_str = f"{prefix}p_{atom.predicate.name}{suffix}:{pos}"
                    # Add predicate node
                    graph.add_node(
                        curr_predicate_node,
                        color=self._index_map[node_str],
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
