import networkx as nx


def get_colors(graph: nx.Graph):
    return sorted(set(graph[node]["color"].value for node in graph))
