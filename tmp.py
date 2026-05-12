import random
import networkx as nx
import numpy as np


def forest_fire_graph(
    num_nodes,
    forward_burn_prob=0.35,
    backward_burn_prob=0.0,
    max_burn_visits=20,
    seed=None,
):
    """
    Simple undirected forest-fire graph generator.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    forward_burn_prob : float
        Probability of recursively burning through neighbors.

    backward_burn_prob : float
        Extra probability of linking to already-visited nodes.

    max_burn_visits : int
        Limits recursion depth/work.

    seed : int or None
        RNG seed.

    Returns
    -------
    networkx.Graph
    """

    rng = random.Random(seed)

    if num_nodes <= 0:
        return nx.Graph()

    if num_nodes == 1:
        g = nx.Graph()
        g.add_node(0)
        return g

    g = nx.Graph()
    g.add_node(0)

    for new_node in range(1, num_nodes):

        g.add_node(new_node)

        # Choose ambassador
        ambassador = rng.randrange(new_node)

        # Always connect to ambassador
        g.add_edge(new_node, ambassador)

        queue = [ambassador]
        visited = {ambassador}

        burn_visits = 0

        while queue and burn_visits < max_burn_visits:

            current = queue.pop(0)
            burn_visits += 1

            neighbors = list(g.neighbors(current))
            rng.shuffle(neighbors)

            for nbr in neighbors:

                if nbr == new_node:
                    continue

                # Forward burn
                if rng.random() < forward_burn_prob:

                    g.add_edge(new_node, nbr)

                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

                # Optional extra densification
                elif rng.random() < backward_burn_prob:
                    g.add_edge(new_node, nbr)

    return g


def average_degree(g):
    return 2 * g.number_of_edges() / g.number_of_nodes()

breakpoint()