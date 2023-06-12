import networkx as nx
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import itertools
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, filename='logging.log', filemode='w')

##########Step 1: generate graph for input#############

# Input
locs = [['A','B','C','D','E','F','G','H','I'], [(1.0, 1.0), (1.5, 4.5), (5.4, 1.5), (1.7, 2.0), (2.2, 4.0), (0.0, 0.0), (4.5, 2.5), (5.1, 4.2), (3.4, 3.5)]]

# Convert into pd DataFrame
def locs_to_pd_df(locations):
    try:
        df = pd.DataFrame({'location_id':locations[0], 'x':[locations[1][i][0] for i in range(len(locations[1]))], 'y':[locations[1][i][1] for i in range(len(locations[1]))]})
    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"Error: {e}")
        raise ValueError("Input has to be a list of two lists, first being the nodes and second being the coordinates.")
    else:
        return df

locs_dataframe = locs_to_pd_df(locs)
logging.info("Input data: \n %s", locs_dataframe)

# Networkx
edge_probability = 0.3

G = nx.Graph()
G.add_nodes_from(locs[0])
nx.set_node_attributes(G, dict(zip(G.nodes(), locs[1])), "pos")

for u in G.nodes:
    for v in G.nodes:
        if random.random() < edge_probability:
            while u == v:
                v = random.choice(locs[0])
            G.add_edge(u, v)

pos = nx.get_node_attributes(G, "pos")
gnodes = nx.get_node_attributes(G, "pos")

distances = []
for i in range(len(G.edges)):
    dis = math.sqrt(sum([(a - b) ** 2 for a, b in zip(gnodes.get(str(list(G.edges)[i][0])), gnodes.get(str(list(G.edges)[i][1])))]))
    distances.append(round(dis, 3))

nx.set_edge_attributes(G, dict(zip(G.edges(), distances)), "distance")

logging.info("Table for paths: \n %s", pd.DataFrame({'paths': list(nx.get_edge_attributes(G, "distance").keys()), 'distances': list(nx.get_edge_attributes(G, "distance").values())}))

####### Step 2: Find all nodes that have an odd number of edges attached ########

# Calculate list of nodes with odd degree
nodes_with_odd_degree = [node for node, edge in G.degree() if edge % 2 == 1]
logging.info('Nodes with odd degree: \n %s', nodes_with_odd_degree)

####### Step 3: Compute all pairs of odd degree nodes ######

# Compute all pairs of odd degree nodes
odd_node_pairs = list(itertools.combinations(nodes_with_odd_degree, 2))
# print(odd_node_pairs)

def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
    return distances

###### Step 4: Compute shortest paths and create complete graph for odd nodes ######

# Compute shortest paths. Return a dictionary with node pairs keys and a single value equal to the shortest path distance.
odd_node_pairs_shortest_paths = get_shortest_paths_distances(G, odd_node_pairs, 'distance')
# print(dict(odd_node_pairs_shortest_paths.items()))
logging.info("Min distance pairs: \n %s", pd.DataFrame({'paths': list(dict(odd_node_pairs_shortest_paths.items()).keys()), 'distances': list(dict(odd_node_pairs_shortest_paths.items()).values())}))

# Graphing the complete graph for all odd nodes
def create_complete_graph(pair_weights, flip_weights=True):
    g = nx.Graph()
    for k, v in pair_weights.items():
        wt_i = - v if flip_weights else v
        g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})  
    return g

g_odd_complete = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)

####### Step 5: Find the set of odd node pairs whose combined sum is the smallest #######

odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)
# print('Number of edges in matching: {}'.format(len(odd_matching_dupes)))

odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in dict(odd_matching_dupes).items()]))
# print(odd_matching)

g_odd_complete_min_edges = nx.Graph(odd_matching)

###### Step 6: Add augmenting paths to the graph ######

def add_augmenting_path_to_graph(graph, min_weight_pairs):
    graph_aug = nx.MultiGraph(graph.copy())  # We need to make the augmented graph a MultiGraph so we can add parallel edges
    for pair in min_weight_pairs:
        graph_aug.add_edge(pair[0], pair[1], **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]), 'path': 'augmented'})
    return graph_aug

g_aug = add_augmenting_path_to_graph(G, odd_matching)

###### Step 7: Create Eulerian circuit using original graph edges ######

def create_eulerian_circuit(graph_augmented, graph_original, starting_node=None):
    euler_circuit = []
    naive_circuit = list(nx.eulerian_circuit(graph_augmented, source=starting_node))
    for edge in naive_circuit:
        edge_data = graph_augmented.get_edge_data(edge[0], edge[1])
        if len(edge_data) == 1 and 'path' not in edge_data[0].keys():
            edge_att = graph_original[edge[0]][edge[1]]
            euler_circuit.append((edge[0], edge[1], edge_att))
        else:
            aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight='distance')
            aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))
            for edge_aug in aug_path_pairs:
                edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
                euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))
    return euler_circuit

euler_circuit = create_eulerian_circuit(g_aug, G, 'A')
logging.info(euler_circuit)

# Plotting all the graphs together
plt.figure(1)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
plt.title('Original Graph')
plt.figure(2)
nx.draw(g_odd_complete, pos, with_labels=True, node_color='lightblue', node_size=500)
plt.title('Complete Graph for Odd Nodes')
plt.figure(3)
nx.draw(g_odd_complete_min_edges, pos, with_labels=True, node_color='lightblue', node_size=500)
plt.title('Minimum Matching Edges')
plt.figure(4)
nx.draw(g_aug, pos, with_labels=True, node_color='lightblue', node_size=500)
plt.title('Augmented Graph')
plt.show()
