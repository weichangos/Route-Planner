import logging
import itertools
import random
import math
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, filename='logging.log', filemode='w')


class RoutePlanner:
    def __init__(self, locs_n_cords) -> None:
        self.locs_n_cords = locs_n_cords
        self.G = nx.Graph()
        self.g_odd_complete = nx.Graph()
        self.g_odd_complete_min_edges = nx.Graph()
        self.g_aug = nx.MultiGraph()
        self.euler_circuit = []

    def locs_to_pd_df(self):
        try:
            df = pd.DataFrame({'location_id': self.locs_n_cords[0],
                               'x': [self.locs_n_cords[1][i][0] for i in range(len(self.locs_n_cords[1]))],
                               'y': [self.locs_n_cords[1][i][1] for i in range(len(self.locs_n_cords[1]))]})
        except (KeyError, IndexError, TypeError) as e:
            logging.error(f"Error: {e}")
            raise ValueError("Input has to be a list of two lists, first being the nodes and second being the coordinates.")
        else:
            return df

    def generate_graph(self, edge_probability=0.3):
        self.G.add_nodes_from(self.locs_n_cords[0])
        nx.set_node_attributes(self.G, dict(zip(self.G.nodes(), self.locs_n_cords[1])), "pos")

        for u in self.G.nodes:
            for v in self.G.nodes:
                if random.random() < edge_probability and u != v:
                    self.G.add_edge(u, v)

        pos = nx.get_node_attributes(self.G, "pos")
        gnodes = nx.get_node_attributes(self.G, "pos")

        distances = []
        for i in range(len(self.G.edges)):
            dis = math.sqrt(sum([(a - b) ** 2 for a, b in zip(gnodes.get(str(list(self.G.edges)[i][0])),
                                                              gnodes.get(str(list(self.G.edges)[i][1])))]))
            distances.append(round(dis, 3))

        nx.set_edge_attributes(self.G, dict(zip(self.G.edges(), distances)), "distance")

    def find_nodes_with_odd_degree(self):
        return [node for node, edge in self.G.degree() if edge % 2 == 1]

    def compute_odd_node_pairs(self, nodes_with_odd_degree):
        return list(itertools.combinations(nodes_with_odd_degree, 2))

    def get_shortest_paths_distances(self, pairs, edge_weight_name):
        distances = {}
        for pair in pairs:
            distances[pair] = nx.dijkstra_path_length(self.G, pair[0], pair[1], weight=edge_weight_name)
        return distances

    def create_complete_graph(self, pair_weights, flip_weights=True):
        g = nx.Graph()
        for k, v in pair_weights.items():
            wt_i = -v if flip_weights else v
            g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})
        return g

    def find_odd_matching(self, g_odd_complete):
        odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)
        odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in dict(odd_matching_dupes).items()]))
        return odd_matching

    def find_com_min_edge(self, g_odd_complete):
        odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)
        odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in dict(odd_matching_dupes).items()]))
        return nx.Graph(odd_matching)

    def add_augmenting_path_to_graph(self, min_weight_pairs):
        graph_aug = nx.MultiGraph(self.G.copy())
        for pair in min_weight_pairs:
            graph_aug.add_edge(pair[0], pair[1], **{'distance': nx.dijkstra_path_length(self.G, pair[0], pair[1]), 'path': 'augmented'})
        return graph_aug

    def create_eulerian_circuit(self, starting_node=None):
        euler_circuit = []
        naive_circuit = list(nx.eulerian_circuit(self.g_aug, source=starting_node))
        for edge in naive_circuit:
            edge_data = self.g_aug.get_edge_data(edge[0], edge[1])
            if len(edge_data) == 1 and 'path' not in edge_data[0].keys():
                edge_att = self.G[edge[0]][edge[1]]
                euler_circuit.append((edge[0], edge[1], edge_att))
            else:
                aug_path = nx.shortest_path(self.G, edge[0], edge[1], weight='distance')
                aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))
                for edge_aug in aug_path_pairs:
                    edge_aug_att = self.G[edge_aug[0]][edge_aug[1]]
                    euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))
        self.euler_circuit = euler_circuit

    def plot_graphs(self):
        plt.figure(1)
        nx.draw(self.G, nx.get_node_attributes(self.G, "pos"), with_labels=True, node_color='lightblue', node_size=500)
        plt.title('Original Graph')

    #     plt.figure(2)
    #     nx.draw(self.g_odd_complete, nx.get_node_attributes(self.g_odd_complete, "pos"), with_labels=True, node_color='lightblue', node_size=500)
    #     plt.title('Complete Graph for Odd Nodes')

    #     plt.figure(3)
    #     nx.draw(self.g_odd_complete_min_edges, nx.get_node_attributes(self.g_odd_complete_min_edges, "pos"), with_labels=True, node_color='lightblue', node_size=500)
    #     plt.title('Minimum Matching Edges')

        plt.figure(4)
        nx.draw(self.g_aug, nx.get_node_attributes(self.g_aug, "pos"), with_labels=True, node_color='lightblue', node_size=500)
        plt.title('Augmented Graph')

        plt.show()


# Usage
locs = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
        [(1.0, 1.0), (1.5, 4.5), (5.4, 1.5), (1.7, 2.0), (2.2, 4.0), (0.0, 0.0), (4.5, 2.5), (5.1, 4.2), (3.4, 3.5)]]

route_planner = RoutePlanner(locs)
locs_dataframe = route_planner.locs_to_pd_df()
logging.info("Input data: \n %s", locs_dataframe)

route_planner.generate_graph(edge_probability=0.3)
logging.info("Table for paths: \n %s", pd.DataFrame({'paths': list(nx.get_edge_attributes(route_planner.G, "distance").keys()), 'distances': list(nx.get_edge_attributes(route_planner.G, "distance").values())}))

nodes_with_odd_degree = route_planner.find_nodes_with_odd_degree()
logging.info('Nodes with odd degree: \n %s', nodes_with_odd_degree)

odd_node_pairs = route_planner.compute_odd_node_pairs(nodes_with_odd_degree)
logging.info("Potential odd node pairs: \n %s", odd_node_pairs)

odd_node_pairs_shortest_paths = route_planner.get_shortest_paths_distances(odd_node_pairs, 'distance')
logging.info("Min distance pairs: \n %s", pd.DataFrame({'paths': list(odd_node_pairs_shortest_paths.keys()),
                                                         'distances': list(odd_node_pairs_shortest_paths.values())}))

route_planner.g_odd_complete = route_planner.create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)

route_planner.g_odd_complete_min_edges = route_planner.find_com_min_edge(route_planner.g_odd_complete)

odd_matching = route_planner.find_odd_matching(route_planner.g_odd_complete)
logging.info('Paths traversed twice, if not existed in original graph, shortest connecting path is automatically augmented: \n %s', odd_matching)


route_planner.g_aug = route_planner.add_augmenting_path_to_graph(odd_matching)

route_planner.create_eulerian_circuit(starting_node='A')
logging.info(route_planner.euler_circuit)

route_planner.plot_graphs()