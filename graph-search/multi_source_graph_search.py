import networkx as nx
import random
import string
import concurrent.futures
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
import unittest
import logger as logger


class NodeNameGenerator:
    def __init__(self):
        """
        Initialize a new NodeNameGenerator instance.

        This constructor sets up the initial state of the NodeNameGenerator:
        - Defines the uppercase letters to be used in node names.
        - Creates generators for single, two, three, and four letter combinations.
        - Chains these generators to create a sequence of unique node names.

        Attributes:
            uppercase (str): A string containing all uppercase ASCII letters.
            generators (list): A list of generator functions for different name lengths.
            current_generator (itertools.chain): An iterator that chains all generators.
        """
        self.uppercase = string.ascii_uppercase
        self.generators = [
            self._single_letter_generator(),
            self._two_letter_generator(),
            self._three_letter_generator(),
            self._four_letter_generator(),
        ]
        self.current_generator = itertools.chain(*self.generators)

    def _single_letter_generator(self):
        return (c for c in self.uppercase)

    def _two_letter_generator(self):
        return ("".join(combo) for combo in itertools.product(self.uppercase, repeat=2))

    def _three_letter_generator(self):
        return ("".join(combo) for combo in itertools.product(self.uppercase, repeat=3))

    def _four_letter_generator(self):
        return ("".join(combo) for combo in itertools.product(self.uppercase, repeat=4))

    def generate_node_names(self, n):
        return list(itertools.islice(self.current_generator, n))


class GraphAnalyzer:
    def __init__(self):
        """
        Initialize a new GraphAnalyzer instance.

        This constructor sets up the initial state of the GraphAnalyzer:
        - Creates an empty undirected graph using networkx.
        - Initializes an empty dictionary to store analysis results.
        - Sets up a logger for the class.
        - Creates a NodeNameGenerator for generating unique node names.
        - Initializes a defaultdict to track path lengths from sources to targets.

        Attributes:
            graph (nx.Graph): An empty undirected graph.
            results (dict): A dictionary to store analysis results.
            logger (Logger): A logger instance for this class.
            node_name_generator (NodeNameGenerator): An instance to generate unique node names.
            source_path_to_targets (defaultdict): A defaultdict to track path lengths.
        """
        self.graph = nx.Graph()
        self.results = {}
        self.logger = logger.get_logger(__name__)
        self.node_name_generator = NodeNameGenerator()
        self.source_path_to_targets = defaultdict(int)

    def generate_node_names(self, n=30, num_sources=10):
        """
        Generate a list of unique node names and select a subset as source nodes.

        This method uses the NodeNameGenerator to create a list of unique names,
        and then randomly selects a subset of these names to be used as source nodes.

        Args:
            n (int, optional): The total number of node names to generate. Defaults to 30.
            num_sources (int, optional): The number of source nodes to select. Defaults to 10.

        Returns:
            tuple: A tuple containing two lists:
                - names (list): A list of all generated node names.
                - sources (list): A randomly selected subset of names to be used as source nodes.

        Note:
            The number of source nodes (num_sources) should not exceed the total number of nodes (n).
        """
        names = self.node_name_generator.generate_node_names(n)
        return names, random.sample(names, num_sources)

    def generate_edges_triple(self, source_nodes, target_nodes):
        """
        Generate a list of edge triples connecting source and target nodes.

        This method creates a path that starts from a target node, passes through
        randomly selected source nodes, and ends at another randomly chosen target node.
        Each edge in the path is represented as a triple (node1, node2, weight).

        Args:
            source_nodes (list): A list of source nodes to be connected.
            target_nodes (list): A list of target nodes to start and end paths.

        Returns:
            list: A list of tuples, each representing an edge (node1, node2, weight).
                  The weight is always 1 in this implementation.

        Note:
            - The function ensures that no duplicate edges are created.
            - The resulting graph may not be fully connected.
            - Each path starts and ends with a target node, passing through source nodes.
        """
        edges = []
        used_pairs = set()

        for start in target_nodes:
            random.shuffle(source_nodes)
            current_node = start
            for intermediate in source_nodes:
                if (
                    current_node,
                    intermediate,
                ) not in used_pairs and current_node != intermediate:
                    weight = 1
                    edges.append((current_node, intermediate, weight))
                    used_pairs.add((current_node, intermediate))
                    current_node = intermediate

            target = random.choice(target_nodes)
            if (current_node, target) not in used_pairs and current_node != target:
                weight = 1
                edges.append((current_node, target, weight))
                used_pairs.add((current_node, target))

        return edges

    def plot_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_size=700)
        nx.draw_networkx_edges(self.graph, pos, width=2)
        nx.draw_networkx_labels(self.graph, pos, font_size=20, font_family="sans-serif")
        plt.title("Network Graph Visualization")
        plt.axis("off")
        plt.show()

    def dijkstra_path(self, source, targets):
        """
        Find the shortest paths from a single source to multiple targets using Dijkstra's algorithm.

        Args:
            source: The starting node for the paths.
            targets (list): A list of target nodes to find paths to.

        Returns:
            dict: A dictionary where keys are target nodes and values are the shortest paths
                  from the source to each target. If no path exists, the value is None.

        Note:
            This method uses networkx's dijkstra_path function to compute the shortest paths.
            If a path to a target doesn't exist, it's recorded as None in the result.
        """
        paths = {}
        for target in targets:
            try:
                path = nx.dijkstra_path(self.graph, source, target)
                paths[target] = path
            except nx.NetworkXNoPath:
                paths[target] = None
        return paths

    def find_shortest_paths(self, sources, targets, num_threads):
        """
        Find the shortest paths from multiple source nodes to multiple target nodes concurrently.

        This method uses a ThreadPoolExecutor to parallelize the computation of shortest paths
        from each source to all targets using Dijkstra's algorithm.

        Args:
            sources (list): A list of source nodes.
            targets (list): A list of target nodes.
            num_threads (int): The number of threads to use for parallel computation.

        Returns:
            None: The results are stored in the self.results dictionary, where keys are source nodes
                  and values are dictionaries mapping target nodes to their shortest paths.

        Note:
            If an exception occurs during path finding for a source, the result for that source
            will be set to None and an error will be logged.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_source = {
                executor.submit(self.dijkstra_path, source, targets): source
                for source in sources
            }
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result()
                    self.results[source] = result
                except Exception as exc:
                    self.logger.error(f"Source {source} generated an exception: {exc}")
                    self.results[source] = None

    def sort_pathlengths(self) -> dict:
        """
        Sort and calculate the total path lengths for each source node.

        This method iterates through the results dictionary, calculating the total
        path length from each source to all its targets. It then sorts the sources
        based on their total path lengths in ascending order.

        Returns:
            dict: A sorted dictionary where keys are source nodes and values are
                  their total path lengths to all targets.
        """
        for source, paths in self.results.items():
            for target, path in paths.items():
                if path is not None:
                    self.source_path_to_targets[source] += len(path) - 1

        self.source_path_to_targets = dict(
            sorted(self.source_path_to_targets.items(), key=lambda item: item[1])
        )
        return self.source_path_to_targets


class TestGraphAnalyzer(unittest.TestCase):
    def test_generated_nodes(
        self, number_nodes=30, num_sources=6, num_targets=6, num_threads=5,
        plot_graph=False
    ):
        """
        Test the graph generation and analysis functionality with generated nodes.

        This method creates a graph with randomly generated nodes and edges,
        performs shortest path analysis, and logs the results.

        Args:
            number_nodes (int, optional): Total number of nodes to generate. Defaults to 30.
            num_sources (int, optional): Number of source nodes to use. Defaults to 6.
            num_targets (int, optional): Number of target nodes to use. Defaults to 6.
            num_threads (int, optional): Number of threads for parallel computation. Defaults to 5.

        The test performs the following steps:
        1. Generate nodes and edges
        2. Add edges to the graph
        3. Find shortest paths from sources to targets
        4. Sort and log path lengths
        5. Log detailed path information for the first source
        6. Log overall performance metrics

        No assertions are made in this test; it primarily serves to exercise
        the graph analysis functionality and provide logging output.
        """
        analyzer = GraphAnalyzer()
        targets, sources = analyzer.generate_node_names(number_nodes, num_sources)
        edges = analyzer.generate_edges_triple(sources, targets)
        analyzer.graph.add_weighted_edges_from(edges)
        targets = random.sample(targets, num_targets)
        start_time = time.time()
        analyzer.find_shortest_paths(sources, targets, num_threads)
        end_time = time.time()
        total_time = end_time - start_time

        source_path_to_targets = analyzer.sort_pathlengths()

        end_time = time.time()
        total_time = end_time - start_time

        for key, value in source_path_to_targets.items():
            analyzer.logger.debug(f"source: {key}, total path length:, {value}")

        for source, pathLen in source_path_to_targets.items():
            analyzer.logger.info(f"Best path length from source {source}: {pathLen}")
            for target, path in analyzer.results[source].items():
                if path is not None:
                    analyzer.logger.info(f" {len(path)-1} to node {target}: {path}")
                else:
                    analyzer.logger.info(f"No path found from {source} to {target}")
            break

        edges = analyzer.graph.edges()
        num_edges = len(edges)
        analyzer.logger.info(
            f"Total time taken with {num_edges} edges, {number_nodes} nodes, "
            f"{len(sources)} sources, {len(targets)} targets, {num_threads} threads: {total_time:.4f} seconds"
        )
        if plot_graph:
            analyzer.plot_graph()

    def test_generate_node_names_default(self):
        """
        Test the generate_node_names method with default parameters.

        This test case verifies that:
        1. The method returns the correct number of names (30) and sources (10).
        2. All source nodes are included in the list of generated names.
        """
        analyzer = GraphAnalyzer()
        names, sources = analyzer.generate_node_names()
        self.assertEqual(len(names), 30)
        self.assertEqual(len(sources), 10)
        self.assertTrue(all(name in names for name in sources))

    def test_generate_node_names_custom(self):
        """
        Test the generate_node_names method with custom parameters.

        This test case verifies that:
        1. The method returns the correct number of names (50) and sources (5) when custom values are provided.
        2. All source nodes are included in the list of generated names.
        3. The method correctly handles non-default input parameters.
        """
        analyzer = GraphAnalyzer()
        names, sources = analyzer.generate_node_names(50, 5)
        self.assertEqual(len(names), 50)
        self.assertEqual(len(sources), 5)
        self.assertTrue(all(name in names for name in sources))

    def test_generate_node_names_large(self):
        """
        Test the generate_node_names method with a large number of nodes and sources.

        This test case verifies that:
        1. The method can handle a large number of nodes (700) and sources (20).
        2. The correct number of names and sources are generated.
        3. All source nodes are included in the list of generated names.
        4. The method performs correctly with larger input values.
        """
        analyzer = GraphAnalyzer()
        names, sources = analyzer.generate_node_names(700, 20)
        self.assertEqual(len(names), 700)
        self.assertEqual(len(sources), 20)
        self.assertTrue(all(name in names for name in sources))

    def test_shortest_paths(self):
        """
        Test the find_shortest_paths method of the GraphAnalyzer class.

        This test case verifies that:
        1. The method correctly finds the shortest paths from multiple sources to multiple targets.
        2. The results match the expected paths in a predefined graph.
        3. The method handles multiple threads correctly.

        The test sets up a simple graph with known shortest paths, runs the find_shortest_paths method,
        and compares the results with the expected paths.
        """
        self.analyzer = GraphAnalyzer()
        self.analyzer.graph.add_weighted_edges_from(
            [(1, 2, 1), (1, 3, 4), (2, 3, 2), (2, 4, 7), (3, 4, 3)]
        )
        self.sources = [1, 2]
        self.targets = [3, 4]
        self.num_threads = 2
        expected_paths = {
            1: {3: [1, 2, 3], 4: [1, 2, 3, 4]},
            2: {3: [2, 3], 4: [2, 3, 4]},
        }
        self.analyzer.find_shortest_paths(self.sources, self.targets, self.num_threads)
        self.assertEqual(self.analyzer.results, expected_paths)


if __name__ == "__main__":
    # Usage
    unittest.main()
