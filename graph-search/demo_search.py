from multi_source_graph_search import TestGraphAnalyzer

test = TestGraphAnalyzer()
test.test_generated_nodes(number_nodes=20, num_sources=6, num_targets=6, num_threads=5,
plot_graph=True)