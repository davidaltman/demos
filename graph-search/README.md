# Setup
## This code was executed with Python 3.10.11 on a macbook pro
## Setup a virtual environment:
>> python -m venv .venv
>> source .venv/bin/activate
>> pip install --upgrade pip
>> pip install -r requirements.txt

## execute unit tests for the multi source graph search:
>> python multi_source_graph_search.py

# Demo of graph with matplotlib
>> python demo_search.py


# Graph Analysis Tool

This project provides a set of tools for generating, analyzing, and visualizing graphs. It's particularly useful for studying network structures and performing shortest path analyses.

## Features

- Generate unique node names
- Create graphs with custom source and target nodes
- Find shortest paths using Dijkstra's algorithm
- Parallel processing for efficient path finding
- Visualize graphs using matplotlib
- Comprehensive unit tests

## Dependencies

- networkx
- matplotlib
- concurrent.futures (part of Python standard library)

## Classes

### NodeNameGenerator

Generates unique node names using combinations of uppercase letters.

### GraphAnalyzer

The main class for graph operations:
- Generate nodes and edges
- Find shortest paths
- Analyze and sort path lengths
- Visualize the graph

### TestGraphAnalyzer

Contains unit tests for the GraphAnalyzer class.

## Usage

To use the GraphAnalyzer:

```python
analyzer = GraphAnalyzer()
targets, sources = analyzer.generate_node_names(number_nodes, num_sources)
edges = analyzer.generate_edges_triple(sources, targets)
analyzer.graph.add_weighted_edges_from(edges)
analyzer.find_shortest_paths(sources, targets, num_threads)
results = analyzer.sort_pathlengths()