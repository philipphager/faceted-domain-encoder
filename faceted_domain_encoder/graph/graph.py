import json
import os
from typing import Dict, Optional, List, Set

import networkx as nx
from networkx.readwrite.json_graph import node_link_graph, node_link_data


class KnowledgeGraph:
    def __init__(self,
                 graph: nx.Graph,
                 term2node: Dict[str, str],
                 node2term: Dict[str, str],
                 category: str = None):
        self.graph = graph
        self.term2node = term2node
        self.node2term = node2term
        self.category = category

    def subgraph(self, category):
        nodes = [n for n, d in self.graph.nodes(data=True) if category in d['categories']]
        subgraph = self.graph.subgraph(nodes)

        keep_nodes = set(subgraph.nodes())
        subgraph_term2node = dict((t, n) for t, n in self.term2node.items() if n in keep_nodes)
        subgraph_node2term = dict((n, t) for n, t in self.node2term.items() if n in keep_nodes)
        return KnowledgeGraph(subgraph, subgraph_term2node, subgraph_node2term)

    def find_node(self, term: str) -> Optional[str]:
        return self.term2node.get(term, None)

    def find_nodes(self, sentence: List[List[str]]) -> List[str]:
        return [self.find_node(t) for t in sentence if self.find_node(t)]

    def find_distinct_nodes(self, sentence) -> Set[str]:
        return set(self.find_nodes(sentence))

    def find_node_candidates(self, term: str, limit: int = 5) -> List[str]:
        return list(sorted([t for t in self.term2node.keys() if term in t], key=len))[:limit]

    def find_graph_neighbours(self, node: str) -> List[str]:
        return list(self.graph.neighbors(node))

    def find_categories(self, node: str) -> str:
        return self.graph.nodes[node]['categories']


class KnowledgeGraphFactory:
    def __init__(self, graph_path):
        self.graph_path = graph_path

    def load(self, force=False) -> KnowledgeGraph:
        if force or not os.path.exists(self.graph_path):
            graph = self._create()
            self._to_json(graph, self.graph_path)

        return self._from_json(self.graph_path)

    def _create(self) -> KnowledgeGraph:
        # Implement parsing logic
        pass

    def _to_json(self, graph: KnowledgeGraph, graph_path: str):
        with open(graph_path, 'w') as f:
            json_dump = json.dumps({
                'graph': node_link_data(graph.graph),
                'term2node': graph.term2node,
                'node2term': graph.node2term,
            })

            f.write(json_dump)

    def _from_json(self, graph_path: str) -> KnowledgeGraph:
        with open(graph_path, 'r') as f:
            json_load = json.loads(f.read())
            graph = node_link_graph(json_load['graph'])
            term2node = json_load['term2node']
            node2term = json_load['node2term']

            return KnowledgeGraph(graph, term2node, node2term)
