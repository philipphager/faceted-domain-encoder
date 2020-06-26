import json

import networkx as nx

from .graph import KnowledgeGraphFactory, KnowledgeGraph

CATEGORIES = [
    'Airport [A]',
    'Airline [B]',
    'Aircraft [C]',
    'Components [D]',
    'Engine [E]',
    'Fault [F]',
    'Geographic Location [G]',
    'Manufacturer [H]',
    'Material [I]',
    'Person [J]',
    'Process [K]',
    'Unit [L]',
]


class AviationFactory(KnowledgeGraphFactory):
    def __init__(self, graph_path: str, in_path: str):
        super().__init__(graph_path)
        self.in_path = in_path

    def _create(self):
        concept2name, term2concept, concept2category, concept2neighbours = self._parse_file(self.in_path)
        graph = self._create_graph(concept2category, concept2neighbours)
        return KnowledgeGraph(graph, term2concept, concept2name)

    def _create_graph(self, concept2category, concept2neighbour):
        graph = nx.Graph()
        graph.add_node('root', name='root', categories=list())

        for concept, category in concept2category.items():
            if concept is not None:
                graph.add_node(concept, name=concept, categories=[category])

        for concept, neighbours in concept2neighbour.items():
            for neighbour in neighbours:
                if concept is not None and neighbour is not None:
                    graph.add_edge(concept, neighbour)

        return graph

    def _parse_file(self, path: str):
        with open(path, 'r') as f:
            json_load = json.loads(f.read())
            concept2name = json_load['concept2name']
            term2concept = json_load['term2concept']
            concept2category = json_load['concept2category']
            concept2neighbours = json_load['concept2neighbours']

        return concept2name, term2concept, concept2category, concept2neighbours
