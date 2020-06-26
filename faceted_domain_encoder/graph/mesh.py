import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List

import networkx as nx

from .graph import KnowledgeGraphFactory, KnowledgeGraph

CATEGORIES = [
    'Anatomy [A]',
    'Organisms [B]',
    'Diseases [C]',
    'Chemicals and Drugs [D]',
    'Analytical, Diagnostic and Therapeutic Techniques and Equipment [E]',
    'Psychiatry and Psychology [F]',
    'Biological Sciences [G]',
    'Physical Sciences [H]',
    'Anthropology, Education, Sociology and Social Phenomena [I]',
    'Technology and Food and Beverages [J]',
    'Humanities [K]',
    'Information Science [L]',
    'Persons [M]',
    'Health Care [N]',
    'Publication Characteristics [V]',
    'Geographic Locations [Z]'
]


class MeSHFactory(KnowledgeGraphFactory):
    def __init__(self, graph_path: str, in_path: str):
        super().__init__(graph_path)
        self.in_path = in_path
        self.max_synonyms = 5

    def _create(self):
        tree2concept, term2concept, concept2name, concept2category = self._parse_xml(self.in_path)
        graph = self._create_graph(tree2concept, concept2name, concept2category)
        return KnowledgeGraph(graph, term2concept, concept2name)

    def _parse_xml(self, path: str):
        tree2concept = {}
        term2concept = {}
        concept2name = {}
        concept2category = defaultdict(lambda: [])

        root = ET.parse(path).getroot()
        concept2name['root'] = 'root'
        tree2concept['root'] = 'root'
        term2concept['root'] = 'root'
        concept2category['root'] = 'Not Assigned'

        # Parse MeSH headings
        for heading in root.iter('DescriptorRecord'):
            concept_id = heading.find('DescriptorUI').text
            concept_name = heading.find('DescriptorName').find('String').text.lower()
            concept2name[concept_id] = concept_name

            # Parse tree numbers for each MeSH heading
            trees = list(heading.iter('TreeNumber'))

            for tree in trees:
                tree_number = tree.text
                tree2concept[tree_number] = concept_id
                # Get category by first letter of tree number
                concept2category[concept_id].append(self._mesh_category(tree_number[0]))

            # Parse synonym terms for each MeSH heading
            for concept in heading.iter('Concept'):
                concept_name = concept.find('ConceptName').find('String').text.lower()
                term2concept[concept_name] = concept_id
                term2concept[concept_id] = concept_id

                for term in concept.iter('Term'):
                    term_name = term.find('String').text.lower()
                    term2concept[term_name] = concept_id

        return tree2concept, term2concept, concept2name, concept2category

    def _create_graph(self, tree2concept: Dict[str, str], concept2name: Dict[str, str],
                      concept2category: Dict[str, List[str]]):
        graph = nx.Graph()

        for concept, name in concept2name.items():
            graph.add_node(concept, name=name, categories=concept2category[concept])

        for tree, concept in tree2concept.items():
            if '.' in tree:
                # Remove last 4 characters from tree number to get parent
                # E.g., 'D03.633.100.221.173' -> 'D03.633.100.221'
                parent_tree = tree[:-4]
            elif tree != 'root':
                # Attach top-level nodes to our root node
                parent_tree = 'root'
            else:
                # Do not attach the root node to any parent
                parent_tree = 'no_parent'

            if parent_tree in tree2concept:
                parent_concept = tree2concept[parent_tree]
                graph.add_edge(concept, parent_concept)

        return graph

    def _mesh_category(self, prefix: str = ''):
        prefix2category = {
            'A': 'Anatomy [A]',
            'B': 'Organisms [B]',
            'C': 'Diseases [C]',
            'D': 'Chemicals and Drugs [D]',
            'E': 'Analytical, Diagnostic and Therapeutic Techniques and Equipment [E]',
            'F': 'Psychiatry and Psychology [F]',
            'G': 'Biological Sciences [G]',
            'H': 'Physical Sciences [H]',
            'I': 'Anthropology, Education, Sociology and Social Phenomena [I]',
            'J': 'Technology and Food and Beverages [J]',
            'K': 'Humanities [K]',
            'L': 'Information Science [L]',
            'M': 'Persons [M]',
            'N': 'Health Care [N]',
            'V': 'Publication Characteristics [V]',
            'Z': 'Geographic Locations [Z]'
        }

        return prefix2category.get(prefix, 'Not Assigned')
