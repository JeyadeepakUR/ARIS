import networkx as nx
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Tuple


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concept_weights = defaultdict(float)

    def build_graph(self, documents: List[object]):
        self.graph.clear()
        for doc in documents:
            self._process_document(doc)
        self._add_weighted_edges()
        return self.graph

    def _process_document(self, doc):
        self.graph.add_node(doc.id, type='document', title=doc.title, impact_score=doc.impact_score, publication_date=doc.publication_date)
        all_concepts = (doc.keywords or []) + (doc.concepts or [])
        for concept in all_concepts:
            concept_id = f"concept_{concept.lower().replace(' ', '_')}"
            if concept_id not in self.graph:
                self.graph.add_node(concept_id, type='concept', name=concept)
            self.graph.add_edge(doc.id, concept_id, type='contains')
            self.concept_weights[concept_id] += doc.impact_score
        for author in doc.authors:
            author_id = f"author_{author.lower().replace(' ', '_')}"
            if author_id not in self.graph:
                self.graph.add_node(author_id, type='author', name=author)
            self.graph.add_edge(author_id, doc.id, type='authored')

    def _add_weighted_edges(self):
        concept_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'concept']
        for i, concept1 in enumerate(concept_nodes):
            for concept2 in concept_nodes[i+1:]:
                docs1 = set(pred for pred in self.graph.predecessors(concept1) if self.graph.nodes[pred].get('type') == 'document')
                docs2 = set(pred for pred in self.graph.predecessors(concept2) if self.graph.nodes[pred].get('type') == 'document')
                common_docs = docs1.intersection(docs2)
                if common_docs:
                    weight = len(common_docs) / max(len(docs1), len(docs2))
                    if weight > 0.1:
                        self.graph.add_edge(concept1, concept2, type='related', weight=weight, common_docs=len(common_docs))

    def find_research_gaps(self, min_connections: int = 2):
        concept_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'concept']
        gaps = []
        for concept in concept_nodes:
            connections = len([n for n in self.graph.neighbors(concept) if self.graph.nodes[n].get('type') == 'concept'])
            if connections < min_connections:
                gaps.append(self.graph.nodes[concept]['name'])
        return gaps

    def identify_trending_concepts(self, time_window_months: int = 12):
        cutoff_date = datetime.now() - timedelta(days=time_window_months * 30)
        concept_trends = defaultdict(list)
        for doc_id in self.graph.nodes():
            if self.graph.nodes[doc_id].get('type') == 'document':
                pub_date = self.graph.nodes[doc_id].get('publication_date')
                if pub_date and pub_date > cutoff_date:
                    for successor in self.graph.successors(doc_id):
                        if self.graph.nodes[successor].get('type') == 'concept':
                            concept_trends[successor].append(pub_date)
        trending = []
        for concept, dates in concept_trends.items():
            if len(dates) >= 3:
                dates.sort()
                recent_half = len(dates) // 2
                recent_count = len([d for d in dates[-recent_half:]])
                total_count = len(dates)
                trend_score = recent_count / total_count if total_count > 0 else 0
                if trend_score > 0.6:
                    trending.append((self.graph.nodes[concept]['name'], trend_score))
        return sorted(trending, key=lambda x: x[1], reverse=True)