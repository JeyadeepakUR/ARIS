import os
import sqlite3
import json
import hashlib
import pickle
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from dataclasses import dataclass, asdict

from utils.config import EMBEDDING_MODELS, DATABASE_PATH, EMBEDDINGS_DIR
from analysis.document_processing import AdvancedNLPProcessor, parse_document_structure, extract_title, extract_authors, extract_abstract
from analysis.impact_scoring import calculate_impact_score
from analysis.knowledge_graph import KnowledgeGraphBuilder
from analysis.literature_review import LiteratureAnalysisEngine


@dataclass
class ResearchDocument:
    id: str
    title: str
    content: str
    abstract: str
    authors: List[str]
    keywords: List[str]
    publication_date: datetime
    source_type: str
    citation_count: int
    impact_score: float
    metadata: dict
    embedding: np.ndarray = None
    entities: List[str] = None
    concepts: List[str] = None
    methodology: str = None
    findings: List[str] = None
    limitations: List[str] = None


@dataclass
class ResearchQuery:
    query: str
    domain: str
    research_type: str
    depth_level: str
    time_range: Tuple[datetime, datetime]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    methodology_focus: List[str]
    quality_threshold: float


@dataclass
class ResearchInsight:
    insight_type: str
    confidence: float
    evidence_count: int
    description: str
    supporting_docs: List[str]
    implications: List[str]
    recommended_actions: List[str]


@dataclass
class LiteratureMap:
    central_concepts: List[str]
    concept_clusters: dict
    methodological_approaches: dict
    temporal_trends: dict
    citation_network: dict
    research_gaps: List[str]
    emerging_themes: List[str]


class ResearchEmbeddingEngine:
    def __init__(self, model_type: str = 'research'):
        self.model_type = model_type
        self.model = None
        self.dimension = None
        self._load_model()

    def _load_model(self):
        try:
            model_name = EMBEDDING_MODELS.get(self.model_type, EMBEDDING_MODELS['general'])
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print('Failed to load embedding model:', e)
            self.model = None
            self.dimension = 768

    def embed_documents(self, documents: List[ResearchDocument], chunk_strategy: str = 'section') -> List[ResearchDocument]:
        if not self.model:
            for doc in documents:
                doc.embedding = np.zeros(self.dimension)
            return documents
        chunk_size = 512
        overlap = 64
        for doc in documents:
            full_text = f"{doc.title}\n\n{doc.abstract}\n\n{doc.content}"
            chunks = self._create_semantic_chunks(full_text, chunk_size, overlap)
            if chunks:
                embeddings = self.model.encode(chunks, show_progress_bar=False)
                doc.embedding = np.mean(embeddings, axis=0)
            else:
                doc.embedding = np.zeros(self.dimension)
        return documents

    def _create_semantic_chunks(self, text: str, chunk_size: int, overlap: int):
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current = ''
        for p in paragraphs:
            if len(p) > chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ''
                sentences = [s.strip() + '.' for s in p.split('.') if s.strip()]
                tmp = ''
                for s in sentences:
                    if len(tmp + s) <= chunk_size:
                        tmp += ' ' + s
                    else:
                        if tmp:
                            chunks.append(tmp.strip())
                        tmp = s
                if tmp:
                    current = tmp
            else:
                if len(current + p) <= chunk_size:
                    current += '\n\n' + p
                else:
                    if current:
                        chunks.append(current.strip())
                    current = p
        if current:
            chunks.append(current.strip())
        return [c for c in chunks if len(c) > 50]


class ARISCore:
    def __init__(self):
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        self.embedding_engine = ResearchEmbeddingEngine('research')
        self.nlp_processor = AdvancedNLPProcessor()
        self.knowledge_graph = KnowledgeGraphBuilder()
        self.analysis_engine = LiteratureAnalysisEngine(self.nlp_processor, self.knowledge_graph)
        self.database_path = DATABASE_PATH
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS research_documents (id TEXT PRIMARY KEY, title TEXT, content TEXT, abstract TEXT, authors TEXT, keywords TEXT, publication_date DATETIME, source_type TEXT, citation_count INTEGER, impact_score REAL, metadata TEXT, embedding_path TEXT, entities TEXT, concepts TEXT, methodology TEXT, findings TEXT, limitations TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

    def ingest_research_document(self, file_content: str, filename: str, metadata: dict = None):
        metadata = metadata or {}
        doc_id = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()
        sections = parse_document_structure(file_content)
        title = metadata.get('title') or extract_title(file_content) or filename
        abstract = metadata.get('abstract') or extract_abstract(file_content)
        authors = metadata.get('authors') or extract_authors(file_content)
        entities = self.nlp_processor.extract_entities(file_content)
        key_phrases = self.nlp_processor.extract_key_phrases(file_content)
        methodology = self.nlp_processor.analyze_methodology(file_content)
        findings, limitations = self.nlp_processor.extract_findings_and_limitations(file_content)
        readability = self.nlp_processor.calculate_readability_metrics(file_content)
        impact_score = calculate_impact_score(metadata.get('citation_count', 0), readability, methodology['confidence'], len(file_content), len(key_phrases))
        doc = ResearchDocument(id=doc_id, title=title, content=file_content, abstract=abstract, authors=authors, keywords=key_phrases, publication_date=metadata.get('publication_date', datetime.now()), source_type=metadata.get('source_type', 'document'), citation_count=metadata.get('citation_count', 0), impact_score=impact_score, metadata={**metadata, 'readability': readability, 'sections': sections}, entities=list({item for sublist in entities.values() for item in sublist}), concepts=key_phrases, methodology=methodology['primary_methodology'], findings=findings, limitations=limitations)
        doc = self.embedding_engine.embed_documents([doc])[0]
        self._store_document(doc)
        return doc

    def _store_document(self, doc: ResearchDocument):
        embedding_path = f"{EMBEDDINGS_DIR}/{doc.id}.pkl"
        with open(embedding_path, 'wb') as f:
            pickle.dump(doc.embedding, f)
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute('''INSERT OR REPLACE INTO research_documents (id, title, content, abstract, authors, keywords, publication_date, source_type, citation_count, impact_score, metadata, embedding_path, entities, concepts, methodology, findings, limitations, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (doc.id, doc.title, doc.content, doc.abstract, json.dumps(doc.authors), json.dumps(doc.keywords), doc.publication_date.isoformat() if doc.publication_date else None, doc.source_type, doc.citation_count, doc.impact_score, json.dumps(doc.metadata, default=str), embedding_path, json.dumps(doc.entities), json.dumps(doc.concepts), doc.methodology, json.dumps(doc.findings), json.dumps(doc.limitations), datetime.now()))
        conn.commit()
        conn.close()

    def get_all_documents(self) -> List[ResearchDocument]:
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM research_documents ORDER BY impact_score DESC')
        rows = cursor.fetchall()
        conn.close()
        documents = []
        for row in rows:
            emb = None
            if row[11] and os.path.exists(row[11]):
                with open(row[11], 'rb') as f:
                    emb = pickle.load(f)
            doc = ResearchDocument(id=row[0], title=row[1], content=row[2], abstract=row[3], authors=json.loads(row[4]) if row[4] else [], keywords=json.loads(row[5]) if row[5] else [], publication_date=datetime.fromisoformat(row[6]) if row[6] else datetime.now(), source_type=row[7], citation_count=row[8] or 0, impact_score=row[9] or 0.0, metadata=json.loads(row[10]) if row[10] else {}, embedding=emb, entities=json.loads(row[12]) if row[12] else [], concepts=json.loads(row[13]) if row[13] else [], methodology=row[14], findings=json.loads(row[15]) if row[15] else [], limitations=json.loads(row[16]) if row[16] else [])
            documents.append(doc)
        return documents

    def conduct_research(self, query: ResearchQuery):
        all_documents = self.get_all_documents()
        if not all_documents:
            return {'error': 'No documents in research database'}
        kg = self.knowledge_graph.build_graph(all_documents)
        review_results = self.analysis_engine.conduct_systematic_review(all_documents, query)
        literature_map = {}  # For brevity; you can generate via analysis engine
        insights = []
        return {'query_id': hashlib.md5(f"{query.query}{time.time()}".encode()).hexdigest(),'systematic_review': review_results,'literature_map': literature_map,'research_insights': insights,'knowledge_graph_stats': {'nodes': kg.number_of_nodes(),'edges': kg.number_of_edges(),'density': nx.density(kg)}}