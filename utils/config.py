from datetime import datetime


EMBEDDING_MODELS = {
"research": "sentence-transformers/allenai-specter",
"general": "sentence-transformers/all-mpnet-base-v2",
"multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
}


DATABASE_PATH = "aris_research_db.sqlite"
EMBEDDINGS_DIR = "aris_embeddings"
KNOWLEDGE_GRAPH_DIR = "knowledge_graphs"
ANALYSIS_CACHE_DIR = "analysis_cache"


CHUNK_SIZES = {"paragraph": 256, "section": 512, "document": 1024, "detailed": 128}
OVERLAP_RATIOS = {"low": 0.1, "medium": 0.2, "high": 0.3}
SIMILARITY_THRESHOLDS = {"strict": 0.8, "moderate": 0.6, "loose": 0.4}
RESEARCH_DEPTH_LEVELS = {"surface": 3, "deep": 7, "exhaustive": 15}


DEFAULT_PUBLICATION_DATE = datetime.now()


# UI related constants (used by app.py)
PAGE_TITLE = "ARIS - Autonomous Research Intelligence System"