from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import re
from datetime import datetime
import math

import spacy
from textstat import flesch_reading_ease, syllable_count


class AdvancedNLPProcessor:
    def __init__(self, spacy_model: str = 'en_core_web_sm'):
        try:
            self.nlp = spacy.load(spacy_model)
            self.nlp.max_length = 5_000_000
        except Exception:
            self.nlp = None

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        if not self.nlp:
            return {"PERSON": [], "ORG": [], "GPE": [], "TECH": []}
        doc = self.nlp(text)
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        return dict(entities)

    def extract_key_phrases(self, text: str, max_phrases: int = 20) -> List[str]:
        if not self.nlp:
            return []
        doc = self.nlp(text)
        phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1 and len(chunk.text) > 5:
                phrases.append(chunk.text.lower().strip())
        phrase_freq = Counter(phrases)
        return [phrase for phrase, _ in phrase_freq.most_common(max_phrases)]

    def analyze_methodology(self, text: str) -> Dict[str, Any]:
        methodology_indicators = {
            "quantitative": ["statistical analysis", "regression", "correlation", "hypothesis testing", "sample size", "p-value"],
            "qualitative": ["interviews", "focus groups", "case study", "thematic analysis", "grounded theory"],
            "mixed_methods": ["triangulation", "sequential", "concurrent", "embedded design"],
            "experimental": ["randomized", "control group", "intervention", "treatment", "experiment"],
            "survey": ["questionnaire", "survey", "likert scale", "response rate"],
            "meta_analysis": ["meta-analysis", "systematic review", "effect size", "forest plot"]
        }
        text_lower = text.lower()
        methodology_scores = {}
        for method, indicators in methodology_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            methodology_scores[method] = score
        primary_method = max(methodology_scores.items(), key=lambda x: x[1])
        return {"primary_methodology": primary_method[0] if primary_method[1] > 0 else "unspecified",
                "methodology_scores": methodology_scores,
                "confidence": min(primary_method[1] / 3.0, 1.0)}

    def extract_findings_and_limitations(self, text: str) -> Tuple[List[str], List[str]]:
        if not self.nlp:
            return [], []
        doc = self.nlp(text)
        findings = []
        limitations = []
        finding_keywords = ["found", "discovered", "revealed", "showed", "demonstrated", "concluded"]
        limitation_keywords = ["limitation", "constraint", "weakness", "shortcoming", "restricted", "limited"]
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if any(k in sent_text.lower() for k in finding_keywords):
                findings.append(sent_text)
            elif any(k in sent_text.lower() for k in limitation_keywords):
                limitations.append(sent_text)
        return findings[:5], limitations[:3]

    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        try:
            return {
                "flesch_score": flesch_reading_ease(text),
                "avg_sentence_length": len(text.split()) / max(len(text.split('.')), 1),
                "complex_words_ratio": self._complex_words_ratio(text),
                "technical_density": self._technical_density(text)
            }
        except Exception:
            return {"flesch_score": 0, "avg_sentence_length": 0, "complex_words_ratio": 0, "technical_density": 0}

    def _complex_words_ratio(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        complex_count = sum(1 for word in words if syllable_count(word) >= 3)
        return complex_count / len(words)

    def _technical_density(self, text: str) -> float:
        technical_indicators = ["algorithm", "methodology", "analysis", "framework", "implementation", "evaluation", "optimization", "correlation", "significance", "hypothesis"]
        words = text.lower().split()
        if not words:
            return 0.0
        technical_count = sum(1 for word in words if any(ind in word for ind in technical_indicators))
        return technical_count / len(words)

# Lightweight document parsers

def parse_document_structure(content: str) -> Dict[str, str]:
    sections = {}
    section_patterns = {
        'abstract': r'(?i)\babstract\b.*?(?=\n\s*\n|\n\s*[A-Z])',
        'introduction': r'(?i)\b(introduction|background)\b.*?(?=\n\s*\n.*?\b(method|approach|literature|related work)\b)',
        'methodology': r'(?i)\b(method|methodology|approach|experimental setup)\b.*?(?=\n\s*\n.*?\b(result|finding|discussion)\b)',
        'results': r'(?i)\b(result|finding|experiment)\b.*?(?=\n\s*\n.*?\b(discussion|conclusion|limitation)\b)',
        'discussion': r'(?i)\b(discussion|analysis)\b.*?(?=\n\s*\n.*?\b(conclusion|limitation|reference)\b)',
        'conclusion': r'(?i)\b(conclusion|summary)\b.*?(?=\n\s*\n.*?\b(reference|acknowledgment|appendix)\b)',
    }
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        if match:
            sections[section_name] = match.group(0).strip()
    return sections


def extract_title(content: str) -> str:
    lines = content.split('\n')[:10]
    for line in lines:
        line = line.strip()
        if 10 <= len(line) <= 200 and not line.lower().startswith(('the', 'a', 'an', 'this', 'that')):
            return line
    return ''


def extract_abstract(content: str) -> str:
    abstract_match = re.search(r'(?i)\babstract\b[:\s]*\n?(.*?)(?=\n\s*\n|\n\s*[A-Z])', content, re.DOTALL)
    if abstract_match:
        return abstract_match.group(1).strip()
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    if paragraphs and 100 <= len(paragraphs[0]) <= 1000:
        return paragraphs[0]
    return ''


def extract_authors(content: str) -> List[str]:
    author_patterns = [r'(?i)authors?\s*: ?([^\.\n]+)', r'(?i)by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)']
    for pattern in author_patterns:
        match = re.search(pattern, content)
        if match:
            authors_str = match.group(1)
            authors = re.split(r'[,;&]', authors_str)
            return [a.strip() for a in authors if a.strip()]
    return []
