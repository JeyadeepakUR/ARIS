from collections import defaultdict, Counter
from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timedelta


class LiteratureAnalysisEngine:
    def __init__(self, nlp_processor, knowledge_graph):
        self.nlp = nlp_processor
        self.kg = knowledge_graph

    def conduct_systematic_review(self, documents: List[object], query: object) -> Dict[str, Any]:
        filtered_docs = self._filter_documents(documents, query)
        quality_scores = self._assess_document_quality(filtered_docs)
        themes = self._extract_themes(filtered_docs)
        methods_analysis = self._analyze_methodologies(filtered_docs)
        temporal_trends = self._analyze_temporal_trends(filtered_docs)
        synthesis = self._synthesize_findings(filtered_docs, themes)
        gaps = self._identify_research_gaps(filtered_docs, themes)
        return {
            'filtered_documents': len(filtered_docs),
            'quality_distribution': quality_scores,
            'themes': themes,
            'methodological_approaches': methods_analysis,
            'temporal_trends': temporal_trends,
            'synthesis': synthesis,
            'research_gaps': gaps,
            'recommendations': self._generate_recommendations(themes, gaps, methods_analysis)
        }

    def _filter_documents(self, documents, query):
        filtered = []
        for doc in documents:
            if query.time_range:
                if not (query.time_range[0] <= doc.publication_date <= query.time_range[1]):
                    continue
            if doc.impact_score < query.quality_threshold:
                continue
            if query.inclusion_criteria:
                doc_text = f"{doc.title} {doc.abstract} {' '.join(doc.keywords or [])}".lower()
                if not any(criterion.lower() in doc_text for criterion in query.inclusion_criteria):
                    continue
            if query.exclusion_criteria:
                doc_text = f"{doc.title} {doc.abstract} {' '.join(doc.keywords or [])}".lower()
                if any(criterion.lower() in doc_text for criterion in query.exclusion_criteria):
                    continue
            filtered.append(doc)
        return filtered

    def _assess_document_quality(self, documents):
        quality_bins = {'high': 0, 'medium': 0, 'low': 0}
        for doc in documents:
            if doc.impact_score >= 0.8:
                quality_bins['high'] += 1
            elif doc.impact_score >= 0.5:
                quality_bins['medium'] += 1
            else:
                quality_bins['low'] += 1
        return quality_bins

    def _extract_themes(self, documents):
        if not documents:
            return {}
        all_concepts = []
        for doc in documents:
            all_concepts.extend(doc.keywords or [])
            all_concepts.extend(doc.concepts or [])
        theme_freq = Counter(all_concepts)
        themes = {}
        processed = set()
        for concept, freq in theme_freq.most_common(20):
            if concept not in processed:
                related = [c for c, f in theme_freq.items() if c != concept and self._concepts_similar(concept, c)]
                theme_docs = [doc for doc in documents if concept in (doc.keywords or []) + (doc.concepts or [])]
                themes[concept] = {
                    'frequency': freq,
                    'related_concepts': related[:5],
                    'document_count': len(theme_docs),
                    'avg_impact': np.mean([doc.impact_score for doc in theme_docs]) if theme_docs else 0
                }
                processed.add(concept)
                processed.update(related)
        return themes

    def _concepts_similar(self, c1, c2):
        w1 = set(c1.lower().split())
        w2 = set(c2.lower().split())
        inter = len(w1.intersection(w2))
        union = len(w1.union(w2))
        return (inter / union) > 0.5 if union > 0 else False

    def _analyze_methodologies(self, documents):
        method_counts = defaultdict(int)
        method_evolution = defaultdict(list)
        for doc in documents:
            if doc.methodology:
                method_counts[doc.methodology] += 1
                method_evolution[doc.methodology].append(doc.publication_date)
        total_docs = len(documents)
        diversity = len(method_counts) / total_docs if total_docs > 0 else 0
        return {
            'method_distribution': dict(method_counts),
            'methodological_diversity': diversity,
            'evolution': {m: sorted(dates) for m, dates in method_evolution.items()},
            'dominant_method': max(method_counts.items(), key=lambda x: x[1])[0] if method_counts else None
        }

    def _analyze_temporal_trends(self, documents):
        if not documents:
            return {}
        dates = [doc.publication_date for doc in documents if doc.publication_date]
        dates.sort()
        annual_counts = defaultdict(int)
        for d in dates:
            annual_counts[d.year] += 1
        annual_impact = defaultdict(list)
        for doc in documents:
            if doc.publication_date:
                annual_impact[doc.publication_date.year].append(doc.impact_score)
        avg_annual_impact = {year: np.mean(scores) for year, scores in annual_impact.items()}
        return {'publication_timeline': dict(annual_counts), 'impact_timeline': avg_annual_impact, 'total_span_years': (max(dates).year - min(dates).year) if dates else 0, 'peak_year': max(annual_counts.items(), key=lambda x: x[1])[0] if annual_counts else None}

    def _synthesize_findings(self, documents, themes):
        synthesis = {'key_findings': [], 'consensus_areas': [], 'controversial_areas': [], 'methodological_insights': [], 'theoretical_contributions': []}
        all_findings = []
        for doc in documents:
            if doc.findings:
                all_findings.extend(doc.findings)
        findings_freq = Counter(all_findings)
        synthesis['key_findings'] = [f for f, _ in findings_freq.most_common(10)]
        consensus_threshold = 0.7
        for theme, data in themes.items():
            if data['avg_impact'] > consensus_threshold:
                synthesis['consensus_areas'].append(theme)
        return synthesis

    def _identify_research_gaps(self, documents, themes):
        gaps = []
        theme_threshold = 3
        for theme, data in themes.items():
            if data['document_count'] < theme_threshold:
                gaps.append(f"Under-researched area: {theme}")
        methods = [doc.methodology for doc in documents if doc.methodology]
        method_counts = Counter(methods)
        expected_methods = ['quantitative', 'qualitative', 'mixed_methods', 'experimental']
        for method in expected_methods:
            if method_counts.get(method, 0) < len(documents) * 0.1:
                gaps.append(f"Methodological gap: Limited {method} studies")
        dates = [doc.publication_date for doc in documents if doc.publication_date]
        if dates:
            recent_cutoff = datetime.now() - timedelta(days=365)
            recent_docs = [d for d in dates if d > recent_cutoff]
            if len(recent_docs) < len(dates) * 0.2:
                gaps.append('Temporal gap: Limited recent research')
        return gaps

    def _generate_recommendations(self, themes, gaps, methods_analysis):
        recommendations = []
        high_impact = [t for t, d in themes.items() if d['avg_impact'] > 0.7]
        if high_impact:
            recommendations.append(f"Priority research areas: {', '.join(high_impact[:3])}")
        if methods_analysis.get('methodological_diversity', 0) < 0.5:
            recommendations.append('Increase methodological diversity in future studies')
        for gap in gaps[:3]:
            recommendations.append(f"Address {gap.lower()}")
        return recommendations
