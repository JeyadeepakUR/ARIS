import math


def calculate_impact_score(citation_count: int, readability_metrics: dict,
                           methodology_confidence: float, content_length: int,
                           keywords_count: int) -> float:
    citation_score = min(math.log10(max(citation_count, 1)) / 3.0, 1.0)
    complexity_score = min((100 - readability_metrics.get('flesch_score', 50)) / 100.0, 1.0)
    tech_density = readability_metrics.get('technical_density', 0)
    length_score = min(content_length / 10000, 1.0)
    keyword_score = min(keywords_count / 20, 1.0)

    weights = {'citation': 0.3, 'complexity': 0.2, 'methodology': 0.2, 'length': 0.15, 'keywords': 0.1, 'technical': 0.05}

    impact_score = (
        weights['citation'] * citation_score +
        weights['complexity'] * complexity_score +
        weights['methodology'] * methodology_confidence +
        weights['length'] * length_score +
        weights['keywords'] * keyword_score +
        weights['technical'] * tech_density
    )

    return round(impact_score, 3)
