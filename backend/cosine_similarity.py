import math
import re
from collections import Counter
from typing import List, Dict

def tokenize(text: str) -> List[str]:
    """
    A simple tokenizer that lowercases and splits on non-alphanumeric characters.
    """
    text = text.lower()
    # Replace non-alphanumeric characters with a space.
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    tokens = text.split()
    return tokens

def cosine_sim_for_text(query: str, text: str) -> float:
    """
    Compute cosine similarity between the query and a given text using bag-of-words counts.
    (This is a simple implementation without TF-IDF weighting.)
    """
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    query_counts = Counter(query_tokens)
    text_counts = Counter(text_tokens)
    
    # Compute dot product.
    dot = sum(query_counts[term] * text_counts.get(term, 0) for term in query_counts)
    
    # Compute Euclidean norms.
    query_norm = math.sqrt(sum(val**2 for val in query_counts.values()))
    text_norm = math.sqrt(sum(val**2 for val in text_counts.values()))
    
    if query_norm > 0 and text_norm > 0:
        return dot / (query_norm * text_norm)
    else:
        return 0.0

def compute_combined_score(query: str, description: str, reviews: List[str]) -> float:
    """
    Compute a combined cosine similarity score for a flavor:
      overall_score = 0.8 * (cosine similarity between query and description)
                    + 0.2 * (average cosine similarity between query and each review)
    """
    desc_score = cosine_sim_for_text(query, description)
    
    review_scores = [cosine_sim_for_text(query, review) for review in reviews if review.strip()]
    avg_review_score = sum(review_scores) / len(review_scores) if review_scores else 0.0
    
    combined_score = 0.8 * desc_score + 0.2 * avg_review_score
    return combined_score
