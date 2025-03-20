import math
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_vectorizer(corpus: List[str]) -> TfidfVectorizer:
    """
    Build and fit a TfidfVectorizer on the given corpus.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(corpus)
    return vectorizer

def compute_cosine_similarity(query: str, text: str, vectorizer: TfidfVectorizer) -> float:
    """
    Compute cosine similarity between the query and a given text using the provided vectorizer.
    """
    query_vec = vectorizer.transform([query])
    text_vec = vectorizer.transform([text])
    sim = cosine_similarity(query_vec, text_vec)
    return sim[0][0]

def compute_combined_score(query: str, description: str, reviews: List[str],
                           vectorizer: TfidfVectorizer) -> float:
    """
    Compute a combined cosine similarity score for a flavor based on its description and reviews.
    Overall score = 0.8 * (cosine similarity between query and description) +
                    0.2 * (average cosine similarity between query and each review)
    """
    desc_score = compute_cosine_similarity(query, description, vectorizer)
    
    review_scores = []
    for review in reviews:
        if review.strip():
            review_scores.append(compute_cosine_similarity(query, review, vectorizer))
    avg_review_score = sum(review_scores) / len(review_scores) if review_scores else 0.0
    
    combined_score = 0.8 * desc_score + 0.2 * avg_review_score
    return combined_score

