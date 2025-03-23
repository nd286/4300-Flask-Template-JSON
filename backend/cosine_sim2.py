from collections import defaultdict
import numpy as np
import re
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import math


def tokenize(text: str) -> List[str]:
    """
    A custom tokenizer that lowercases text, replaces non-alphanumeric 
    characters with spaces, and splits on whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return text.split()


def build_vectorizer(docs: List[str]) -> Tuple[TfidfVectorizer, any]:
    """
    Create and fit a TfidfVectorizer using the custom tokenizer.
    Returns the fitted vectorizer and the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix


def build_inverted_index(vectorizer: TfidfVectorizer, tfidf_matrix) -> dict:
    """
    Builds an inverted index from a fitted TfidfVectorizer and its TF-IDF matrix.
    Returns a dict mapping terms to a list of (doc_id, tfidf_value) tuples.
    """
    inv_index = defaultdict(list)
    terms = vectorizer.get_feature_names_out()
    term_to_index = vectorizer.vocabulary_

    for term in terms:
        term_index = term_to_index[term]
        column = tfidf_matrix[:, term_index]
        doc_indices = column.nonzero()[0]
        for doc_id in doc_indices:
            tfidf_val = column[doc_id, 0]
            inv_index[term].append((doc_id, tfidf_val))

    return dict(inv_index)


def cosine_sim_idf(query: str, text: str, vectorizer: TfidfVectorizer) -> float:
    """
    Compute cosine similarity between a query and a text using IDF weighting.
    Each term's frequency is weighted by its IDF.
    """
    query_vec = vectorizer.transform([query])
    text_vec = vectorizer.transform([text])

    q = query_vec.toarray()[0]
    t = text_vec.toarray()[0]

    dot = np.dot(q, t)

    norm_q = np.linalg.norm(q)
    norm_t = np.linalg.norm(t)

    if norm_q == 0 or norm_t == 0:
        return 0.0

    return dot / (norm_q * norm_t)


def compute_combined_score(query: str, description: str, subhead: str, ingredients: str,
                           idf: Dict[str, float]) -> float:
    """
    Compute an overall cosine similarity score for a flavor using IDF weighting.

    overall_score = 0.6 * cosine_sim(query, description) +
                    0.3 * cosine_sim(query, subhead) +
                    0.1 * cosine_sim(query, ingredients)
    """
    desc_score = cosine_sim_idf(query, description, idf)
    subhead_score = cosine_sim_idf(query, subhead, idf)
    ingr_score = cosine_sim_idf(query, ingredients, idf)
    return 0.4 * desc_score + 0.5 * subhead_score + 0.1 * ingr_score
