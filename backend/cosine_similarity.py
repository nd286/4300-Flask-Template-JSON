import math
import re
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

def tokenize(text: str) -> List[str]:
    """
    A simple tokenizer that lowercases and splits on non-alphanumeric characters.
    No NLTK needed.
    """
    text = text.lower()
    # Replace any sequence of non-alphanumeric characters with a space.
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    tokens = text.split()
    return tokens

def build_inverted_index(docs: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build an inverted index from the 'description' field of each document.
    Each document is expected to have a 'description' key.
    Returns a dict mapping from term -> list of (doc_id, term_frequency).
    """
    inv_index = defaultdict(list)
    for doc_id, doc in enumerate(docs):
        description = doc.get('description', '')
        tokens = tokenize(description)
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            inv_index[token].append((doc_id, count))
    # Sort postings by doc_id for consistency
    for token in inv_index:
        inv_index[token].sort(key=lambda x: x[0])
    return dict(inv_index)

def compute_idf(inv_index: Dict[str, List[Tuple[int, int]]], n_docs: int,
                min_df: int = 1, max_df_ratio: float = 1.0) -> Dict[str, float]:
    """
    Compute IDF values (log base 2) for terms in the inverted index.
    - min_df: words must appear in at least this many documents
    - max_df_ratio: words must appear in at most this fraction of docs
    """
    idf = {}
    for term, postings in inv_index.items():
        df = len(postings)
        if df < min_df or (df / n_docs) > max_df_ratio:
            continue
        # IDF with log base 2
        idf[term] = math.log2(n_docs / df)
    return idf

def compute_doc_norms(inv_index: Dict[str, List[Tuple[int, int]]],
                      idf: Dict[str, float],
                      n_docs: int) -> np.ndarray:
    """
    Precompute the Euclidean norm (L2 norm) of each document's TF-IDF vector.
    Returns an array of length n_docs.
    """
    norms = np.zeros(n_docs)
    for term, postings in inv_index.items():
        if term in idf:
            weight = idf[term]
            for doc_id, count in postings:
                # Accumulate squared weight for doc_id
                norms[doc_id] += (count * weight) ** 2
    return np.sqrt(norms)

def accumulate_dot_scores(query_counts: Dict[str, int],
                          inv_index: Dict[str, List[Tuple[int, int]]],
                          idf: Dict[str, float]) -> Dict[int, float]:
    """
    Compute the dot product between the query vector and each document vector (term-at-a-time).
    Returns doc_scores: a dict of { doc_id -> dot_score }.
    """
    doc_scores = defaultdict(float)
    for term, q_count in query_counts.items():
        if term in inv_index and term in idf:
            query_weight = q_count * idf[term]
            for doc_id, doc_count in inv_index[term]:
                doc_weight = doc_count * idf[term]
                doc_scores[doc_id] += query_weight * doc_weight
    return dict(doc_scores)

def index_search(query: str,
                 docs: List[Dict],
                 inv_index: Dict[str, List[Tuple[int, int]]],
                 idf: Dict[str, float],
                 doc_norms: np.ndarray) -> List[Tuple[float, int]]:
    """
    Compute cosine similarity for 'query' against each document's 'description'.
    Returns a sorted list of (score, doc_id) by descending score.
    """
    # Tokenize and count terms in the query
    tokens = tokenize(query)
    query_counts = Counter(tokens)

    # Compute the dot products for all docs
    dot_scores = accumulate_dot_scores(query_counts, inv_index, idf)

    # Compute the query's norm
    query_norm_sq = 0.0
    for term, q_count in query_counts.items():
        if term in idf:
            query_norm_sq += (q_count * idf[term]) ** 2
    query_norm = math.sqrt(query_norm_sq)

    # Compute cosine similarity
    results = []
    for doc_id, dot in dot_scores.items():
        if doc_norms[doc_id] > 0 and query_norm > 0:
            cos_sim = dot / (doc_norms[doc_id] * query_norm)
        else:
            cos_sim = 0
        results.append((cos_sim, doc_id))

    # Sort by highest score first
    results.sort(key=lambda x: x[0], reverse=True)
    return results
