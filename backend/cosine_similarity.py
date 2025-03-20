import math
import re
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

def tokenize(text: str) -> List[str]:
    """
    A simple tokenizer that lowercases and splits on non-alphanumeric characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    tokens = text.split()
    return tokens

def build_inverted_index(docs: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build an inverted index from the 'description' field of each document.
    (In our original code this was used to compute global IDF.)
    """
    inv_index = defaultdict(list)
    for doc_id, doc in enumerate(docs):
        # For the global idf, we index over the combined text.
        combined = doc.get('description', '') + " " + doc.get('text', '')
        tokens = tokenize(combined)
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            inv_index[token].append((doc_id, count))
    for token in inv_index:
        inv_index[token].sort(key=lambda x: x[0])
    return dict(inv_index)

def compute_idf(inv_index: Dict[str, List[Tuple[int, int]]], n_docs: int,
                min_df: int = 1, max_df_ratio: float = 1.0) -> Dict[str, float]:
    """
    Compute IDF values (log base 2) for terms in the inverted index.
    """
    idf = {}
    for term, postings in inv_index.items():
        df = len(postings)
        if df < min_df or (df / n_docs) > max_df_ratio:
            continue
        idf[term] = math.log2(n_docs / df)
    return idf

def compute_doc_norms(inv_index: Dict[str, List[Tuple[int, int]]],
                      idf: Dict[str, float],
                      n_docs: int) -> np.ndarray:
    """
    Precompute the Euclidean norm (L2 norm) of each document's TF-IDF vector.
    """
    norms = np.zeros(n_docs)
    for term, postings in inv_index.items():
        if term in idf:
            weight = idf[term]
            for doc_id, count in postings:
                norms[doc_id] += (count * weight) ** 2
    return np.sqrt(norms)

def accumulate_dot_scores(query_counts: Dict[str, int],
                          inv_index: Dict[str, List[Tuple[int, int]]],
                          idf: Dict[str, float]) -> Dict[int, float]:
    """
    Compute the dot product between the query vector and each document's TF-IDF vector.
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
    (This function is kept for legacy use.)
    """
    tokens = tokenize(query)
    query_counts = Counter(tokens)
    dot_scores = accumulate_dot_scores(query_counts, inv_index, idf)
    query_norm_sq = sum((q_count * idf.get(term, 0)) ** 2 for term, q_count in query_counts.items() if term in idf)
    query_norm = math.sqrt(query_norm_sq)
    
    results = []
    for doc_id, dot in dot_scores.items():
        if doc_norms[doc_id] > 0 and query_norm > 0:
            cos_sim = dot / (doc_norms[doc_id] * query_norm)
        else:
            cos_sim = 0
        results.append((cos_sim, doc_id))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results
