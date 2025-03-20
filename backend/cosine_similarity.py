import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

def tokenize(text: str) -> List[str]:
    """
    A custom tokenizer that lowercases text, replaces non-alphanumeric 
    characters with spaces, and splits on whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return text.split()

def build_inverted_index(docs: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build an inverted index from a list of documents.
    Each document is a string (the combined text of description and ingredients).
    Returns a dict mapping each term to a sorted list of (doc_id, term frequency) tuples.
    """
    inv_idx = defaultdict(list)
    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        counts = Counter(tokens)
        for token, count in counts.items():
            inv_idx[token].append((doc_id, count))
    for token in inv_idx:
        inv_idx[token].sort(key=lambda x: x[0])
    return dict(inv_idx)

def compute_idf(inv_idx: Dict[str, List[Tuple[int, int]]], n_docs: int, 
                min_df: int = 1, max_df_ratio: float = 1.0) -> Dict[str, float]:
    """
    Compute IDF values (using log base 2) from the inverted index.
    """
    idf = {}
    for term, postings in inv_idx.items():
        df = len(postings)
        if df < min_df or (df / n_docs) > max_df_ratio:
            continue
        idf[term] = math.log2(n_docs / df)
    return idf

def compute_doc_norms(inv_idx: Dict[str, List[Tuple[int, int]]],
                      idf: Dict[str, float],
                      n_docs: int) -> List[float]:
    """
    Compute the Euclidean (L2) norm for each document's TF–IDF vector.
    Returns a list of norms (one per document).
    """
    norms = [0.0] * n_docs
    for term, postings in inv_idx.items():
        if term in idf:
            w = idf[term]
            for doc_id, count in postings:
                norms[doc_id] += (count * w) ** 2
    return [math.sqrt(val) for val in norms]

def cosine_sim_idf(query: str, text: str, idf: Dict[str, float]) -> float:
    """
    Compute cosine similarity between a query and a text using IDF weighting.
    Each term's frequency is weighted by its IDF.
    """
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    q_counts = Counter(query_tokens)
    t_counts = Counter(text_tokens)
    
    dot = 0.0
    for term, q_freq in q_counts.items():
        if term in t_counts and term in idf:
            dot += q_freq * t_counts[term] * (idf[term] ** 2)
    
    norm_q = math.sqrt(sum((q_counts[t] * idf.get(t, 0)) ** 2 for t in q_counts))
    norm_t = math.sqrt(sum((t_counts[t] * idf.get(t, 0)) ** 2 for t in t_counts))
    
    if norm_q == 0 or norm_t == 0:
        return 0.0
    return dot / (norm_q * norm_t)

def compute_combined_score(query: str, description: str, ingredients: str, 
                           idf: Dict[str, float]) -> float:
    """
    Compute an overall cosine similarity score for a flavor using IDF weighting.
    overall_score = 0.8 * cosine_sim(query, description) +
                    0.2 * cosine_sim(query, ingredients)
    """
    desc_score = cosine_sim_idf(query, description, idf)
    ingr_score = cosine_sim_idf(query, ingredients, idf)
    return 0.8 * desc_score + 0.2 * ingr_score

