import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

def tokenize(text: str) -> List[str]:
    """
    A custom tokenizer that lowercases text, replaces non-alphanumeric characters with spaces,
    and splits on whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return text.split()

def build_inverted_index(docs: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build an inverted index from the combined text of each document.
    Each document is expected to have 'description' and/or 'text' (reviews) fields.
    Returns a dict mapping each term to a sorted list of (doc_id, term frequency) tuples.
    """
    inv_idx = {}
    for doc_id, doc in enumerate(docs):
        combined = doc.get('description', '') + " " + doc.get('text', '')
        tokens = tokenize(combined)
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if token not in inv_idx:
                inv_idx[token] = []
            inv_idx[token].append((doc_id, count))
    for token in inv_idx:
        inv_idx[token].sort(key=lambda x: x[0])
    return inv_idx

def compute_idf(inv_idx: Dict[str, List[Tuple[int, int]]], n_docs: int,
                min_df: int = 10, max_df_ratio: float = 0.95) -> Dict[str, float]:
    """
    Compute IDF values (using log base 2) from the inverted index.
    Terms that appear in fewer than min_df documents or in more than max_df_ratio of documents are ignored.
    """
    idf = {}
    for term, postings in inv_idx.items():
        df = len(postings)
        if df < min_df or (df / n_docs) > max_df_ratio:
            continue
        idf[term] = math.log(n_docs / (df + 1), 2)
    return idf

def compute_doc_norms(inv_idx: Dict[str, List[Tuple[int, int]]],
                      idf: Dict[str, float],
                      n_docs: int) -> List[float]:
    """
    Precompute the Euclidean (L2) norm for each document's TF–IDF vector.
    Returns a list of norms, one per document.
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
    Compute the cosine similarity between a query and a given text using IDF weighting.
    For each term, we weight its frequency by its IDF.
    """
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    q_counts = Counter(query_tokens)
    t_counts = Counter(text_tokens)
    
    dot = 0.0
    for term, q_freq in q_counts.items():
        if term in t_counts and term in idf:
            dot += q_freq * t_counts[term] * (idf[term] ** 2)
    
    norm_q = math.sqrt(sum((q_counts[term] * idf.get(term, 0))**2 for term in q_counts))
    norm_t = math.sqrt(sum((t_counts[term] * idf.get(term, 0))**2 for term in t_counts))
    
    if norm_q == 0 or norm_t == 0:
        return 0.0
    return dot / (norm_q * norm_t)

def compute_combined_score(query: str, description: str, reviews: List[str], idf: Dict[str, float]) -> float:
    """
    Compute the overall cosine similarity score for a flavor:
       overall_score = 0.8 * (cosine similarity between query and description) +
                       0.2 * (average cosine similarity between query and each review)
    """
    desc_score = cosine_sim_idf(query, description, idf)
    review_scores = [cosine_sim_idf(query, review, idf) for review in reviews if review.strip()]
    avg_review_score = sum(review_scores) / len(review_scores) if review_scores else 0.0
    return 0.8 * desc_score + 0.2 * avg_review_score
