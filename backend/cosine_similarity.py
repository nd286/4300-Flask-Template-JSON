import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

def tokenize(text: str) -> List[str]:
    """
    A custom tokenizer that lowercases the text, replaces non-alphanumeric 
    characters with spaces, and splits on whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return text.split()

def build_inverted_index(docs: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build an inverted index from a list of documents.
    Each document is assumed to be a string (e.g. the combined text of description and texts).
    Returns a dict mapping each term to a sorted list of (doc_id, term frequency) tuples.
    """
    inv_idx = {}
    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        counts = Counter(tokens)
        for token, count in counts.items():
            if token not in inv_idx:
                inv_idx[token] = []
            inv_idx[token].append((doc_id, count))
    for token in inv_idx:
        inv_idx[token].sort(key=lambda x: x[0])
    return inv_idx

def compute_idf(inv_idx: Dict[str, List[Tuple[int, int]]], n_docs: int, 
                min_df: int = 1, max_df_ratio: float = 1.0) -> Dict[str, float]:
    """
    Compute IDF values (using log base 2) from the inverted index.
    Terms that appear in fewer than min_df docs or in more than max_df_ratio of docs are ignored.
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
    
    norm_q = math.sqrt(sum((q_counts[t] * idf.get(t, 0))**2 for t in q_counts))
    norm_t = math.sqrt(sum((t_counts[t] * idf.get(t, 0))**2 for t in t_counts))
    
    if norm_q == 0 or norm_t == 0:
        return 0.0
    return dot / (norm_q * norm_t)

def compute_combined_score(query: str, description: str, texts: List[str], 
                           idf: Dict[str, float]) -> float:
    """
    Compute the overall cosine similarity score for a flavor:
       overall_score = 0.8 * cosine_sim(query, description) +
                       0.2 * (average cosine_sim(query, each text))
    Here, texts is a list of strings from the "text" field.
    """
    desc_score = cosine_sim_idf(query, description, idf)
    text_scores = [cosine_sim_idf(query, text, idf) for text in texts if text.strip()]
    avg_text_score = sum(text_scores) / len(text_scores) if text_scores else 0.0
    return 0.8 * desc_score + 0.2 * avg_text_score

