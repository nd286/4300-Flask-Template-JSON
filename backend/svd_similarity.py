# svd_similarity.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def build_svd_model(corpus, n_components=50):
    """
    Build the TF–IDF matrix from the corpus and reduce its dimensionality using Truncated SVD.

    Args:
        corpus (List[str]): List of document texts.
        n_components (int): Number of SVD components (default is 50).

    Returns:
        vectorizer: Fitted TfidfVectorizer.
        svd_model: Fitted TruncatedSVD model.
        doc_vectors: Array of document vectors in the reduced space.
    """
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Make sure n_components does not exceed the number of available features
    n_components = min(n_components, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1

    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    doc_vectors = svd_model.fit_transform(tfidf_matrix)
    return vectorizer, svd_model, doc_vectors

def query_svd_similarity(query, vectorizer, svd_model, doc_vectors):
    """
    Calculate cosine similarity between the query (after SVD projection) and all document vectors.

    Args:
        query (str): The search query text.
        vectorizer: The TfidfVectorizer fitted on the corpus.
        svd_model: The TruncatedSVD model fitted on the TF–IDF matrix.
        doc_vectors: The reduced-dimension document vectors.

    Returns:
        A numpy array of similarity scores for each document.
    """
    if not query.strip():
        return np.array([])
    query_tfidf = vectorizer.transform([query])
    query_vec = svd_model.transform(query_tfidf)
    sim_scores = cosine_similarity(query_vec, doc_vectors)
    return sim_scores.flatten()
