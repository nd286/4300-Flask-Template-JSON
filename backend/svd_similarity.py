# svd_similarity.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def build_field_svd(corpus, n_components=50):
    """
    Build the SVD model for a given corpus.
    
    Args:
        corpus (List[str]): List of document strings.
        n_components (int): Number of SVD components to use.
        
    Returns:
        tuple: (vectorizer, svd_model, doc_vectors)
    """
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    # Ensure we don't set n_components too high
    n_comp = min(n_components, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
    svd_model = TruncatedSVD(n_components=n_comp, random_state=42)
    doc_vectors = svd_model.fit_transform(tfidf_matrix)
    return vectorizer, svd_model, doc_vectors

def build_composite_svd_models(documents, n_components=50):
    """
    Given a list of documents (each a dict), extract fields and build SVD models for each.
    
    Expected fields:
      - "description"
      - "subhead"
      - "ingredients_y" (used as ingredients)
      - "text" (used as reviews)
      
    Args:
        documents (List[Dict]): List of document dictionaries.
        n_components (int): Number of SVD components to use for each field.
        
    Returns:
        dict: Mapping from field name to a tuple (vectorizer, svd_model, doc_vectors)
    """
    # Extract field corpora from the documents.
    desc_corpus = [doc.get("description", "") for doc in documents]
    subhead_corpus = [doc.get("subhead", "") for doc in documents]
    ingr_corpus = [doc.get("ingredients_y", "") for doc in documents]
    reviews_corpus = [doc.get("text", "") for doc in documents]
    
    models = {}
    models["description"] = build_field_svd(desc_corpus, n_components)
    models["subhead"] = build_field_svd(subhead_corpus, n_components)
    models["ingredients"] = build_field_svd(ingr_corpus, n_components)
    models["reviews"] = build_field_svd(reviews_corpus, n_components)
    
    return models

def query_composite_svd_similarity(query, models, weights):
    """
    Compute a composite similarity score for a query by evaluating each field separately
    and then combining the scores via the provided weights.
    
    Args:
        query (str): The search query.
        models (dict): A dictionary with keys "description", "subhead", "ingredients", "reviews".
                       Each value is a tuple (vectorizer, svd_model, doc_vectors).
        weights (dict): Weights for each field, e.g.:
                        {"description": 0.4, "subhead": 0.3, "ingredients": 0.1, "reviews": 0.2}
                        
    Returns:
        numpy.ndarray: Array of composite similarity scores for all documents.
    """
    if not query.strip():
        return np.array([])
    
    composite_scores = None
    for field, (vectorizer, svd_model, doc_vectors) in models.items():
        # Transform the query for the current field.
        query_tfidf = vectorizer.transform([query])
        query_vec = svd_model.transform(query_tfidf)
        # Compute cosine similarity between the query and all document vectors for this field.
        field_scores = cosine_similarity(query_vec, doc_vectors).flatten()
        weighted_scores = weights.get(field, 0) * field_scores
        if composite_scores is None:
            composite_scores = weighted_scores
        else:
            composite_scores += weighted_scores
    return composite_scores
