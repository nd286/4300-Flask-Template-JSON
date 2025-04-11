# svd_similarity.py
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def improved_tokenize(text: str):
    """
    An improved tokenizer that handles negation.
    It lowercases the text, rewrites tokens with a "non-" prefix as negated tokens,
    and then splits on non-alphanumeric characters.
    
    For example:
      "non-dairy"  --> "not_dairy"
      
    Args:
        text (str): The input text.
        
    Returns:
        List[str]: A list of tokens.
    """
    # Lowercase the text.
    text = text.lower()
    # Rewrite occurrences of 'non-<word>' as 'not_<word>'
    text = re.sub(r'\bnon-([a-z0-9]+)\b', r'not_\1', text)
    # Replace any non-alphanumeric characters with spaces.
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return text.split()

def build_field_svd(corpus, n_components=50):
    """
    Build the SVD model for a given corpus using improved tokenization.
    
    Args:
        corpus (List[str]): List of document strings.
        n_components (int): Number of SVD components to use.
        
    Returns:
        tuple: (vectorizer, svd_model, doc_vectors)
    """
    # Use our improved_tokenize function by setting tokenizer and disable token_pattern.
    vectorizer = TfidfVectorizer(tokenizer=improved_tokenize, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    # Ensure n_components is not set higher than available features.
    n_comp = min(n_components, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
    svd_model = TruncatedSVD(n_components=n_comp, random_state=42)
    doc_vectors = svd_model.fit_transform(tfidf_matrix)
    return vectorizer, svd_model, doc_vectors

def build_composite_svd_models(documents, n_components=50):
    """
    Given a list of document dictionaries, build SVD models for each of the fields:
      - "description"
      - "subhead"
      - "ingredients_y" (used as the ingredients field)
      - "text" (used as the reviews field)
    
    Args:
        documents (List[Dict]): List of document dictionaries.
        n_components (int): Number of SVD components for each field.
    
    Returns:
        dict: Mapping from field name to a tuple (vectorizer, svd_model, doc_vectors)
    """
    # Extract corpora for each field.
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
    Compute a composite similarity score for a query by calculating cosine similarities
    for each field and then combining them based on provided weights.
    
    Args:
        query (str): The search query.
        models (dict): A dict with keys "description", "subhead", "ingredients", "reviews".
                       Each value is a tuple (vectorizer, svd_model, doc_vectors).
        weights (dict): Field weights (e.g., {"description": 0.4, "subhead": 0.3, "ingredients": 0.1, "reviews": 0.2}).
    
    Returns:
        numpy.ndarray: Composite similarity scores (one per document).
    """
    if not query.strip():
        return np.array([])
    
    composite_scores = None
    for field, (vectorizer, svd_model, doc_vectors) in models.items():
        # Transform the query into this field's vector space using improved_tokenize.
        query_tfidf = vectorizer.transform([query])
        query_vec = svd_model.transform(query_tfidf)
        # Compute cosine similarities.
        field_scores = cosine_similarity(query_vec, doc_vectors).flatten()
        weighted_scores = weights.get(field, 0) * field_scores
        if composite_scores is None:
            composite_scores = weighted_scores
        else:
            composite_scores += weighted_scores
    return composite_scores
