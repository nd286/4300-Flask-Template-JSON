import nltk
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn

def wordnet_normalize(token):
    synsets = wn.synsets(token)
    for syn in synsets:
        for hyper in syn.hypernyms():
            hyper_names = hyper.lemma_names()
            if "dairy_product" in hyper_names or "dairy" in hyper_names:
                return "dairy"
    return token

def wordnet_improved_tokenize(text: str):
    text = text.lower().strip()
    text = re.sub(r'\bnon[\s-]+([a-z0-9]+)\b', r'not_\1', text)
    text = re.sub(r'\bnot[\s-]+([a-z0-9]+)\b', r'not_\1', text)
    text = re.sub(r'\b([a-z0-9]+)[\s-]+free\b', r'not_\1', text)
    tokens = re.findall(r"\w+", text)
    tokens = [wordnet_normalize(token) for token in tokens]
    negation_words = {"not", "no", "never", "none", "without"}
    result = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in negation_words:
            result.append(token)
            if i + 1 < len(tokens):
                result.append("not_" + tokens[i + 1])
                skip_next = True
        else:
            result.append(token)
    return result

def improved_tokenize(text: str):
    return wordnet_improved_tokenize(text)

def build_field_svd(corpus, n_components=300):
    vectorizer = TfidfVectorizer(tokenizer=improved_tokenize, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    n_comp = min(n_components, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
    svd_model = TruncatedSVD(n_components=n_comp, random_state=42)
    doc_vectors = svd_model.fit_transform(tfidf_matrix)
    return vectorizer, svd_model, doc_vectors

def build_composite_svd_models(documents, n_components=100):
    desc_corpus = [doc.get("description", "") for doc in documents]
    subhead_corpus = [doc.get("subhead", "").strip() if doc.get("subhead", "").strip() != "" else doc.get("title", "") for doc in documents]
    ingr_corpus = [doc.get("ingredients_y", "") for doc in documents]
    reviews_corpus = [doc.get("text", "") for doc in documents]
    models = {}
    models["description"] = build_field_svd(desc_corpus, n_components)
    models["subhead"] = build_field_svd(subhead_corpus, n_components)
    models["ingredients"] = build_field_svd(ingr_corpus, n_components)
    models["reviews"] = build_field_svd(reviews_corpus, n_components)
    return models

def query_composite_svd_similarity(query, models, weights):
    if not query.strip():
        return np.array([])
    composite_scores = None
    for field, (vectorizer, svd_model, doc_vectors) in models.items():
        query_tfidf = vectorizer.transform([query])
        query_vec = svd_model.transform(query_tfidf)
        field_scores = cosine_similarity(query_vec, doc_vectors).flatten()
        weighted_scores = weights.get(field, 0) * field_scores
        if composite_scores is None:
            composite_scores = weighted_scores
        else:
            composite_scores += weighted_scores
    return composite_scores







