# bert_similarity.py
from sentence_transformers import SentenceTransformer, util
import torch

def compute_composite_embedding(model, doc, weights=None):
    """
    Compute a composite embedding for a document using BERT, weighting four fields:
       - description (from the "description" key)
       - subhead (from the "subhead" key)
       - ingredients (from the "ingredients_y" key)
       - reviews (from the "text" key)
    
    Args:
        model: An instance of SentenceTransformer.
        doc (dict): A document dictionary.
        weights (dict): Field weights (default is:
                        {"description": 0.4, "subhead": 0.3, "ingredients": 0.1, "reviews": 0.2})
    
    Returns:
        torch.Tensor: The composite embedding.
    """
    if weights is None:
        weights = {"description": 0.4, "subhead": 0.3, "ingredients": 0.1, "reviews": 0.2}
    
    # Get text for each field (or empty string if missing)
    desc = doc.get("description", "")
    subhead = doc.get("subhead", "")
    ingr = doc.get("ingredients_y", "")
    reviews = doc.get("text", "")
    
    # Compute individual field embeddings
    emb_desc = model.encode(desc, convert_to_tensor=True)
    emb_subhead = model.encode(subhead, convert_to_tensor=True)
    emb_ingr = model.encode(ingr, convert_to_tensor=True)
    emb_reviews = model.encode(reviews, convert_to_tensor=True)
    
    # Compute and return the weighted composite embedding.
    composite = (weights.get("description", 0) * emb_desc +
                 weights.get("subhead", 0) * emb_subhead +
                 weights.get("ingredients", 0) * emb_ingr +
                 weights.get("reviews", 0) * emb_reviews)
    return composite

def build_document_embeddings(documents, model_name='paraphrase-MiniLM-L6-v2', weights=None):
    """
    Build composite embeddings for a list of document dictionaries.
    
    Args:
        documents (list of dict): The documents.
        model_name (str): Name of the pre-trained sentence transformer model.
        weights (dict): Field weights.
    
    Returns:
        tuple: (model, embeddings)
            model: The SentenceTransformer instance.
            embeddings: A torch.Tensor with stacked composite embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = [compute_composite_embedding(model, doc, weights) for doc in documents]
    embeddings = torch.stack(embeddings)
    return model, embeddings

def query_bert_similarity(query, model, doc_embeddings):
    """
    Compute cosine similarity scores for a query against document embeddings.
    
    Args:
        query (str): The search query.
        model: The SentenceTransformer model.
        doc_embeddings (torch.Tensor): Composite embeddings for all documents.
    
    Returns:
        numpy.ndarray: 1D array of cosine similarity scores.
    """
    # Encode the query (using the same model) to obtain its embedding.
    emb_query = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb_query, doc_embeddings)
    # Convert the result to a numpy 1D array.
    return cosine_scores.cpu().numpy().flatten()
