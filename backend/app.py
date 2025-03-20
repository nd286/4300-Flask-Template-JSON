import json
import os
import math
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from collections import defaultdict, Counter

from cosine_similarity import tokenize, build_inverted_index, compute_idf, compute_doc_norms

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load JSON data from init.json; it should have a top-level "flavors" key.
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

# Convert the DataFrame to a list of documents (each review instance)
docs = flavors_df.to_dict(orient='records')

# Build a global inverted index using all docs (for idf computation).
inv_index = build_inverted_index(docs)
n_docs = len(docs)
idf = compute_idf(inv_index, n_docs, min_df=1, max_df_ratio=1.0)
doc_norms = compute_doc_norms(inv_index, idf, n_docs)  # Not used in our new per-document computations

app = Flask(__name__)
CORS(app)

# Helper function: compute cosine similarity between a query and a given text using the global idf.
def cosine_sim_for_text(query: str, text: str, idf: dict) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    query_counts = Counter(query_tokens)
    text_counts = Counter(text_tokens)
    dot = 0.0
    for term, q_count in query_counts.items():
        if term in text_counts and term in idf:
            dot += q_count * text_counts[term] * (idf[term] ** 2)
    query_norm = math.sqrt(sum((q_count * idf.get(term, 0)) ** 2 for term, q_count in query_counts.items()))
    text_norm = math.sqrt(sum((text_counts[term] * idf.get(term, 0)) ** 2 for term in text_counts))
    if query_norm > 0 and text_norm > 0:
        return dot / (query_norm * text_norm)
    else:
        return 0.0

# Group documents by flavor title.
# For each unique flavor, assume the description is the same across duplicates.
unique_flavors = {}
for doc in docs:
    title = doc.get("title", "")
    if title not in unique_flavors:
        unique_flavors[title] = {
            "title": title,
            "description": doc.get("description", ""),
            "reviews": [],
            "rating": doc.get("rating", 0)
        }
    review_text = doc.get("text", "")
    if review_text.strip():
        unique_flavors[title]["reviews"].append(review_text)

# Compute overall cosine similarity score for a flavor given a query.
def compute_flavor_score(query: str, flavor: dict) -> float:
    # Compute description score.
    desc = flavor.get("description", "")
    desc_score = cosine_sim_for_text(query, desc, idf)
    
    # Compute review score: average the cosine similarity for each review.
    reviews = flavor.get("reviews", [])
    review_scores = [cosine_sim_for_text(query, review, idf) for review in reviews if review.strip()]
    avg_review_score = sum(review_scores) / len(review_scores) if review_scores else 0
    
    # Combine (50% each)
    overall_score = 0.5 * desc_score + 0.5 * avg_review_score
    return overall_score

def json_search(query):
    # If no query provided, return all unique flavors with score 0.
    if not query:
        out = []
        for flavor in unique_flavors.values():
            out.append({
                "title": flavor["title"],
                "description": flavor["description"],
                "rating": flavor.get("rating", 0),
                "score": 0
            })
        return json.dumps(out[:10])
    
    # Compute overall score for each unique flavor.
    scored_flavors = []
    for flavor in unique_flavors.values():
        score = compute_flavor_score(query, flavor)
        if score > 0:
            scored_flavors.append((score, flavor))
    scored_flavors.sort(key=lambda x: x[0], reverse=True)
    
    # Return only the top 10 flavors.
    out = []
    for score, flavor in scored_flavors[:10]:
        out.append({
            "title": flavor["title"],
            "description": flavor["description"],
            "rating": flavor.get("rating", 0),
            "score": score
        })
    return json.dumps(out)

@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/flavors")
def flavors_search():
    text = request.args.get("title", "")
    return json_search(text)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)




