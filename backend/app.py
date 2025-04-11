# app.py
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from svd_similarity import build_svd_model, query_svd_similarity

# Set up the ROOT_PATH and file paths.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load the flavor data from JSON and create a DataFrame.
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

# Convert DataFrame records to a list of dictionaries.
docs = flavors_df.to_dict(orient='records')

# Build a unique collection of flavors (using the "title" as key).
unique_flavors = {}
for doc in docs:
    title = doc.get("title", "").strip()
    if title not in unique_flavors:
        unique_flavors[title] = {
            "title": title,
            "description": doc.get("description", ""),
            "subhead": doc.get("subhead", ""),
            "ingredients_y": doc.get("ingredients_y", ""),
            "rating": doc.get("rating", 0)
        }

# Build the corpus: For each unique flavor, combine description, subhead, and ingredients.
corpus = []
flavor_list = []  # Preserving the same order of documents in the corpus.
for flavor in unique_flavors.values():
    combined = flavor.get("description", "") + " " + flavor.get("subhead", "") + " " + flavor.get("ingredients_y", "")
    corpus.append(combined)
    flavor_list.append(flavor)

# Build the SVD model from the corpus. Adjust n_components if needed.
vectorizer, svd_model, doc_vectors = build_svd_model(corpus, n_components=50)

app = Flask(__name__)
CORS(app)

def json_search(query: str) -> str:
    if not query.strip():
        return json.dumps([])
    # Compute the similarity scores using the SVD-based model.
    sim_scores = query_svd_similarity(query, vectorizer, svd_model, doc_vectors)
    
    # Prepare the list of scored flavors.
    scored_flavors = []
    for idx, score in enumerate(sim_scores):
        if score > 0:
            scored_flavors.append((score, flavor_list[idx]))
    scored_flavors.sort(key=lambda x: x[0], reverse=True)
    
    # Build the output (top 10 results).
    out = []
    for rec_num, (score, flavor) in enumerate(scored_flavors[:10], start=1):
        out.append({
            "recommendation": rec_num,
            "title": flavor["title"],
            "description": flavor.get("description", ""),
            "subhead": flavor.get("subhead", ""),
            "ingredients_y": flavor.get("ingredients_y", ""),
            "rating": flavor.get("rating", 0),
            "similarity_score": score
        })
    return json.dumps(out)

@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/flavors")
def flavors_search():
    query = request.args.get("title", "")
    return json_search(query)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)












