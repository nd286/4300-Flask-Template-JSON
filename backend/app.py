# app.py
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from bert_similarity import build_document_embeddings, query_bert_similarity

# Set file paths and environment variables.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load the JSON data and create a DataFrame.
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

# Convert the DataFrame records to a list of dictionaries.
docs = flavors_df.to_dict(orient='records')

# Build a unique flavors dictionary (keyed by "title").  
# For flavors with the same title, combine their review texts (the "text" field).
unique_flavors = {}
for doc in docs:
    title = doc.get("title", "").strip()
    if title not in unique_flavors:
        unique_flavors[title] = {
            "title": title,
            "description": doc.get("description", ""),
            "subhead": doc.get("subhead", ""),
            "ingredients_y": doc.get("ingredients_y", ""),
            "rating": doc.get("rating", 0),
            "text": doc.get("text", "")
        }
    else:
        unique_flavors[title]["text"] += " " + doc.get("text", "")

# Convert the unique flavors into a list (maintaining order).
flavor_list = list(unique_flavors.values())

# Define field weights for the composite embedding.
# (These weights sum to 1.0; adjust as needed.)
weights = {"description": 0.4, "subhead": 0.3, "ingredients": 0.1, "reviews": 0.2}

# Build BERT-based composite embeddings for all unique flavor documents.
# This loads a pre-trained model and computes a composite embedding for each document.
model, doc_embeddings = build_document_embeddings(flavor_list, model_name='paraphrase-MiniLM-L6-v2', weights=weights)

app = Flask(__name__)
CORS(app)

def json_search(query: str) -> str:
    if not query.strip():
        return json.dumps([])
    
    # Use BERT to compute similarity scores.
    scores = query_bert_similarity(query, model, doc_embeddings)
    
    scored_flavors = []
    for idx, score in enumerate(scores):
        if score > 0:
            scored_flavors.append((score, flavor_list[idx]))
    scored_flavors.sort(key=lambda x: x[0], reverse=True)
    
    out = []
    for rec_num, (score, flavor) in enumerate(scored_flavors[:10], start=1):
        out.append({
            "recommendation": rec_num,
            "title": flavor["title"],
            "description": flavor.get("description", ""),
            "subhead": flavor.get("subhead", ""),
            "ingredients_y": flavor.get("ingredients_y", ""),
            "rating": flavor.get("rating", 0),
            "bert_similarity_score": float(score),
            "reviews": flavor.get("text", "")
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













