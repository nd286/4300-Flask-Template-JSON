import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

from cosine_similarity import (
    build_inverted_index,
    compute_idf,
    compute_doc_norms,
    index_search
)

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load JSON data into a DataFrame
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

# Convert DataFrame to list of docs
docs = flavors_df.to_dict(orient='records')
n_docs = len(docs)

# Build the inverted index from 'description'
inv_index = build_inverted_index(docs)

# Compute IDF and doc norms
idf = compute_idf(inv_index, n_docs, min_df=1, max_df_ratio=1.0)
doc_norms = compute_doc_norms(inv_index, idf, n_docs)

app = Flask(__name__)
CORS(app)

def json_search(query):
    if query:
        # Perform cosine similarity search
        results = index_search(query, docs, inv_index, idf, doc_norms)
        # Keep only docs with score > 0
        sorted_docs = [docs[doc_id] for (score, doc_id) in results if score > 0]
    else:
        # No query => return all docs
        sorted_docs = docs

    # Remove duplicates if needed
    unique_docs = []
    seen = set()
    for d in sorted_docs:
        key = (d.get('title', ''), d.get('description', ''))
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    # Format as JSON
    # Return only the fields you want to expose
    output = [
        {
            "title": doc["title"],
            "description": doc["description"],
            "rating": doc.get("rating", 0)
        }
        for doc in unique_docs
    ]
    return json.dumps(output)

@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/flavors")
def flavors_search():
    text = request.args.get("title", "")
    return json_search(text)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)


