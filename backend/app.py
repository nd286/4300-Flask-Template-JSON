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

# Set ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script.
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load JSON data from init.json and create a DataFrame from the "flavors" key.
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

# Convert DataFrame to a list of document dictionaries.
docs = flavors_df.to_dict(orient='records')
n_docs = len(docs)

# Build an inverted index from the "description" field.
inv_index = build_inverted_index(docs)
idf = compute_idf(inv_index, n_docs, min_df=1, max_df_ratio=1.0)
doc_norms = compute_doc_norms(inv_index, idf, n_docs)

app = Flask(__name__)
CORS(app)

def json_search(query):
    if query:
        # Perform cosine similarity search using the query against document descriptions.
        results = index_search(query, docs, inv_index, idf, doc_norms)
        # results is a list of (score, doc_id) tuples sorted by descending score.
        out = []
        seen = set()
        for score, doc_id in results:
            if score > 0:
                doc = docs[doc_id]
                key = (doc.get('title', ''), doc.get('description', ''))
                if key not in seen:
                    seen.add(key)
                    out.append({
                        "title": doc["title"],
                        "description": doc["description"],
                        "rating": doc.get("rating", 0),
                        "score": score  # Include the cosine similarity score
                    })
    else:
        # If no query is provided, return all docs with a default score (e.g., 0).
        out = []
        seen = set()
        for doc in docs:
            key = (doc.get('title', ''), doc.get('description', ''))
            if key not in seen:
                seen.add(key)
                out.append({
                    "title": doc["title"],
                    "description": doc["description"],
                    "rating": doc.get("rating", 0),
                    "score": 0
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


