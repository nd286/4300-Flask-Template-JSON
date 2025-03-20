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

# Load JSON data from init.json (which should have a top-level key "flavors")
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

# Convert DataFrame to a list of document dictionaries.
docs = flavors_df.to_dict(orient='records')
n_docs = len(docs)

# Build an inverted index using the combined 'description' and 'text' fields.
inv_index = build_inverted_index(docs)
idf = compute_idf(inv_index, n_docs, min_df=1, max_df_ratio=1.0)
doc_norms = compute_doc_norms(inv_index, idf, n_docs)

app = Flask(__name__)
CORS(app)

def json_search(query):
    if query:
        results = index_search(query, docs, inv_index, idf, doc_norms)
        out = []
        seen = set()
        for score, doc_id in results:
            if score > 0:
                doc = docs[doc_id]
                key = (doc.get('title', ''), doc.get('description', ''), doc.get('text', ''))
                if key not in seen:
                    seen.add(key)
                    out.append({
                        "title": doc["title"],
                        "description": doc.get("description", ""),
                        "review": doc.get("text", ""),
                        "rating": doc.get("rating", 0),
                        "score": score  
                    })
    else:
        out = []
        seen = set()
        for doc in docs:
            key = (doc.get('title', ''), doc.get('description', ''), doc.get('text', ''))
            if key not in seen:
                seen.add(key)
                out.append({
                    "title": doc["title"],
                    "description": doc.get("description", ""),
                    "review": doc.get("text", ""),
                    "rating": doc.get("rating", 0),
                    "score": 0
                })
    # Return only the first 10 matches.
    return json.dumps(out[:10])

@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/flavors")
def flavors_search():
    text = request.args.get("title", "")
    return json_search(text)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)




