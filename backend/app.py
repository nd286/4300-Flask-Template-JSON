import json
import os
import math
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from cosine_similarity import tokenize, build_inverted_index, compute_idf, compute_doc_norms, compute_combined_score

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

docs = flavors_df.to_dict(orient='records')

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

corpus = []
for flavor in unique_flavors.values():
    combined = flavor.get("description", "") + " " + flavor.get("subhead", "") + " " + flavor.get("ingredients_y", "")
    corpus.append(combined)

inv_index = {}
for doc_id, text in enumerate(corpus):
    tokens = tokenize(text)
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    for token, count in counts.items():
        if token not in inv_index:
            inv_index[token] = []
        inv_index[token].append((doc_id, count))
n_docs = len(corpus)

idf = {}
for term, postings in inv_index.items():
    if len(postings) >= 1 and (len(postings) / n_docs) <= 1.0:
        idf[term] = math.log2(n_docs / len(postings))

doc_norms = [0.0] * n_docs
for term, postings in inv_index.items():
    if term in idf:
        w = idf[term]
        for doc_id, count in postings:
            doc_norms[doc_id] += (count * w) ** 2
doc_norms = [math.sqrt(x) for x in doc_norms]

app = Flask(__name__)
CORS(app)

def json_search(query: str) -> str:
    scored_flavors = []
    if query:
        for flavor in unique_flavors.values():
            score = compute_combined_score(query,
                                           flavor.get("description", ""),
                                           flavor.get("subhead", ""),
                                           flavor.get("ingredients_y", ""),
                                           idf)
            if score > 0:
                scored_flavors.append((score, flavor))
        scored_flavors.sort(key=lambda x: x[0], reverse=True)
        out = []
        for rank, (score, flavor) in enumerate(scored_flavors[:10], start=1):
            out.append({
                "recommendation": rank,
                "title": flavor["title"],
                "description": flavor.get("description", ""),
                "subhead": flavor.get("subhead", ""),
                "ingredients_y": flavor.get("ingredients_y", ""),
                "rating": flavor.get("rating", 0)
            })
    else:
        out = []
        for rank, flavor in enumerate(list(unique_flavors.values())[:10], start=1):
            out.append({
                "recommendation": rank,
                "title": flavor["title"],
                "description": flavor.get("description", ""),
                "subhead": flavor.get("subhead", ""),
                "ingredients_y": flavor.get("ingredients_y", ""),
                "rating": flavor.get("rating", 0)
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











