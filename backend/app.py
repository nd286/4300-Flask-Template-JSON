import json
import os
import math
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from cosine_similarity import (
    tokenize,
    build_inverted_index,
    compute_idf,
    compute_doc_norms,
    compute_combined_score
)

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

docs = flavors_df.to_dict(orient='records')

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

corpus = []
for flavor in unique_flavors.values():
    combined = flavor.get("description", "")
    for review in flavor.get("reviews", []):
        combined += " " + review
    corpus.append(combined)

msgs = [{"text": text} for text in corpus]
inv_index = {}
for doc_id, msg in enumerate(msgs):
    tokens = tokenize(msg.get("text", ""))
    token_counts = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    for token, count in token_counts.items():
        if token not in inv_index:
            inv_index[token] = []
        inv_index[token].append((doc_id, count))
n_docs = len(msgs)

idf = {}
for term, postings in inv_index.items():
    if len(postings) >= 1 and (len(postings)/n_docs) <= 0.9:
        idf[term] = math.log(n_docs / (len(postings) + 1), 2)

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
                                           flavor.get("reviews", []),
                                           idf)
            if score > 0:
                scored_flavors.append((score, flavor))

        scored_flavors.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, flavor in scored_flavors[:10]:
            out.append({
                "title": flavor["title"],
                "description": flavor.get("description", ""),
                "rating": flavor.get("rating", 0),
                "score": score
            })
    else:
        out = []
        for flavor in list(unique_flavors.values())[:10]:
            out.append({
                "title": flavor["title"],
                "description": flavor.get("description", ""),
                "rating": flavor.get("rating", 0),
                "score": 0
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









