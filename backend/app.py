import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from cosine_similarity import compute_combined_score

# Set ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

# Load JSON data from init.json (expects a top-level "flavors" key).
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

# Convert DataFrame to a list of document dictionaries.
docs = flavors_df.to_dict(orient='records')

# Group documents by flavor title.
# For each unique flavor, store one description (assumed constant) and collect all review texts.
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

app = Flask(__name__)
CORS(app)

def json_search(query):
    if query:
        scored_flavors = []
        # Compute overall combined cosine similarity for each unique flavor.
        for flavor in unique_flavors.values():
            score = compute_combined_score(query,
                                           flavor.get("description", ""),
                                           flavor.get("reviews", []))
            if score > 0:
                scored_flavors.append((score, flavor))
        # Sort by descending score.
        scored_flavors.sort(key=lambda x: x[0], reverse=True)
        # Keep only the top 10 unique flavors.
        out = []
        for score, flavor in scored_flavors[:10]:
            out.append({
                "title": flavor["title"],
                "description": flavor.get("description", ""),
                "rating": flavor.get("rating", 0),
                "score": score
            })
    else:
        # If no query is provided, simply return the first 10 unique flavors.
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
    text = request.args.get("title", "")
    return json_search(text)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)






