import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from svd_similarity import build_composite_svd_models, query_composite_svd_similarity

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
            "brand": doc.get("brand", ""),
            "description": doc.get("description", ""),
            "subhead": doc.get("subhead", ""),
            "ingredients_y": doc.get("ingredients_y", ""),
            "rating": doc.get("rating", 0),
            "text": doc.get("text", "")
        }
    else:
        unique_flavors[title]["text"] += " " + doc.get("text", "")

flavor_list = list(unique_flavors.values())

composite_models = build_composite_svd_models(flavor_list, n_components=300)

weights = {
    "description": 0.4,
    "subhead": 0.3,
    "ingredients": 0.2,
    "reviews": 0.1
}

app = Flask(__name__)
CORS(app)


def normalize_brand(brand):
    brand_lower = brand.lower()
    if brand_lower == "bj":
        return "Ben and Jerry's"
    elif brand_lower == "hd":
        return "Haagen Dazs"
    else:
        return brand.title()


def make_safe_id(brand, title):
    raw = f"{brand}-{title}"
    raw = raw.replace(" ", "-")
    safe = "".join(ch for ch in raw if ch.isalnum() or ch == "-").lower()
    return safe


def json_search(query: str, min_rating=0, allergy_list=[]) -> str:
    if not query.strip():
        return json.dumps([])

    composite_scores = query_composite_svd_similarity(
        query, composite_models, weights)
    scored_flavors = []
    for idx, score in enumerate(composite_scores):
        if score > 0:
            scored_flavors.append((score, flavor_list[idx]))
    scored_flavors.sort(key=lambda x: x[0], reverse=True)

    filtered_flavors = []
    for score, flavor in scored_flavors:
        if float(flavor.get("rating", 0)) < min_rating:
            continue
        ingredients = flavor.get("ingredients_y", "").lower()

        exclude = False
        for allergy in allergy_list:
            keywords = ALLERGY_KEYWORDS.get(allergy.lower(), [])
            if any(kw in ingredients for kw in keywords):
                exclude = True
                break
        if exclude:
            continue

        filtered_flavors.append((score, flavor))
        if len(filtered_flavors) >= 10:
            break

    out = []
    for score, flavor in filtered_flavors:
        norm_brand = normalize_brand(flavor.get("brand", ""))
        out.append({
            "safeId": make_safe_id(norm_brand, flavor["title"]),
            "title": flavor["title"].title(),
            "brand": norm_brand,
            "description": flavor.get("description", ""),
            "subhead": flavor.get("subhead", ""),
            "ingredients_y": flavor.get("ingredients_y", ""),
            "rating": flavor.get("rating", 0),
            "composite_score": score,
            "reviews": flavor.get("text", "")
        })
    return json.dumps(out)


ALLERGY_KEYWORDS = {
    "dairy": ["milk", "cream", "cheese", "butter", "whey", "casein", "yogurt"],
    "nuts": ["peanut", "almond", "cashew", "walnut", "hazelnut", "macadamia", "pecan", "pistachio", "nut"],
    "gluten": ["wheat", "barley", "rye", "spelt", "farro", "malt"],
    "soy": ["soy", "soya", "soybean", "edamame", "tofu"],
    "eggs": ["egg", "egg yolk", "egg white", "albumin"]
}


@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")


@app.route("/flavors")
def flavors_search():
    query = request.args.get("title", "")
    min_rating = float(request.args.get("min_rating", 0))
    allergies = request.args.get("allergies", "")
    allergy_list = [a.strip().lower() for a in allergies.split(",") if a]
    return json_search(query, min_rating, allergy_list)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
