import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from svd_similarity import build_composite_svd_models, query_composite_svd_similarity, get_latent_themes_for_all_fields

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
weights = {"description": 0.1, "subhead": 0.4, "ingredients": 0.3, "reviews": 0.2}

app = Flask(__name__)
CORS(app)

ALLERGY_KEYWORDS = {
    "dairy": ["milk", "cream", "cheese", "butter", "whey", "casein", "yogurt", "skim milk"],
    "nuts": ["peanut", "almond", "cashew", "walnut", "hazelnut", "macadamia", "pecan", "pistachio", "nut", "peanus", "almonds", "cashews", "walnuts", "hazelnuts", "macadamias", "pecans", "pistachios", "nuts"],
    "gluten": ["wheat", "barley", "rye", "spelt", "farro", "malt"],
    "soy": ["soy", "soya", "soybean", "edamame", "tofu"],
    "eggs": ["egg", "egg yolk", "egg white", "albumin", "eggs", "egg yolks", "egg whites", "albumins"]
}

def normalize_brand(brand):
    b = brand.lower()
    if b == "bj":
        return "Ben and Jerry's"
    if b == "hd":
        return "Haagen Dazs"
    return brand.title()

def make_safe_id(brand, title):
    raw = f"{brand}-{title}".replace(" ", "-")
    return "".join(c for c in raw if c.isalnum() or c == "-").lower()

def json_search(query, min_rating=0, allergy_list=[]):
    if not query.strip():
        return json.dumps([])
    scores = query_composite_svd_similarity(query, composite_models, weights)
    pairs = [(s, flavor_list[i], i) for i, s in enumerate(scores) if s > 0]
    pairs.sort(key=lambda x: x[0], reverse=True)
    filtered = []
    for s, fl, idx in pairs:
        if float(fl["rating"]) < min_rating:
            continue
        ing = fl["ingredients_y"].lower()
        if any(kw in ing for a in allergy_list for kw in ALLERGY_KEYWORDS.get(a, [])):
            continue
        filtered.append((s, fl, idx))
        if len(filtered) >= 10:
            break
    out = []
    for s, fl, idx in filtered:
        nb = normalize_brand(fl["brand"])
        raw_themes = get_latent_themes_for_all_fields(query, composite_models, idx)
        explanation = {}
        for field, theme_list in raw_themes.items():
            words, scores_vals = zip(*theme_list)
            explanation[field] = {"words": list(words), "scores": list(scores_vals)}
        out.append({
            "safeId": make_safe_id(nb, fl["title"]),
            "title": fl["title"].title(),
            "brand": nb,
            "description": fl["description"],
            "subhead": fl["subhead"],
            "ingredients_y": fl["ingredients_y"],
            "rating": fl["rating"],
            "composite_score": s,
            "reviews": fl["text"],
            "explanation": explanation
        })
    return json.dumps(out)

@app.route("/")
def home():
    popup_text = "this is a popup"
    return render_template("base.html", popup_text=popup_text)

@app.route("/flavors")
def flavors_search():
    q = request.args.get("title", "")
    mr = float(request.args.get("min_rating", 0))
    al = [a.strip().lower() for a in request.args.get("allergies", "").split(",") if a]
    return json_search(q, mr, al)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
