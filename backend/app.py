import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from svd_similarity import build_composite_svd_models, query_composite_svd_similarity

# Initialize Flask application and load flavor data
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as f:
    data = json.load(f)
    flavors_df = pd.DataFrame(data['flavors'])

docs = flavors_df.to_dict(orient='records')
unique = {}
for d in docs:
    t = d.get("title","").strip()
    if t not in unique:
        unique[t] = {
            "title": t,
            "brand": d.get("brand",""),
            "description": d.get("description",""),
            "subhead": d.get("subhead",""),
            "ingredients_y": d.get("ingredients_y",""),
            "rating": d.get("rating",0),
            "text": d.get("text","")
        }
    else:
        unique[t]["text"] += " "+d.get("text","")

flavor_list = list(unique.values())
composite_models = build_composite_svd_models(flavor_list, n_components=300)
weights = {"description":0.1,"subhead":0.4,"ingredients":0.3,"reviews":0.2}

app = Flask(__name__)
CORS(app)

def normalize_brand(b):
    l = b.lower()
    if l == "bj": return "Ben and Jerry's"
    if l == "hd": return "Haagen Dazs"
    if l == "breyers": return "Breyers"
    return b.title()

def make_safe_id(b,t):
    raw = f"{b}-{t}".replace(" ","-")
    return "".join(c for c in raw if c.isalnum() or c=="-").lower()

def make_explanation(query, models, top_n=5):
    expl = {}
    for field, (vect, svd) in models.items():
        tf = vect.transform([query])
        lat = svd.transform(tf)[0]
        idxs = lat.argsort()[::-1][:top_n]
        names = vect.get_feature_names_out()
        expl[field] = [{"theme": names[i], "score": float(lat[i])} for i in idxs]
    return expl

@app.route("/")
def home():
    return render_template("base.html", title="Dairy Godmothers")

@app.route("/flavors")
def flavors_search():
    q = request.args.get("title", "")
    mr = float(request.args.get("min_rating", 0))
    al = [a.strip().lower() for a in request.args.get("allergies", "").split(",") if a]
    if not q.strip():
        return json.dumps([])

    scores = query_composite_svd_similarity(q, composite_models, weights)
    pairs = [(s, flavor_list[i], i) for i, s in enumerate(scores) if s > 0]
    pairs.sort(key=lambda x: x[0], reverse=True)

    out = []
    for s, fl, i in pairs:
        if float(fl["rating"]) < mr: continue
        expl = make_explanation(q, composite_models, top_n=5)
        nb = normalize_brand(fl["brand"])
        out.append({
            "safeId": make_safe_id(nb, fl["title"]),
            "title": fl["title"].title(),
            "brand": nb,
            "description": fl["description"],
            "subhead": fl["subhead"],
            "ingredients_y": fl["ingredients_y"],
            "rating": fl["rating"],
            "explanation": expl
        })
        if len(out) >= 10: break
    return json.dumps(out)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
