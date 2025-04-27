import json
import os
import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from svd_similarity import build_composite_svd_models, query_composite_svd_similarity, CUSTOM_STOPWORDS

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as f:
    data = json.load(f)
    flavors_df = pd.DataFrame(data['flavors'])

docs = flavors_df.to_dict(orient='records')
unique_flavors = {}
for doc in docs:
    t = doc.get("title","").strip()
    if t not in unique_flavors:
        unique_flavors[t] = {
            "title": t,
            "brand": doc.get("brand",""),
            "description": doc.get("description",""),
            "subhead": doc.get("subhead",""),
            "ingredients_y": doc.get("ingredients_y",""),
            "rating": doc.get("rating",0),
            "reviews_list": [f"{doc.get('author','')}: {doc.get('text','')}"]
        }
    else:
        unique_flavors[t]["reviews_list"].append(f"{doc.get('author','')}: {doc.get('text','')}")

flavor_list = list(unique_flavors.values())
models = build_composite_svd_models(flavor_list,n_components=300)
weights = {"description":0.1,"subhead":0.4,"ingredients":0.3,"reviews":0.2}

app = Flask(__name__)
CORS(app)

ALLERGY_KEYWORDS = {
    "dairy":["milk","cream","cheese","butter","whey","casein","yogurt","skim milk"],
    "nuts":["peanut","almond","cashew","walnut","hazelnut","macadamia","pecan","pistachio","nut","peanus","almonds","cashews","walnuts","hazelnuts","macadamias","pecans","pistachios","nuts"],
    "gluten":["wheat","barley","rye","spelt","farro","malt"],
    "soy":["soy","soya","soybean","edamame","tofu"],
    "eggs":["egg","egg yolk","egg white","albumin","eggs","egg yolks","egg whites","albumins"]
}

def normalize_brand(b):
    bb = b.lower()
    if bb == "bj": return "Ben and Jerry's"
    if bb == "hd": return "Haagen Dazs"
    return b.title()

def make_safe_id(b,t):
    r = f"{b}-{t}".replace(" ","-")
    return "".join(c for c in r if c.isalnum() or c=="-").lower()

def json_search(query,min_rating=0,allergy_list=[]):
    if not query.strip(): return json.dumps([])
    comp_scores = query_composite_svd_similarity(query,models,weights)
    pairs = [(s,flavor_list[i],i) for i,s in enumerate(comp_scores) if s>0]
    pairs.sort(key=lambda x:x[0],reverse=True)
    filt = []
    for s,fl,idx in pairs:
        if float(fl["rating"]) < min_rating: continue
        ing = fl["ingredients_y"].lower()
        if any(kw in ing for a in allergy_list for kw in ALLERGY_KEYWORDS.get(a,[])): continue
        filt.append((s,fl,idx))
        if len(filt)>=10: break
    out = []
    for s,fl,idx in filt:
        nb = normalize_brand(fl["brand"])
        expl = {}
        top_n, terms_p = 3,5
        for field,(vec,svd,doc_vecs) in models.items():
            qv = svd.transform(vec.transform([query]))[0]
            fv = doc_vecs[idx]
            dims = np.argsort(-np.abs(qv*fv))[:top_n]
            terms = vec.get_feature_names_out()
            ts = {}
            for d in dims:
                comp = svd.components_[d]
                for term,wt in zip(terms,comp):
                    if term in CUSTOM_STOPWORDS: continue
                    ts[term] = ts.get(term,0) + abs(wt)
            top = sorted(ts.items(),key=lambda x:-x[1])[:terms_p]
            expl[field] = {"themes":[t for t,_ in top],"scores":[float(v) for _,v in top]}
        out.append({
            "safeId": make_safe_id(nb,fl["title"]),
            "title": fl["title"].title(),
            "brand": nb,
            "description": fl["description"],
            "subhead": fl["subhead"],
            "ingredients_y": fl["ingredients_y"],
            "rating": fl["rating"],
            "composite_score": s,
            "reviews": fl["reviews_list"],
            "explanation": expl
        })
    return json.dumps(out)

@app.route("/")
def home():
    return render_template("base.html",popup_text="this is a popup")

@app.route("/flavors")
def flavors_search():
    q = request.args.get("title","")
    mr = float(request.args.get("min_rating",0))
    al = [a.strip().lower() for a in request.args.get("allergies","").split(",") if a]
    return json_search(q,mr,al)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
