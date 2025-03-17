import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

# ROOT_PATH for linking with all your files. 
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the new JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'new_flavors.json')

# Assuming your new JSON data is stored in a file named 'new_flavors.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = flavors_df[flavors_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'subhead', 'description', 'rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    return render_template('base.html', title="sample html")

@app.route("/flavors")
def flavors_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)