import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

# Set ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script.
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script.
json_file_path = os.path.join(current_directory, 'init.json')

# Load the JSON data from init.json and create a DataFrame using the "flavors" key.
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

app = Flask(__name__)
CORS(app)

# Search function: filters the DataFrame by "title" (case-insensitive) and returns selected fields.
def json_search(query):
    if query:
        matches = flavors_df[flavors_df['title'].str.lower().str.contains(query.lower())]
    else:
        matches = flavors_df
    matches_filtered = matches[['title', 'description', 'rating']]
    return matches_filtered.to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html', title="sample html")

@app.route("/flavors")
def flavors_search():
    text = request.args.get("title", "")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
