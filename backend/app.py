import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

# Set the ROOT_PATH for linking files
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the new JSON file containing flavors
json_file_path = os.path.join(current_directory, 'init.json')

# Load the JSON data and create a DataFrame using the 'flavors' key
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

app = Flask(__name__)
CORS(app)

# Search function: filters based solely on the flavor 'title'
def json_search(query):
    # If no query provided, you can choose to return all records or an empty list.
    if query:
        # Use case-insensitive matching to filter the 'title' field
        matches = flavors_df[flavors_df['title'].str.lower().str.contains(query.lower())]
    else:
        matches = flavors_df
    # Select the desired columns for the response
    matches_filtered = matches[['title', 'subhead', 'description', 'rating']]
    return matches_filtered.to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

@app.route("/flavors")
def flavors_search():
    # Get the query parameter 'title'
    query = request.args.get("title", "")
    return json_search(query)

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
