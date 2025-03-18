import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd

# Set ROOT_PATH for linking with all your files.
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script.
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file (init.json) relative to the current script.
json_file_path = os.path.join(current_directory, 'init.json')

# Load the JSON data and create a DataFrame from the "flavors" key.
with open(json_file_path, 'r') as file:
    data = json.load(file)
    flavors_df = pd.DataFrame(data['flavors'])

app = Flask(__name__)
CORS(app)

# Search function: filters based on the flavor "title" (case-insensitive).
def json_search(query):
    if query:
        matches = flavors_df[flavors_df['title'].str.lower().str.contains(query.lower())]
    else:
        matches = flavors_df
    # Return only the fields you want (here: title, description, rating)
    matches_filtered = matches[['title', 'description', 'rating']]
    return matches_filtered.to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html', title="Sample HTML")

# Use the /flavors endpoint for searching flavors by title.
@app.route("/flavors")
def flavors_search():
    query = request.args.get("title", "")
    return json_search(query)

if __name__ == '__main__':
    # Run the server if 'DB_NAME' is not set in the environment.
    if 'DB_NAME' not in os.environ:
        app.run(debug=True, host="0.0.0.0", port=5000)
