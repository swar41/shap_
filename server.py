from flask import Flask, request, jsonify
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Flask app
app = Flask(__name__)

# Load sample dataset
data = pd.DataFrame({
    "student_expertise": [7, 6, 8, 4, 9, 5],
    "topic_difficulty": [5, 9, 3, 8, 4, 6],
    "supervisor_availability": [8, 5, 9, 6, 8, 5],
    "past_project_similarity": [6, 4, 8, 3, 7, 5],
    "time_to_completion": [12, 15, 10, 18, 8, 14],
    "suitability": [1, 0, 1, 0, 1, 0],
})

X = data.drop("suitability", axis=1)
y = data["suitability"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

@app.route("/api/explain", methods=["POST"])
def explain():
    input_data = request.json
    X_sample = np.array(input_data["features"]).reshape(1, -1)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value

    # Prepare response
    response = {
        "shap_values": shap_values[1].tolist(),
        "expected_value": expected_value[1],
        "features": input_data["features"],
        "feature_names": list(X.columns),
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
