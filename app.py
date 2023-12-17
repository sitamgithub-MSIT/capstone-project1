# Required imports
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request, render_template
from flask_cors import CORS

# Local Imports
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

load_dotenv()

# Creating the application and enabling CORS
application = Flask(__name__)
app = application
CORS(app)

# Environment variables for the application configuration settings
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG")


@app.route("/")
def index():
    """
    This function handles the root route and returns the home.html template.
    """
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    """
    Endpoint for predicting insurance charges based on user input.

    Returns:
        If the request method is GET, renders the prediction.html template.
        If the request method is POST, processes the user input, predicts the insurance charges,
        and renders the prediction.html template with the predicted results.
    """

    # If the request method is GET, render the prediction.html template
    if request.method == "GET":
        return render_template("prediction.html")

    # Getting the user input
    data = CustomData(
        age=int(request.form.get("age")),
        sex=request.form.get("sex"),
        bmi=float(request.form.get("bmi")),
        children=int(request.form.get("children")),
        smoker=request.form.get("smoker"),
        region=request.form.get("region"),
    )

    # Converting the data to a pandas DataFrame and convert children to object type
    pred_df = data.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()

    # Predicting the target variable
    results = predict_pipeline.predict(pred_df)
    return render_template("prediction.html", results=results[0])


# Run the application
if __name__ == "__main__":
    app.run()
