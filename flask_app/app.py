from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import logging
import os
import mlflow
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_Token")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_Token environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
# repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
# repo_name = os.getenv("DAGSHUB_REPO_NAME")
repo_owner = "NitinNandeshwar"
repo_name = "learnyard-capstone-project1"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# ----------------------------------------------
# Configuration
# ----------------------------------------------
MODEL_NAME = "my_model"
PREPROCESSOR_PATH = "models/feature_transformer.pkl"

# Load artifacts
# feature_transformer = joblib.load("models/feature_transformer.pkl")
# model = joblib.load("models/model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Custom Metrics for Monitoring
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions", registry=registry)

# ----------------------------------------------
# Load Model and Preprocessor
# ----------------------------------------------
def get_latest_model_version(model_name):
    """Fetch the latest model version from MLflow."""
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(versions, key=lambda v: int(v.version)).version
        return latest_version
    except Exception as e:
        logging.error(f"Error fetching model version: {e}")
        return None
    
def load_model(model_name):
    """Load the latest model from MLflow."""
    model_version = get_latest_model_version(model_name)
    if model_version:
        model_uri = f"models:/{model_name}/{model_version}"
        logging.info(f"Loading model from: {model_uri}")
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    return None


def load_preprocessor(preprocessor_path):
    """Load Feature Transformer from file."""
    try:
        with open(preprocessor_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading Feature Transformer: {e}")
        return None

# Load ML components
model = load_model(MODEL_NAME)
feature_transformer = load_preprocessor(PREPROCESSOR_PATH)


# ----------------------------------------------
# Helper Functions
# ----------------------------------------------
def preprocess_input(data):
    """Preprocess user input before prediction."""
    try:
        # input_array = np.array(data).reshape(1, -1)  # Ensure correct shape
        transformed_input = feature_transformer.transform(data)  # Apply transformation
        return transformed_input
    except Exception as e:
        logging.error(f"Preprocessing feature transformer Error: {e}")
        return None
    


# ----------------------------------------------
# Routes
# ----------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        form = request.form

        input_df = pd.DataFrame([{
            "Podcast_Name": form.get("Podcast_Name"),
            "Episode_Title": form.get("Episode_Title"),
            "Episode_Length_minutes": float(form.get("Episode_Length")),
            "Genre": form.get("Genre"),
            "Host_Popularity_percentage": float(form.get("Host_Popularity")),
            "Publication_Day": form.get("Publication_Day"),
            "Publication_Time": form.get("Publication_Time"),
            "Guest_Popularity_percentage": float(form.get("Guest_Popularity")),
            "Number_of_Ads": int(form.get("Number_of_Ads")),
            "Episode_Sentiment": form.get("Episode_Sentiment")
        }])

        X_transformed = preprocess_input(input_df)

        if hasattr(feature_transformer, "feature_names_"):
            X_transformed = pd.DataFrame(
                X_transformed,
                columns=feature_transformer.feature_names_
            )

        if X_transformed is None and model is None:
            raise ValueError("Preprocessing or Model loading failed.")
            prediction = model.predict(X_transformed)[0]
            PREDICTION_COUNT.labels(prediction=str(round(float(prediction), 2))).inc()
            REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

            return render_template(
                "result.html",
                prediction=round(float(prediction), 2),
                data=form
            )

        return "Error: Model or Transformer not loaded properly."
    
    except Exception as e:
        logging.error("Prediction failed: %s", e)
        return render_template("result.html", error="Prediction failed")

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
