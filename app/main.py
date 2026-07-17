# Import the JSON module to read metadata files
import json

# Import Path to work with file and folder paths
from pathlib import Path

# Import joblib to load saved machine learning models
import joblib

# Import NumPy for numerical operations
import numpy as np

# Import Pandas for data manipulation
import pandas as pd

# Import the FastAPI framework
from fastapi import FastAPI, Request, Form

# Import StaticFiles to serve CSS, JavaScript, and images
from fastapi.staticfiles import StaticFiles

# Import Jinja2 template engine for rendering HTML pages
from fastapi.templating import Jinja2Templates

# Import HTMLResponse to return HTML pages
from fastapi.responses import HTMLResponse

# Get the current project directory
BASE_DIR = Path(__file__).resolve().parent

# Path to the folder containing trained models
MODELS_DIR = BASE_DIR.parent / "models"

# Create the FastAPI application
app = FastAPI(title="Energy Consumption Dashboard")

# Mount the static folder for CSS, JavaScript, and images
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Load HTML templates from the templates folder
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ------------------------------------------------------------------
# Load the saved pipeline (model + scaler + PCA) and feature metadata
# ------------------------------------------------------------------

# Load the trained machine learning model
best_model = joblib.load(MODELS_DIR / "best_model.joblib")

# Load the fitted StandardScaler
scaler = joblib.load(MODELS_DIR / "scaler.joblib")

# Load the fitted PCA model
pca = joblib.load(MODELS_DIR / "pca.joblib")

# Open the metadata JSON file
with open(MODELS_DIR / "feature_metadata.json") as f:

    # Read the metadata into a Python dictionary
    METADATA = json.load(f)

# Get the list of numerical feature names
NUMERIC_COLS = METADATA["numeric_cols"]

# Get the list of categorical feature names
CATEGORICAL_COLS = METADATA["categorical_cols"]

# Get the possible values for each categorical feature
CATEGORICAL_OPTIONS = METADATA["categorical_options"]

# Get statistics (min, max, etc.) for numerical features
NUMERIC_STATS = METADATA["numeric_stats"]

# Get the exact encoded column order used during training
ENCODED_COLUMNS = METADATA["encoded_columns"]


def build_feature_vector(form_values: dict) -> pd.DataFrame:
    """
    Takes the raw form inputs and reproduces the exact preprocessing used
    in training: manually construct the one-hot encoded row (get_dummies
    is unreliable on a single row -- with only one category present it
    always drops it, silently zeroing out every categorical input), then
    scaler -> PCA.
    """
    encoded = pd.DataFrame(0, index=[0], columns=ENCODED_COLUMNS, dtype=float)

    for col in NUMERIC_COLS:
        encoded[col] = float(form_values[col])

    for col in CATEGORICAL_COLS:
        value = form_values[col]
        dummy_col = f"{col}_{value}"
        if dummy_col in encoded.columns:
            encoded[dummy_col] = 1
        # if dummy_col isn't in ENCODED_COLUMNS, this value was the
        # drop_first baseline category -- correctly stays all-zero

    encoded = encoded[ENCODED_COLUMNS]  # exact training column order

    scaled = scaler.transform(encoded)
    reduced = pca.transform(scaled)
    return reduced

# Home page route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    # Render the home page
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request}
    )


# Dashboard page route
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):

    # Render the dashboard page
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"request": request}
    )


# Display the prediction form
@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):

    # Render the prediction form page
    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "request": request,
            "numeric_cols": NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "categorical_options": CATEGORICAL_OPTIONS,
            "numeric_stats": NUMERIC_STATS,
            "prediction": None,
            "error": None,
            "form_values": {},
        },
    )


# Handle prediction requests
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):

    # Read all submitted form data
    form = await request.form()

    # Convert form data into a Python dictionary
    form_values = dict(form)

    try:

        # Preprocess the input data
        features = build_feature_vector(form_values)

        # Generate prediction using the trained model
        prediction = float(best_model.predict(features)[0])

        # Round prediction to 3 decimal places
        prediction = round(prediction, 3)

        # No error occurred
        error = None

    except Exception as e:

        # If an error occurs, prediction is unavailable
        prediction = None

        # Store the error message
        error = f"Could not generate a prediction: {e}"

    # Return the prediction page with results
    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "request": request,
            "numeric_cols": NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "categorical_options": CATEGORICAL_OPTIONS,
            "numeric_stats": NUMERIC_STATS,
            "prediction": prediction,
            "error": error,
            "form_values": form_values,
        },
    )