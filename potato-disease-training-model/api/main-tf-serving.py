# Import necessary libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

# Initialize FastAPI
app = FastAPI()

# Define allowed origins for CORS (Cross-Origin Resource Sharing)
origins = [
    "http://localhost",
    "http://localhost:3000",
]

# Configure CORS middleware to allow specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define TensorFlow Serving endpoint
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

# Define class names for classification
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


# Define a route for a GET request to "/ping", returns a simple message
@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# Define a function to read file data and convert it into an image array
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# Define a route for a POST request to "/predict" which accepts an uploaded file
@app.post("/predict")
async def predict(file: UploadFile = File(...)):  # Accepts an uploaded file
    # Read uploaded file as an image array
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(
        image, 0
    )  # Expand dimensions to create a batch of size 1

    # Prepare data for inference in JSON format
    json_data = {"instances": img_batch.tolist()}

    # Send a POST request to TensorFlow Serving endpoint for inference
    response = requests.post(endpoint, json=json_data)
    prediction = np.array(
        response.json()["predictions"][0]
    )  # Extract predictions from response

    # Determine predicted class and confidence score
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Return prediction result as a dictionary
    return {"class": predicted_class, "confidence": float(confidence)}


# Run the FastAPI application using Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
