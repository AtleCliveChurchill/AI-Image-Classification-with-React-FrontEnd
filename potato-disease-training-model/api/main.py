# Importing necessary modules and frameworks
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Creating a FastAPI application instance
app = FastAPI()

# Defining the allowed origins for CORS (Cross-Origin Resource Sharing)
origins = [
    "http://localhost",
    "http://localhost:3000",
]

# Adding CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading the pre-trained TensorFlow model
MODEL = tf.keras.models.load_model("../saved_models/1")

# Defining class names for different disease categories
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


# Endpoint for a simple ping to check if the application is alive
@app.get("/ping")
async def ping():
    return "Hello, I am alive"


# Function to read the uploaded file as an image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# Endpoint for predicting the disease class based on the uploaded image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Reading the uploaded file as an image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    # Making predictions using the loaded model
    predictions = MODEL.predict(img_batch)

    # Extracting the predicted class and confidence level
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Returning the prediction result
    return {"class": predicted_class, "confidence": float(confidence)}


# Running the FastAPI application using uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
