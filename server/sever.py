# Import necessary libraries
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
from joblib import load
from PIL import Image
from tensorflow.keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load KNN model and ResNet model
knn_model = load("KNN_ResNet.joblib")
resnet_model = load_model("ResNet_Furniture_Classification.h5")

# Load image DataFrame
image_df = pd.read_csv("df.csv")

# Define directory for serving images
IMAGE_DIR = "Furniture_Data"

# Define route to handle recommendations request
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Get the uploaded image file from the request
        uploaded_image = request.files['image']

        # Preprocess the image for inference
        img = Image.open(uploaded_image)
        img = img.resize((224, 224))  # Resize image to match model input size
        img_array = img_preprocessing.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocess the image array

        # Extract features using ResNet model
        feature_vector = resnet_model.predict(img_array)

        # Query k-NN model for recommendations
        _, indices = knn_model.kneighbors(feature_vector, n_neighbors=10)

        # Get filenames of recommended images
        recommended_images = []

        # Iterate through indices and get corresponding image paths from DataFrame
        for index in indices[0]:
            image_path = image_df.iloc[index]['Img']
            recommended_images.append(image_path)

        # Remove the duplicate '/Furniture_Data' from image paths
        recommended_images = [path.replace('/Furniture_Data', '') for path in recommended_images]

        return jsonify(recommended_images)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify(error='Failed to process request'), 500


# Route to serve images
@app.route('/Furniture_Data/<category>/<style>/<filename>')
def get_image(category, style, filename):
    # Construct the full path to the image file
    image_path = os.path.join(IMAGE_DIR, category, style, filename)
    
    # Check if the image exists
    if os.path.isfile(image_path):
        # Serve the image
        return send_file(image_path, mimetype='image/jpeg')
    else:
        # Return a 404 error if the image does not exist
        return "Image not found", 404


# Function to get the full image path
def get_image_path(category, style, filename):
    # Define the base directory for images
    base_dir = os.path.join(IMAGE_DIR, "Furniture_Data")

    # Define a dictionary to map category and style to their respective directories
    category_dirs = {
        "beds": "beds",
        "chairs": "chairs",
        "dressers": "dressers",
        "lamps": "lamps",
        "sofas": "sofas",
        "tables": "tables"
    }

    style_dirs = {
        "Asian": "Asian",
        "Beach": "Beach",
        "Contemp": "Contemp",
        "Craftsman": "Craftsman",
        "Eclectic": "Eclectic",
        "Farmhouse": "Farmhouse",
        "Industrial": "Industrial",
        "Media": "Media",
        "Midcentury": "Midcentury",
        "Modern": "Modern",
        "Rustic": "Rustic",
        "Scandinavian": "Scandinavian",
        "Southwestern": "Southwestern",
        "Traditional": "Traditional",
        "Transitional": "Transitional",
        "Tropical": "Tropical",
        "Victorian": "Victorian"
    }

    # Construct the full directory path
    category_dir = category_dirs.get(category)
    style_dir = style_dirs.get(style)
    if category_dir and style_dir:
        full_dir = os.path.join(base_dir, category_dir, style_dir)
        return os.path.join(full_dir, filename)
    else:
        return None


# Run the Flask app
if __name__ == '__main__':
    print("Starting the Flask server...")
    app.run(debug=True)
