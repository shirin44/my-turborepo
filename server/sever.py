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
import random
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load KNN model and ResNet model
knn_model = load("KNN_ResNet_V2.joblib")
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
        input_image_features = resnet_model.predict(img_array).flatten()

        # Log the shape of the extracted features
        print("Shape of extracted features:", input_image_features.shape)

        # Query k-NN model for recommendations
        _, indices = knn_model.kneighbors(input_image_features.reshape(1, -1), n_neighbors=10)  # Increased neighbors for more options

        # Get filenames and similarity scores of recommended images
        recommended_images = []

        # Add randomization to ensure diverse recommendations
        selected_indices = random.sample(list(indices[0]), 10)  # Randomly select 10 from 20 neighbors

        for index in selected_indices:
            image_path = image_df.iloc[index]['Img']
            similar_image_features = resnet_model.predict(np.expand_dims(img_preprocessing.load_img(image_path, target_size=(224, 224)), axis=0)).flatten()
            similarity_score = float(cosine_similarity([input_image_features], [similar_image_features])[0][0])
            recommended_images.append({'path': image_path.replace('/Furniture_Data', ''), 'score': similarity_score})

            # Print the extracted features for each image
            print(f"Extracted features for image '{image_path}': {similar_image_features.tolist()}")

        print(f"Recommended images: {recommended_images}")
        return jsonify(recommended_images)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify(error='Failed to process request'), 500

# Route to serve images
@app.route('/Furniture_Data/<category>/<style>/<filename>')
def get_image(category, style, filename):
    image_path = os.path.join(IMAGE_DIR, category, style, filename)
    if os.path.isfile(image_path):
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        return "Image not found", 404

# Function to get the full image path
def get_image_path(category, style, filename):
    base_dir = os.path.join(IMAGE_DIR, "Furniture_Data")
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
