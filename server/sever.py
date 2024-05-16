from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
from joblib import load
from PIL import Image
import traceback
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load KNN model and ResNet model
knn_model = load("KNN_ResNet_V2S.joblib")
resnet_model = load_model("ResNet_Furniture_Classification_local.h5", compile=False)
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load image DataFrame
image_df = pd.read_csv("df.csv")

# Define directory for serving images
IMAGE_DIR = "Furniture_Data"

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = preprocess_input(img_array)  # Ensure consistent preprocessing
    return img_array

# Extract features from the preprocessed image
def extract_features(img_array):
    features = resnet_model.predict(img_array)
    return features.flatten()

# Find similar images based on extracted features
def find_similar_images(input_features, knn_model, k=10):
    distances, indices = knn_model.kneighbors(input_features.reshape(1, -1), n_neighbors=k)
    return distances, indices

# Define route to handle recommendations request
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Get the uploaded image file from the request
        uploaded_image = request.files['image']

        # Preprocess the uploaded image
        img = Image.open(uploaded_image)
        img_array = preprocess_image(img)

        # Extract features using ResNet model
        input_image_features = extract_features(img_array)

        # Query k-NN model for recommendations
        distances, indices = find_similar_images(input_image_features, knn_model)

        # Get filenames and similarity scores of recommended images
        recommended_images = []

        for index in indices[0]:
            image_path = image_df.iloc[index]['Img']
                    
            if not os.path.exists(image_path):
                print(f"Image not found at path: {image_path}")
                continue  # Skip this image if the file does not exist

            similar_img = Image.open(image_path)
            similar_img_resized = similar_img.resize((224, 224))
            similar_image_features = extract_features(np.expand_dims(img_preprocessing.img_to_array(similar_img_resized) / 255.0, axis=0))
            similarity_score = cosine_similarity([input_image_features], [similar_image_features])[0][0]
            recommended_images.append({'path': image_path, 'score': float(similarity_score)})
            print("Recommended Image Features:", similar_image_features)

        # Sort recommended images by similarity score
        recommended_images.sort(key=lambda x: x['score'], reverse=True)

        # Select the top 10 recommendations
        top_10_recommendations = recommended_images[:10]

        # Extracted features to be sent along with recommendations
        extracted_features = input_image_features.tolist()

        # Prepare response
        response_data = {
            'recommendations': top_10_recommendations,
            'extracted_features': extracted_features
        }

        return jsonify(response_data)
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()  # Print the traceback for detailed error information
        return jsonify(error='Failed to process request'), 500

# Route to serve images
@app.route('/Furniture_Data/<path:filepath>')
def get_image(filepath):
    image_path = os.path.join(IMAGE_DIR, filepath)
    if os.path.isfile(image_path):
        response = send_file(image_path, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    else:
        return "Image not found", 404

# Run the Flask app
if __name__ == '__main__':
    print("Starting the Flask server...")
    app.run(debug=True)
