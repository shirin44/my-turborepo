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
import tensorflow as tf
import random
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Constants
FURNITURE_DATA_DIR = "Furniture_Data"
IMAGE_DIR = "Furniture_Data"
CATEGORY_LABELS = {
    0: "Beds",
    1: "Chairs",
    2: "Dressers",
    3: "Lamps",
    4: "Sofas",
    5: "Tables"
}

STYLE_LABELS = {
    0: "Asian",
    1: "Beach",
    2: "Contemporary",
    3: "Craftsman",
    4: "Eclectic",
    5: "Farmhouse",
    6: "Industrial",
    7: "Mediterranean",
    8: "Midcentury",
    9: "Modern",
    10: "Rustic",
    11: "Scandinavian",
    12: "Southwestern",
    13: "Traditional",
    14: "Transitional",
    15: "Tropical",
    16: "Victorian"
}

# Load models
knn_model = load("KNN_ResNet_V2S.joblib")
resnet_model = load_model("ResNet_Furniture_Classification.h5", compile=False)
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

resnet_furniture_model = load_model("ResNet_Furniture_Classification.h5", compile=False)


chairs_style_model = load_model("task3/Chairs_Style_Classification.h5", compile=False)
chairs_style_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

sofas_style_model = load_model("task3/ResNet_Furniture_Classification_sofasv4.h5", compile=False)
sofas_style_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

beds_style_model = load_model("task3/Beds_Style_Classification.h5", compile=False)
beds_style_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

tables_style_model = load_model("task3/Tables_Style_Classification.h5", compile=False)
tables_style_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


dressers_style_model = load_model("task3/ResNet_Furniture_Classification_dressersv2.h5", compile=False)
dressers_style_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preprocess the image
def preprocess_image3(image):
    resized_img = tf.image.resize(image, [224, 224])
    processed_img = resized_img / 255.0
    expanded_img = tf.expand_dims(processed_img, axis=0)
    return expanded_img

def classify_image(image):
    category_label = resnet_furniture_model.predict(image)
    category_name = CATEGORY_LABELS[np.argmax(category_label)]

    if category_name == "Chairs":
        style_label = chairs_style_model.predict(image)
    elif category_name == "Beds":
        style_label = beds_style_model.predict(image)
    elif category_name == "sofas":
        style_label = sofas_style_model.predict(image)
    elif category_name == "tables":
        style_label = tables_style_model.predict(image)
    elif category_name == "dressers":
        style_label = dressers_style_model.predict(image)
    

    style_name = STYLE_LABELS[np.argmax(style_label)]

    return category_name, style_name

# Get random images from directory
def get_random_images(directory, num_images=10):
    images = os.listdir(directory)
    random_images = random.sample(images, min(num_images, len(images)))
    return [os.path.join(directory, image) for image in random_images]

# Route to handle recommendations request
@app.route('/api/recommendations_task3', methods=['POST'])
def get_recommendations_task3():
    try:
        uploaded_image = request.files['image']
        img = Image.open(uploaded_image)
        img = img.resize((224, 224))
        img_array = img_preprocessing.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        category_name, style_name = classify_image(img_array)

        category_folder = os.path.join(FURNITURE_DATA_DIR, category_name)
        style_folder = os.path.join(category_folder, style_name)

        recommended_images = get_random_images(style_folder, num_images=10)

        response_data = {
            'predictedCategory': category_name,
            'predictedStyle': style_name,
            'recommendations': recommended_images
        }

        return jsonify(response_data)
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify(error='Failed to process request'), 500

# Load image DataFrame
image_df = pd.read_csv("df.csv")

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    img_array = preprocess_input(img_array)
    return img_array

# Extract features from the preprocessed image
def extract_features(img_array):
    features = resnet_model.predict(img_array)
    return features.flatten()

# Find similar images based on extracted features
def find_similar_images(input_features, knn_model, k=10):
    distances, indices = knn_model.kneighbors(input_features.reshape(1, -1), n_neighbors=k)
    return distances, indices

# Load image DataFrame and feature matrix
image_df = pd.read_csv("df.csv")
features_matrix = np.load("image_features_v2.npy")

# Preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    img_array = preprocess_input(img_array)
    return img_array

# Extract features from the preprocessed image
def extract_features(image_pil, model):
    img = img_preprocessing.img_to_array(image_pil)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    features = model.predict(img)
    return features.flatten()

# Find most similar images based on extracted features
def get_most_similar_images(query_features, features_matrix, df, top_n=10):
    similarity_scores = cosine_similarity([query_features], features_matrix).flatten()
    sorted_indices = np.argsort(similarity_scores)[::-1]
    most_similar_indices = sorted_indices[:top_n]
    similar_images = [(df['Img'][i], df['Category'][i], df['Style'][i], similarity_scores[i]) for i in most_similar_indices]
    return similar_images

# Route to handle recommendations request
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        uploaded_image = request.files['image']
        img = Image.open(uploaded_image)
        img_array = preprocess_image(img)
        query_features = extract_features(img, resnet_model)

        # Logging the extracted features
        logging.info(f"Extracted features: {query_features}")

        similar_images = get_most_similar_images(query_features, features_matrix, image_df)

        # Logging the similar images
        logging.info(f"Similar images: {similar_images}")

        recommended_images = []
        for img_path, category, style, score in similar_images:
            if os.path.exists(img_path):
                recommended_images.append({'path': img_path, 'category': category, 'style': style, 'score': float(score)})

        response_data = {
            'recommendations': recommended_images,
            'extracted_features': query_features.tolist()
        }

        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        traceback.print_exc()
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
