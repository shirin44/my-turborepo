"use client";

import React, { useEffect, useState } from "react";
import { Typography, Button, message, Input, Spin, Alert } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import * as tf from "@tensorflow/tfjs";

const { Title, Paragraph } = Typography;

const Task1 = () => {
  const [label, setLabel] = useState<string>("");
  const [model, setModel] = useState<any>();
  const [uploadedImage, setUploadedImage] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    tf.ready().then(() => {
      loadModel("http://localhost:3000/model.json");
    });
  }, []);

  async function loadModel(url: string) {
    try {
      const model = await tf.loadLayersModel(url);
      setModel(model);
      console.log("Load model success", model);
    } catch (err: any) {
      console.log(err.message);
    }
  }

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLoading(true);
    setError(null);
    const file = e.target.files && e.target.files[0];
    if (file) {
      console.log("Image uploaded.");
      message.success(`${file.name} file uploaded successfully.`);
      const imageUrl = URL.createObjectURL(file);
      setUploadedImage(imageUrl); // Display uploaded image
      classifyImage(file);
    }
  };

  const classifyImage = async (imageFile: File) => {
    try {
      const img = new Image();
      const reader = new FileReader();

      reader.onload = async (e: ProgressEvent<FileReader>) => {
        if (e.target && e.target.result) {
          img.src = e.target.result as string;
          img.onload = async () => {
            // Load the image data into a tensor
            const tensor = tf.browser.fromPixels(img);
            // Preprocess the image
            const processedImg = preprocessImage(tensor);
            // Make predictions using the loaded model
            const predictions = model.predict(processedImg) as tf.Tensor;
            // Get the predicted class label
            const predictedLabel = await getPredictedLabel(predictions);
            // Set the label state with the predicted class
            setLabel(predictedLabel);
            setLoading(false);
          };
        }
      };

      reader.readAsDataURL(imageFile);
    } catch (error) {
      setError("Error classifying image. Please try again.");
      setLoading(false);
    }
  };

  const preprocessImage = (image: tf.Tensor3D) => {
    // Resize the image to match the input shape of the model
    const resizedImg = tf.image.resizeBilinear(image, [224, 224]);
    // Convert the image tensor to float32 and normalize it
    const processedImg = resizedImg.toFloat().div(tf.scalar(255));
    // Expand the dimensions to match the input shape of the model
    const expandedImg = processedImg.expandDims();
    return expandedImg;
  };

  const getPredictedLabel = async (predictions: tf.Tensor): Promise<string> => {
    // Get the index of the class with the highest probability
    const predictedClassIndex = predictions.argMax(1).dataSync()[0];
    // Retrieve the class labels from your dataset
    const classLabels = ["bed", "chair", "dresser", "lamp", "sofa", "table"];
    // Get the label corresponding to the predicted class index
    const predictedLabel = classLabels[predictedClassIndex];
    return predictedLabel;
  };

  const handleGoBack = () => {
    window.location.href = "/";
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Top section */}
      <div className="flex justify-center items-center h-1/6 text-xl border-b border-gray-300 bg-white">
        <div className="p-6 text-center">
          <Title level={2} className="text-gray-800">
            Task 1: Classify Images by Furniture Category
          </Title>
          <Paragraph className="text-gray-600">
            Upload an image of furniture to classify it into one of the
            following categories: beds, chairs, dressers, lamps, sofas, or
            tables.
          </Paragraph>
        </div>
      </div>
      {/* Middle section */}
      <div className="flex-1 flex items-center justify-center text-xl relative">
        {/* Upload area */}
        <input
          type="file"
          accept="image/*"
          onChange={handleUpload}
          className="hidden"
          id="uploadInput"
        />
        <label
          htmlFor="uploadInput"
          className="w-2/3 h-3/4 flex justify-center items-center border border-gray-300 rounded-lg overflow-hidden cursor-pointer bg-white shadow-sm hover:bg-gray-200 transition duration-300 ease-in-out"
        >
          <div className="text-center">
            {uploadedImage ? (
              <img
                src={uploadedImage}
                alt="Uploaded"
                className="max-w-full max-h-full"
              />
            ) : (
              <>
                <UploadOutlined
                  style={{ fontSize: "32px", color: "#6B7280" }}
                />
                <Paragraph className="mt-2 text-gray-600">
                  Click or drag image to upload
                </Paragraph>
              </>
            )}
          </div>
        </label>
        {/* Label display */}
        <div
          style={{
            position: "absolute",
            top: "0",
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 1,
            animation: label ? "pulse 1s infinite" : "none"
          }}
        >
          <Input
            value={label}
            placeholder="Classified Label"
            style={{
              width: "300px",
              fontSize: "24px",
              padding: "15px",
              borderRadius: "10px",
              boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
              backgroundColor: "white",
              color: "black",
              transition: "all 0.3s ease-in-out"
            }}
          />
        </div>
        {/* Loading spinner */}
        {loading && (
          <Spin className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
        )}
        {/* Error message */}
        {error && (
          <Alert
            message={error}
            type="error"
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
          />
        )}
      </div>
      {/* Bottom section */}
      <div className="flex justify-center items-center h-1/6 text-xl border-t border-gray-300 bg-white">
        <Button
          type="primary"
          onClick={handleGoBack}
          className="w-1/3 h-1/3"
          style={{ backgroundColor: "#6B7280", borderColor: "#6B7280" }}
        >
          Back
        </Button>
      </div>
    </div>
  );
};

export default Task1;
