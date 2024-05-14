"use client";

import React, { useState, useEffect } from "react";
import { Typography, Button, message } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import * as tf from "@tensorflow/tfjs";

const { Title, Paragraph } = Typography;

const Task2 = () => {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [recommendedItems, setRecommendedItems] = useState<string[]>([]);
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [model, setModel] = useState<tf.LayersModel | null>(null);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      console.log("File uploaded:", file);
      setUploadedImage(file); // Set the file object, not the data URL
      const reader = new FileReader();
      reader.onload = () => {
        // Optionally, you can set the data URL as well if needed
        // setUploadedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const loadedModel = await tf.loadLayersModel(
          "http://localhost:3000/model.json"
        );
        setModel(loadedModel);
        setModelLoaded(true);
        console.log("Model loaded successfully:", loadedModel);
      } catch (error) {
        console.error("Error loading model:", error);
      }
    };

    loadModel();
  }, []);

  const fetchRecommendations = async () => {
    if (uploadedImage && modelLoaded && model) {
      try {
        console.log("Preparing to fetch recommendations...");

        const formData = new FormData();
        formData.append("image", uploadedImage);

        console.log("Sending request to server...");
        const response = await fetch(
          "http://localhost:5000/api/recommendations",
          {
            method: "POST",
            body: formData,
          }
        );

        console.log("Response:", response);

        if (response.ok) {
          const data = await response.json();
          console.log("Received recommendations:", data);
          setRecommendedItems(data);
        } else {
          throw new Error("Failed to fetch recommendations");
        }
      } catch (error) {
        console.error("Error fetching recommendations:", error);
        message.error("Failed to get recommendations.");
      }
    } else {
      message.error("Please upload an image and ensure the model is loaded.");
    }
  };

  const handleFetchRecommendations = () => {
    fetchRecommendations();
  };

  const handleGoBack = () => {
    window.location.href = "/";
  };

  return (
    <div className="flex h-screen bg-gray-100 flex-1 flex-col">
      {/* Top section */}
      <div className="flex justify-center items-center h-1/6 text-xl border-b border-black bg-gray-200">
        <div className="p-6">
          <Title level={2}>Task 2: Furniture Recommendation</Title>
          <Paragraph>
            Recommend 10 furniture items in our dataset which are similar to the
            input furniture item image from users.
          </Paragraph>
        </div>
      </div>
      {/* Middle section */}
      <div className="flex-1 flex bg-gray-300">
        {/* Upload area */}
        <div className="w-1/2 h-full flex flex-col justify-center items-center border-r border-black">
          <input
            type="file"
            accept="image/*"
            onChange={handleUpload}
            className="hidden"
            id="uploadInput"
          />
          <label
            htmlFor="uploadInput"
            className="w-2/3 h-3/4 flex justify-center items-center border rounded-lg overflow-hidden cursor-pointer bg-white border-black"
          >
            <div className="text-center">
              {uploadedImage ? (
                <img
                  src={URL.createObjectURL(uploadedImage)}
                  alt="Uploaded"
                  className="max-w-full max-h-full"
                />
              ) : (
                <>
                  <UploadOutlined
                    style={{ fontSize: "32px", color: "#1890ff" }}
                  />
                  <Paragraph className="mt-2 text-gray-800">
                    Click or drag image to upload
                  </Paragraph>
                </>
              )}
            </div>
          </label>
          <Button
            type="primary"
            onClick={handleFetchRecommendations}
            className="mt-4"
          >
            Fetch Recommendations
          </Button>
        </div>
        {/* Recommended images area */}
        <div className="w-1/2 bg-white p-4">
          <Paragraph className="mb-4">Recommended Furniture Items</Paragraph>
          {/* Display recommended items in two columns */}
          <div className="grid grid-cols-2 gap-4">
            // Inside Task2 component
            {recommendedItems.map((imageUrl, index) => (
              <img
                key={index}
                src={encodeURI(`http://localhost:5000/${imageUrl}`)} // Adjusted image path
                alt={`Recommended ${index + 1}`}
                className="mb-2 max-w-xs"
                style={{ maxWidth: "100%", height: "auto" }} // Maintain aspect ratio
              />
            ))}
          </div>
        </div>
      </div>
      {/* Bottom section */}
      <div className="flex justify-center items-center h-1/6 text-xl border-t border-black bg-gray-200">
        <Button type="primary" onClick={handleGoBack} className="w-1/3 h-1/3">
          Return
        </Button>
      </div>
    </div>
  );
};

export default Task2;
