
"use client";
import React, { useState, useEffect } from "react";
import { Typography, Button, message, Spin, Modal } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import * as tf from "@tensorflow/tfjs";

const { Title, Paragraph } = Typography;

const Task2 = () => {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [recommendedItems, setRecommendedItems] = useState<any[]>([]);
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [extractedFeatures, setExtractedFeatures] = useState<number[]>([]);
  const [modalVisible, setModalVisible] = useState<boolean>(false);
  const [selectedFeatures, setSelectedFeatures] = useState<number[] | null>(null);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      console.log("File uploaded:", file);
      setUploadedImage(file);
      fetchRecommendations(file);
    }
  };

  const fetchRecommendations = async (imageFile: File) => {
    if (modelLoaded && model) {
      try {
        setLoading(true);

        const formData = new FormData();
        formData.append("image", imageFile);

        const response = await fetch(
          "http://localhost:5000/api/recommendations",
          {
            method: "POST",
            body: formData,
          }
        );

        if (response.ok) {
          const data = await response.json();
          setRecommendedItems(data.recommendations);
          setExtractedFeatures(data.extracted_features);
        } else {
          throw new Error("Failed to fetch recommendations");
        }
      } catch (error) {
        console.error("Error fetching recommendations:", error);
        message.error("Failed to get recommendations.");
      } finally {
        setLoading(false);
      }
    } else {
      message.error("Please ensure the model is loaded.");
    }
  };

  const handleGoBack = () => {
    window.location.href = "/";
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

  const showFeatures = (features: number[]) => {
    setSelectedFeatures(features);
    setModalVisible(true);
  };

  return (
    <div className="flex h-screen bg-gray-100 flex-1 flex-col">
      <div className="flex justify-center items-center h-1/6 text-xl border-b border-black bg-gray-200">
        <div className="p-6 text-3xl">
          <Title level={1}>Task 2: Furniture Recommendation</Title>
          <Paragraph className=" text-3xl">
            Recommend 10 furniture items in our dataset which are similar to the
            input furniture item image from users.
          </Paragraph>
        </div>
      </div>
      <div className="flex-1 flex bg-gray-300">
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
          {uploadedImage && (
            <Button
              type="link"
              onClick={() => showFeatures(extractedFeatures)}
            >
              Show Uploaded Image Features
            </Button>
          )}
        </div>
        <div className="w-1/2 bg-white p-4">
          <Paragraph className="mb-4 text-2xl">Recommended Furniture Items</Paragraph>
          {loading ? (
            <div className="flex justify-center items-center h-full">
              <Spin size="large" />
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {recommendedItems.map((item, index) => (
                <div key={index} className="flex flex-col items-center">
                  <img
                    src={`http://localhost:5000/${item.path}`}
                    alt={`Recommended ${index + 1}`}
                    className="mb-2 max-w-xs"
                    style={{ maxWidth: "100%", height: "auto" }}
                  />
                  <span className="text-sm text-gray-600">
                    Similarity Score: {item.score.toFixed(2)}
                  </span>
                  
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="flex justify-center items-center h-1/6 text-xl border-t border-black bg-gray-200">
        <Button type="primary" onClick={handleGoBack} className="w-1/3 h-1/3">
          Return
        </Button>
      </div>
      <Modal
        title="Extracted Features"
        visible={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>
            Close
          </Button>,
        ]}
      >
        <pre>{JSON.stringify(selectedFeatures, null, 2)}</pre>
      </Modal>
    </div>
  );
};

export default Task2;
