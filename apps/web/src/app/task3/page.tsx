"use client";
import React, { useState } from 'react';
import { Typography, message, Spin, Button } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

const { Title, Paragraph } = Typography;

const Task3: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [predictedCategory, setPredictedCategory] = useState<string>('');
  const [predictedStyle, setPredictedStyle] = useState<string>('');
  const [recommendations, setRecommendations] = useState<any[]>([]);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      const file = event.target.files[0];
      setSelectedFile(file);
      await handleSubmit(file);
    }
  };

  const handleSubmit = async (file: File) => {
    setLoading(true);
    setPredictedCategory('');
    setPredictedStyle('');
    setRecommendations([]);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('http://localhost:5000/api/recommendations_task3', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();

      setPredictedCategory(data.predictedCategory);
      setPredictedStyle(data.predictedStyle);
      setRecommendations(data.recommendations);
    } catch (error) {
      console.error('Error uploading file:', error);
      message.error('Failed to upload file.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 flex-col">
      <div className="flex justify-center items-center h-1/6 text-xl border-b border-black bg-gray-200">
        <div className="p-6">
          <Title level={2}>Task 3: Image Classification and Recommendations</Title>
          <Paragraph>
            Upload an image to get recommendations for similar furniture items.
          </Paragraph>
          {predictedCategory && predictedStyle && (
            <Paragraph>
              Predicted Category: {predictedCategory}, Predicted Style: {predictedStyle}
            </Paragraph>
          )}
        </div>
      </div>
      <div className="flex-1 flex bg-gray-300">
        <div className="w-1/2 h-full flex flex-col justify-center items-center border-r border-black">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            id="uploadInput"
          />
          <label
            htmlFor="uploadInput"
            className="w-2/3 h-3/4 flex justify-center items-center border rounded-lg overflow-hidden cursor-pointer bg-white border-black"
          >
            <div className="text-center">
              {selectedFile ? (
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Uploaded"
                  className="max-w-full max-h-full"
                />
              ) : (
                <>
                  <UploadOutlined
                    style={{ fontSize: '32px', color: '#1890ff' }}
                  />
                  <Paragraph className="mt-2 text-gray-800">
                    Click or drag image to upload
                  </Paragraph>
                </>
              )}
            </div>
          </label>
        </div>
        <div className="w-1/2 bg-white p-4">
          <Paragraph className="mb-4">Recommended Furniture Items</Paragraph>
          {loading ? (
            <div className="flex justify-center items-center h-full">
              <Spin size="large" />
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {recommendations.map((recommendation: any, index: number) => (
                <div key={index} className="flex flex-col items-center">
                  <img
                    src={`http://localhost:5000/${recommendation.Img}`}
                    alt={`Recommended ${index + 1}`}
                    className="mb-2 max-w-xs"
                    style={{ maxWidth: '100%', height: 'auto' }}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="flex justify-center items-center h-1/6 text-xl border-t border-black bg-gray-200">
        <Button type="primary" className="w-1/3 h-1/3">
          Return
        </Button>
      </div>
    </div>
  );
};

export default Task3;
