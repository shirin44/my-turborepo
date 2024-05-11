"use client";

import React, { useState } from 'react';
import { Typography, Button, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

const { Title, Paragraph } = Typography;

const Task2 = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target) {
          // Set the uploaded image to display
          setUploadedImage(event.target.result as string);
        }
      };
      reader.readAsDataURL(file);
      message.success(`${file.name} file uploaded successfully.`);
    }
  };

  const handleGoBack = () => {
    window.location.href = '/';
  };

  return (
    <div className="flex h-screen bg-gray-100 flex-1 flex-col">
      {/* Top section */}
      <div className="flex justify-center items-center h-1/6 text-xl border-b border-black bg-gray-200">
        <div className="p-6">
          <Title level={2}>Task 2: Furniture Recommendation</Title>
          <Paragraph>
            Recommend 10 furniture items in our dataset which are similar to the input furniture item image from users.
          </Paragraph>
        </div>
      </div>
      {/* Middle section */}
      <div className="flex-1 flex bg-gray-300">
        {/* Upload area */}
        <div className="w-2/3 h-full flex flex-col justify-center items-center border-r border-black">
          <input
            type="file"
            accept="image/*"
            onChange={handleUpload}
            className="hidden"
            id="uploadInput"
          />
          <label htmlFor="uploadInput" className="w-2/3 h-3/4 flex justify-center items-center border rounded-lg overflow-hidden cursor-pointer bg-white border-black">
            <div className="text-center">
              {uploadedImage ? (
                <img src={uploadedImage} alt="Uploaded" className="max-w-full max-h-full" />
              ) : (
                <>
                  <UploadOutlined style={{ fontSize: '32px', color: '#1890ff' }} />
                  <Paragraph className="mt-2 text-gray-800">Click or drag image to upload</Paragraph>
                </>
              )}
            </div>
          </label>
        </div>
        {/* Recommended images area */}
        <div className="w-1/3 bg-white">
          <div className="p-4">
            <Paragraph className="mb-4">Recommended Furniture Items</Paragraph>
            {/* Placeholder for recommended images */}
            <div className="flex justify-center">
              {/* Recommended images go here */}
            </div>
          </div>
        </div>
      </div>
      {/* Bottom section */}
      <div className="flex justify-center items-center h-1/6 text-xl border-t border-black bg-gray-200">
        <Button type="primary" onClick={handleGoBack} className="w-1/3 h-1/3">Return</Button>
      </div>
    </div>
  );
};

export default Task2;
