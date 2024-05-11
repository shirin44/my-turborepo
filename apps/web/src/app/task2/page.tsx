"use client";
import React, { useState } from 'react';
import { Typography, Button, Upload, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

const { Title, Paragraph } = Typography;

const Task2 = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const handleUpload = (info: any) => {
    const { status } = info.file;
    if (status !== 'uploading') {
      console.log(info.file, info.fileList);
    }
    if (status === 'done') {
      message.success(`${info.file.name} file uploaded successfully.`);
      // Set the uploaded image to display
      setUploadedImage(URL.createObjectURL(info.file.originFileObj));
    } else if (status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
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
          <Upload
            action=""
            beforeUpload={() => false}
            showUploadList={false}
            onChange={handleUpload}
            className="w-full h-3/4 flex justify-center items-center rounded-lg overflow-hidden"
          >
            {uploadedImage ? (
              <img src={uploadedImage} alt="Uploaded" style={{ maxWidth: '100%', maxHeight: '100%' }} />
            ) : (
              <div className="text-center">
                <UploadOutlined style={{ fontSize: '32px' }} />
                <Paragraph className="mt-2">Click or drag image to upload</Paragraph>
              </div>
            )}
          </Upload>
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
