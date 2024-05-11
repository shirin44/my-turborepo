
"use client";
// LandingPage.js
import React from 'react';
import { Typography, Button } from 'antd';

const { Title, Paragraph } = Typography;

const LandingPage = () => {
  const handleTask1Click = () => {
    window.location.href = "/task1";
  };

  const handleTask2Click = () => {
    window.location.href = "/task2";
  };

  const handleTask3Click = () => {
    window.location.href = "/task3";
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="w-2/3 p-8 rounded-lg bg-white shadow-lg">
        <Title className="text-4xl mb-6 text-gray-800">Furniture Recommender</Title>
        <Paragraph className="text-lg mb-6 text-gray-900">
          This website showcases the results of our machine learning project. 
          Users can explore furniture recommendations and more!
        </Paragraph>
        <div className="text-lg mb-6 text-gray-900">
          <Title level={3}>Team Number: 7</Title>
          <Title level={3}>Team Members:</Title>
          <ul className="list-disc list-inside">
            <li>Shirin Shujaa - S3983427</li>
            <li>Nguyen Vu Thuy Duong - S3865443</li>
            <li>Huynh Quang Dong - S3938006</li>
            <li>Nguyen Bao Minh - S3926080</li>
            <li>Tran Viet Hoang - S3928141</li>
          </ul>
        </div>
        <div className="flex justify-center">
          <Button type="primary" className="mr-4 bg-gray-800 hover:bg-gray-900" onClick={handleTask1Click}>
            Task 1: Furniture Image Classification
          </Button>
          <Button type="primary" className="mr-4 bg-gray-800 hover:bg-gray-900" onClick={handleTask2Click}>
            Task 2: Furniture Recommendation
          </Button>
          <Button type="primary" className="bg-gray-800 hover:bg-gray-900" onClick={handleTask3Click}>
            Task 3: Furniture Style Recognition
          </Button>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
