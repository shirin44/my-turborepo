"use client";
import React from 'react';
import { Typography, Button } from 'antd';
import { CameraOutlined, HeartOutlined, StarOutlined, RocketOutlined } from '@ant-design/icons';

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

  const handleTask12Click = () => {
    window.location.href = "/task1-2";
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="w-4/5 p-10 rounded-lg bg-white shadow-lg">
        <Title className="text-6xl mb-8 text-gray-800 text-center">Furniture Recommender</Title>
        <Paragraph className=" text-3xl mb-8 text-gray-900 text-center">
          This website showcases the results of our machine learning project. 
          Users can explore furniture recommendations and more!
        </Paragraph>
        <div className="text-2xl mb-8 text-gray-900">
          <Title level={2}>Team Number: 7</Title>
          <Title level={2}>Team Members:</Title>
          <ul className="list-disc list-inside">
            <li>Shirin Shujaa - S3983427</li>
            <li>Nguyen Vu Thuy Duong - S3865443</li>
            <li>Huynh Quang Dong - S3938006</li>
            <li>Nguyen Bao Minh - S3926080</li>
            <li>Tran Viet Hoang - S3928141</li>
          </ul>
        </div>
        <div className="flex flex-col md:flex-row justify-center space-y-4 md:space-y-0 md:space-x-4">
          <Button type="primary" className="w-full md:w-auto md:h-auto" style={{ backgroundColor: '#888', borderColor: '#888', fontSize: '1.6rem', padding: '20px' }} icon={<CameraOutlined />} onClick={handleTask1Click}>
            Image Classification using Resnet 
          </Button>
          <Button type="primary" className="w-full md:w-auto md:h-auto" style={{ backgroundColor: '#888', borderColor: '#888', fontSize: '1.6rem', padding: '20px' }} icon={<RocketOutlined />} onClick={handleTask12Click}>
            Image Classification using Inception
          </Button>
        </div>
        <div className="flex flex-col md:flex-row justify-center space-y-4 md:space-y-0 md:space-x-4 mt-4">
          <Button type="primary" className="w-full md:w-auto md:h-auto" style={{ backgroundColor: '#888', borderColor: '#888', fontSize: '1.6rem', padding: '20px' }} icon={<HeartOutlined />} onClick={handleTask2Click}>
            Furniture Recommendation
          </Button>
          <Button type="primary" className="w-full md:w-auto md:h-auto" style={{ backgroundColor: '#888', borderColor: '#888', fontSize: '1.6rem', padding: '20px' }} icon={<StarOutlined />} onClick={handleTask3Click}>
            Style Recognition
          </Button>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
