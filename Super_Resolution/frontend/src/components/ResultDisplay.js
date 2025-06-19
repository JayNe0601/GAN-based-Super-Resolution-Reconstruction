import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Image, Spin, Typography, Button, Space } from 'antd';
import { DownloadOutlined, ReloadOutlined } from '@ant-design/icons';
import styled from '@emotion/styled';
import axios from 'axios';

const { Title, Text } = Typography;

const StyledCard = styled(Card)`
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`;

const ImageContainer = styled.div`
  position: relative;
  width: 100%;
  height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
  border-radius: 4px;
  overflow: hidden;
`;

const ResultDisplay = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/results');
      setResults(response.data);
    } catch (error) {
      console.error('获取结果失败：', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, []);

  const handleDownload = (url, filename) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <StyledCard>
      <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={4}>处理结果</Title>
        <Button 
          icon={<ReloadOutlined />} 
          onClick={fetchResults}
          loading={loading}
        >
          刷新
        </Button>
      </div>
      
      {loading ? (
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" />
          <Text style={{ display: 'block', marginTop: 16 }}>正在加载结果...</Text>
        </div>
      ) : results.length > 0 ? (
        <Row gutter={[16, 16]}>
          {results.map((result, index) => (
            <Col xs={24} sm={12} md={8} key={index}>
              <Card>
                <ImageContainer>
                  <Image
                    src={result.originalUrl}
                    alt="原始图片"
                    style={{ maxWidth: '100%', maxHeight: '100%' }}
                  />
                </ImageContainer>
                <div style={{ marginTop: 16 }}>
                  <Text strong>原始图片</Text>
                  <Space style={{ float: 'right' }}>
                    <Button 
                      icon={<DownloadOutlined />}
                      onClick={() => handleDownload(result.originalUrl, `original_${index}.png`)}
                    >
                      下载
                    </Button>
                  </Space>
                </div>
                
                <ImageContainer style={{ marginTop: 16 }}>
                  <Image
                    src={result.srUrl}
                    alt="超分辨率图片"
                    style={{ maxWidth: '100%', maxHeight: '100%' }}
                  />
                </ImageContainer>
                <div style={{ marginTop: 16 }}>
                  <Text strong>超分辨率图片</Text>
                  <Space style={{ float: 'right' }}>
                    <Button 
                      icon={<DownloadOutlined />}
                      onClick={() => handleDownload(result.srUrl, `sr_${index}.png`)}
                    >
                      下载
                    </Button>
                  </Space>
                </div>
                
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">PSNR: {result.psnr?.toFixed(2)} dB</Text>
                  <br />
                  <Text type="secondary">SSIM: {result.ssim?.toFixed(4)}</Text>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      ) : (
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Text type="secondary">暂无处理结果</Text>
        </div>
      )}
    </StyledCard>
  );
};

export default ResultDisplay; 