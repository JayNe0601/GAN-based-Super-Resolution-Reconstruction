import React from 'react';
import { Layout, Typography } from 'antd';
import styled from '@emotion/styled';
import ImageUploader from './components/ImageUploader';
import ResultDisplay from './components/ResultDisplay';

const { Header, Content } = Layout;
const { Title } = Typography;

const StyledLayout = styled(Layout)`
  min-height: 100vh;
`;

const StyledHeader = styled(Header)`
  background: #fff;
  padding: 0 50px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
`;

const StyledContent = styled(Content)`
  padding: 24px 50px;
  background: #f0f2f5;
`;

const App = () => {
  return (
    <StyledLayout>
      <StyledHeader>
        <Title level={2} style={{ margin: '16px 0' }}>
          图像超分辨率重建系统
        </Title>
      </StyledHeader>
      <StyledContent>
        <ImageUploader />
        <ResultDisplay />
      </StyledContent>
    </StyledLayout>
  );
};

export default App;