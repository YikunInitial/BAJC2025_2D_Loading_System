// src/components/UploadPanel.jsx
import React, { useState } from 'react';
import axios from 'axios';

const UploadPanel = () => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [pdfUrl, setPdfUrl] = useState("");
  const [jsonUrl, setJsonUrl] = useState("");
  const [summary, setSummary] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("请先选择文件");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setStatus("正在上传，请稍候...");
      const response = await axios.post("/api/upload_excel", formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });

      const data = response.data;
      setStatus(data.message || "上传成功");
      setPdfUrl(data.pdf_url);
      setJsonUrl(data.json_url);
      setSummary(data.summary);
    } catch (err) {
      console.error(err);
      setStatus("上传失败，请检查服务器是否启动并返回正确格式");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>钢卷配载系统</h2>
      <input type="file" accept=".xlsx" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginLeft: "10px" }}>
        上传并处理
      </button>

      <div style={{ marginTop: "20px" }}>
        <p><strong>状态：</strong>{status}</p>
        {pdfUrl && (
          <p>
            📄 <a href={pdfUrl} target="_blank" rel="noopener noreferrer">下载 PDF 报告</a>
          </p>
        )}
        {jsonUrl && (
          <p>
            📦 <a href={jsonUrl} target="_blank" rel="noopener noreferrer">下载 JSON 文件</a>
          </p>
        )}
        {summary && (
          <div style={{ marginTop: "20px" }}>
            <h4>📊 配载摘要</h4>
            <p>总车皮数：{summary.total_freight_cars}</p>
            {summary.destinations.map((d, i) => (
              <div key={i}>
                <p>🚚 目的地：{d.destination}</p>
                <p> 产品数量：{d.product_count}</p>
                <p> 使用车皮：{d.freight_cars_used}</p>
                <p> 平均装载率：{d.average_loading_rate_percent.toFixed(2)}%</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPanel;
