import React, { useState } from 'react';

const UploadPanel = () => {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');
  const [jsonUrl, setJsonUrl] = useState('');
  const [summary, setSummary] = useState(null);
  const [strategy, setStrategy] = useState('vehicle_cost');

  const [customDestination, setCustomDestination] = useState('');
  const [selectedProductSpec, setSelectedProductSpec] = useState('');
  const [productQuantity, setProductQuantity] = useState('');
  const [vehicleType, setVehicleType] = useState('60t');
  const [vehicleQuantity, setVehicleQuantity] = useState('');

  const productOptions = [
    { label: '酸洗卷 - 厚1.56 宽1180 外径1414 重10.52t', value: '1.56-1180-1414-10.52' },
    { label: '酸洗卷 - 厚1.56 宽1180 外径1422 重10.66t', value: '1.56-1180-1422-10.66' },
    { label: '热轧卷 - 厚1.6 宽1087 外径1185 重6.78t', value: '1.6-1087-1185-6.78' },
    { label: '热轧卷 - 厚1.6 宽1407 外径1120 重7.87t', value: '1.6-1407-1120-7.87' },
    { label: '冷轧卷 - 厚2 宽1144 外径1212 重7.64t', value: '2-1144-1212-7.64' },
    { label: '冷轧卷 - 厚1.36 宽1110 外径1213 重7.57t', value: '1.36-1110-1213-7.57' },
    { label: '冷轧卷 - 厚1.36 宽1110 外径1272 重8.60t', value: '1.36-1110-1272-8.60' }
  ];

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setStatus('');
    setPdfUrl('');
    setJsonUrl('');
    setSummary(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus('请先选择文件');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('strategy', strategy);

    setStatus('上传中...');

    try {
      const response = await fetch('/api/process_excel/', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (data.status === 'success') {
        setStatus(data.message || '上传成功');
        setPdfUrl(data.data.pdf_url);
        setJsonUrl(data.data.json_url);
        setSummary(data.data.summary);
      } else {
        setStatus(data.message || '上传失败');
      }
    } catch (err) {
      console.error(err);
      setStatus('上传失败，请检查服务器');
    }
  };

  const handleAddProduct = async () => {
    if (!customDestination || !selectedProductSpec || !productQuantity) {
      setStatus('请先选择目的地、产品规格并输入数量');
      return;
    }

    const payload = {
      destination: customDestination,
      product_spec: selectedProductSpec,
      quantity: productQuantity
    };

    try {
      const response = await fetch('/api/add_product/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();

      if (data.status === 'success') {
        setStatus(data.message || '添加成功');
        // 可选：刷新 summary 简化为只显示目的地和数量
        const res = await fetch('/api/download_json/current_grouped_data.json');
        if (res.ok) {
          const updated = await res.json();
          setSummary(updated.grouped_data);
        }
      } else {
        setStatus(data.message || '添加失败');
      }
    } catch (err) {
      console.error(err);
      setStatus('添加失败，请检查服务器');
    }
  };

  const handleGenerateFromCurrent = async () => {
    setStatus('正在生成方案...');
    try {
      const response = await fetch('/api/generate_from_current/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy })
      });
      const data = await response.json();

      if (data.status === 'success') {
        setStatus(data.message || '生成成功');
        setPdfUrl(data.data.pdf_url);
        setJsonUrl(data.data.json_url);
        setSummary(data.data.summary);
      } else {
        setStatus(data.message || '生成失败');
      }
    } catch (err) {
      console.error(err);
      setStatus('生成失败，请检查服务器');
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>钢卷配载系统</h2>

      <div style={{ marginBottom: '10px' }}>
        <label>策略选择：</label>
        <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          <option value="vehicle_cost">🚚 Vehicle Cost Priority</option>
          <option value="vehicle_optimization">📦 Vehicle Optimization Priority</option>
          <option value="balance">⚖️ Balance Priority</option>
        </select>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label>新增目的地：</label>
        <select value={customDestination} onChange={(e) => setCustomDestination(e.target.value)}>
          <option value="">请选择</option>
          <option value="北京">北京</option>
          <option value="上海">上海</option>
          <option value="广州">广州</option>
          <option value="深圳">深圳</option>
          <option value="重庆">重庆</option>
          <option value="成都">成都</option>
          <option value="武汉">武汉</option>
          <option value="西安">西安</option>
        </select>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label>选择产品规格：</label>
        <select value={selectedProductSpec} onChange={(e) => setSelectedProductSpec(e.target.value)}>
          <option value="">请选择</option>
          {productOptions.map((opt, idx) => (
            <option key={idx} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label>产品数量：</label>
        <input
          type="number"
          min="1"
          value={productQuantity}
          onChange={(e) => setProductQuantity(e.target.value)}
          placeholder="请输入数量"
        />
        <button onClick={handleAddProduct} style={{ marginLeft: '10px' }}>➕ 添加产品</button>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label>车辆类型：</label>
        <select value={vehicleType} onChange={(e) => setVehicleType(e.target.value)}>
          <option value="60t">60t: 13000mm x 3000mm</option>
          <option value="70t">70t: 14000mm x 3200mm</option>
        </select>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <label>车辆数量：</label>
        <input
          type="number"
          min="1"
          value={vehicleQuantity}
          onChange={(e) => setVehicleQuantity(e.target.value)}
          placeholder="请输入数量"
        />
      </div>

      <input type="file" accept=".xlsx,.xls" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginTop: '10px' }}>上传并处理 Excel</button>
      <button onClick={handleGenerateFromCurrent} style={{ marginLeft: '10px' }}>📊 从现有数据生成方案</button>

      <p><strong>状态：</strong>{status}</p>

      {pdfUrl && (
        <p><a href={pdfUrl} download>📄 下载配载 PDF</a></p>
      )}
      {jsonUrl && (
        <p><a href={jsonUrl} download>🧾 下载结构化 JSON</a></p>
      )}

      {summary && (
        <div>
          <h4>📊 装载统计摘要</h4>
          {Object.entries(summary).map(([destination, cars], idx) => (
            <div key={idx} style={{ marginBottom: '10px', padding: '5px', border: '1px solid #ccc' }}>
              <strong>目的地：</strong> {destination} <br />
              <strong>产品数：</strong> {cars.length}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default UploadPanel;
