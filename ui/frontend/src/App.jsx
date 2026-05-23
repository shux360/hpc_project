import { useState } from 'react';
import { processImage } from './api';
import './App.css';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [method, setMethod] = useState('openmp');
  const [threads, setThreads] = useState(4);
  const [processes, setProcesses] = useState(2);
  const [blockSize, setBlockSize] = useState(16);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [allResults, setAllResults] = useState([]);
  const [serialTime, setSerialTime] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (event) => {
        setFilePreview(event.target.result);
      };
      reader.readAsDataURL(file);
      setError('');
    }
  };

  const handleRun = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await processImage({
        file: selectedFile,
        method,
        threads: parseInt(threads),
        processes: parseInt(processes),
        blockSize: parseInt(blockSize)
      });
      setResult(response);
      
      // Track serial time for speedup calculation
      if (method === 'serial') {
        setSerialTime(response.executionTime);
      }
    } catch (err) {
      const msg = err.message || String(err);
      if (msg.includes('CUDA executable not found')) {
        setError(
          'CUDA is not available on this machine.\n\n' +
          'CUDA requires an NVIDIA GPU. To test it:\n' +
          '1. Open colab_cuda_test.ipynb in Google Colab\n' +
          '2. Enable GPU runtime (Runtime > Change runtime type > T4 GPU)\n' +
          '3. Run all cells to verify the CUDA code works\n\n' +
          'Once you have a GPU machine, compile with:\n' +
          'cd cuda && nvcc -O2 denoise_cuda.cu -o cuda_denoise_edge $(pkg-config --cflags --libs opencv4)'
        );
      } else {
        setError(msg || 'Error processing image');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleProcessAll = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');
    const results = [];

    const methods = ['serial', 'openmp', 'pthreads', 'mpi', 'hybrid', 'cuda'];
    for (const m of methods) {
      try {
        const response = await processImage({
          file: selectedFile,
          method: m,
          threads: m === 'serial' ? 1 : parseInt(threads),
          processes: 1,
          blockSize: parseInt(blockSize)
        });
        results.push(response);
        // Track the first (serial) execution time for speedup calculation
        if (m === 'serial') {
          setSerialTime(response.executionTime);
        }
      } catch (err) {
        setError((prev) => prev + `\n${m}: ${err.message}`);
      }
    }

    setAllResults(results);
    setLoading(false);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>HPC Image Processing</h1>
        <p>Denoise & Edge Detection Benchmarking</p>
      </header>

      <div className="main-layout">
        <div className="panel upload-panel">
          <h2>Upload Image</h2>
          <div className="upload-box">
            {filePreview ? (
              <img src={filePreview} alt="Preview" className="image-preview" />
            ) : (
              <div className="placeholder">
                <p>Select an image to preview</p>
              </div>
            )}
          </div>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="file-input"
            disabled={loading}
          />
          {selectedFile && (
            <p className="file-info">
              {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
            </p>
          )}
        </div>

        <div className="panel control-panel">
          <h2>Settings</h2>

          <div className="control-group">
            <label>Processing Method</label>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              disabled={loading}
            >
              <option value="serial">Serial (Baseline)</option>
              <option value="openmp">OpenMP</option>
              <option value="pthreads">Pthreads</option>
              <option value="mpi">MPI</option>
              <option value="hybrid">Hybrid (OpenMP + MPI)</option>
              <option value="cuda">CUDA (GPU)</option>
            </select>
          </div>

          <div className="control-group">
            <label>Threads: {threads}</label>
            <input
              type="range"
              min="1"
              max="16"
              value={threads}
              onChange={(e) => setThreads(Number(e.target.value))}
              disabled={loading || method === 'serial' || method === 'cuda'}
              className="slider"
            />
          </div>

          <div className="control-group">
            <label>Processes: {processes}</label>
            <input
              type="range"
              min="1"
              max="8"
              value={processes}
              onChange={(e) => setProcesses(Number(e.target.value))}
              disabled={loading || !['mpi', 'hybrid'].includes(method)}
              className="slider"
            />
          </div>

          {method === 'cuda' && (
            <div className="control-group">
              <label>Block Size: {blockSize}×{blockSize} ({blockSize * blockSize} threads/block)</label>
              <select
                value={blockSize}
                onChange={(e) => setBlockSize(Number(e.target.value))}
                disabled={loading}
                className="block-size-select"
              >
                <option value={8}>8×8 — 64 threads/block (fine grain)</option>
                <option value={16}>16×16 — 256 threads/block (recommended)</option>
                <option value={32}>32×32 — 1024 threads/block (max occupancy)</option>
              </select>
              <p className="control-hint">
                Grid: {'≈'} {Math.ceil(1024 / blockSize)}×{Math.ceil(1024 / blockSize)} blocks for a 1024×1024 image.
                Requires a CUDA-capable NVIDIA GPU.
              </p>
            </div>
          )}

          <div className="button-group">
            <button
              onClick={handleRun}
              disabled={loading || !selectedFile}
              className="btn btn-primary"
            >
              {loading ? 'Processing...' : 'Process Image'}
            </button>
            <button
              onClick={handleProcessAll}
              disabled={loading || !selectedFile}
              className="btn btn-secondary"
            >
              {loading ? 'Comparing...' : 'Compare All'}
            </button>
          </div>

          {error && (
            <div className="error-box">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        <div className="panel results-panel">
          <h2>Results</h2>
          {result ? (
            <div className="result-card">
              <div className="result-header">
                <span className="method-badge">{result.method.toUpperCase()}</span>
              </div>
              <div className="result-metrics">
                <div className="metric">
                  <span className="metric-label">Execution Time</span>
                  <span className="metric-value">{result.executionTime?.toFixed(2)} ms</span>
                </div>
                <div className="metric">
                  <span className="metric-label">RMSE</span>
                  <span className="metric-value">{result.rmse !== null ? result.rmse.toFixed(4) : 'N/A'}</span>
                </div>
                {serialTime && result.method !== 'serial' && (
                  <div className="metric">
                    <span className="metric-label">Speedup</span>
                    <span className="metric-value">{(serialTime / result.executionTime).toFixed(2)}x</span>
                  </div>
                )}
              </div>
              {result.denoisedImage && result.edgeImage && (
                <div className="result-images">
                  <div className="image-container">
                    <h4>Denoised</h4>
                    <img src={result.denoisedImage} alt={`${result.method} denoised`} className="result-image" />
                  </div>
                  <div className="image-container">
                    <h4>Edges</h4>
                    <img src={result.edgeImage} alt={`${result.method} edges`} className="result-image" />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="placeholder">
              <p>Run processing to see results</p>
            </div>
          )}
        </div>
      </div>

      {allResults.length > 0 && (
        <div className="comparison-section">
          <h2>Comparison Table</h2>
          <div className="comparison-table">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Execution Time (ms)</th>
                  <th>RMSE</th>
                  <th>Speedup</th>
                </tr>
              </thead>
              <tbody>
                {allResults.map((res, idx) => {
                  const serialTime = allResults[0]?.executionTime || 1;
                  const speedup = (serialTime / (res.executionTime || 1)).toFixed(2);
                  return (
                    <tr key={idx}>
                      <td className="method-col">{res.method}</td>
                      <td>{res.executionTime?.toFixed(2)}</td>
                      <td>{res.rmse !== null && res.rmse !== undefined ? res.rmse.toFixed(4) : 'N/A'}</td>
                      <td className="speedup-col">{speedup}x</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <h2>Detailed Results</h2>
          <div className="all-results-container">
            {allResults.map((res, idx) => (
              <div key={idx} className="result-item">
                <div className="result-header">
                  <span className="method-badge">{res.method?.toUpperCase()}</span>
                  <div className="result-metrics-inline">
                    <span>{res.executionTime?.toFixed(2)} ms</span>
                    <span>RMSE: {res.rmse !== null && res.rmse !== undefined ? res.rmse.toFixed(4) : 'N/A'}</span>
                    {res.method !== 'serial' && (
                      <span>{((allResults[0]?.executionTime || 1) / (res.executionTime || 1)).toFixed(2)}x speedup</span>
                    )}
                  </div>
                </div>
                {res.denoisedImage && res.edgeImage && (
                  <div className="result-images-row">
                    <div className="image-box">
                      <img src={res.denoisedImage} alt={`${res.method} denoised`} />
                      <p>Denoised</p>
                    </div>
                    <div className="image-box">
                      <img src={res.edgeImage} alt={`${res.method} edges`} />
                      <p>Edges</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
