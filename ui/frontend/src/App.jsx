import { useState } from 'react';
import { processImage } from './api';
import './App.css';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [method, setMethod] = useState('openmp');
  const [threads, setThreads] = useState(4);
  const [processes, setProcesses] = useState(2);
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
        processes: parseInt(processes)
      });
      setResult(response);
      
      // Track serial time for speedup calculation
      if (method === 'serial') {
        setSerialTime(response.executionTime);
      }
    } catch (err) {
      setError(err.message || 'Error processing image');
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

    const methods = ['serial', 'openmp', 'pthreads', 'mpi', 'hybrid'];
    for (const m of methods) {
      try {
        const response = await processImage({
          file: selectedFile,
          method: m,
          threads: m === 'serial' ? 1 : parseInt(threads),
          processes: 1
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
              disabled={loading || method === 'serial'}
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
                      <td>{res.rmse?.toFixed(4) || 'N/A'}</td>
                      <td className="speedup-col">{speedup}x</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
