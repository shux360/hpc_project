const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const { Jimp } = require('jimp');

console.log('Server.js loading...');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

const uploadsDir = path.join(__dirname, 'uploads');
const resultsDir = path.join(__dirname, 'results');

if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

app.use('/results', express.static(resultsDir));

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    cb(null, `input_${Date.now()}${ext}`);
  }
});

const upload = multer({ storage });

function parseExecutionTime(outputText) {
  const match = outputText.match(/execution time:\s*([0-9.]+)/i);
  return match ? parseFloat(match[1]) : null;
}

async function calculateRMSE(serialEdgesPath, methodEdgesPath) {
  const serialImg = await Jimp.read(serialEdgesPath);
  const methodImg = await Jimp.read(methodEdgesPath);

  const width = serialImg.bitmap.width;
  const height = serialImg.bitmap.height;

  if (methodImg.bitmap.width !== width || methodImg.bitmap.height !== height) {
    await methodImg.resize({ width, height });
  }

  let sumSquaredDiff = 0;
  const pixelCount = width * height;
  
  for (let i = 0; i < pixelCount; i++) {
    // Extract RGBA from 32-bit pixel data
    const serialColor = serialImg.bitmap.data[i * 4];
    const methodColor = methodImg.bitmap.data[i * 4];
    
    const diff = Math.abs(serialColor - methodColor);
    sumSquaredDiff += diff * diff;
  }

  const mse = sumSquaredDiff / pixelCount;
  const rmse = Math.sqrt(mse);
  
  return parseFloat(rmse.toFixed(4));
}

function getOutputFiles(method) {
  const projectRoot = path.resolve(__dirname, '../../');
  
  const nameMap = {
    'serial': 'serial',
    'openmp': 'openmp',
    'pthreads': 'pthread',
    'mpi': 'mpi',
    'hybrid': 'hybrid',
    'cuda': 'cuda'
  };
  
  const baseName = nameMap[method] || method;
  
  return {
    denoised: path.join(projectRoot, `${baseName}_denoised.png`),
    edges: path.join(projectRoot, `${baseName}_edges.png`)
  };
}

function copyResultToPublic(sourceFile, destFileName) {
  if (!fs.existsSync(sourceFile)) {
    return null;
  }
  const destPath = path.join(resultsDir, destFileName);
  fs.copyFileSync(sourceFile, destPath);
  return `/results/${destFileName}`;
}

function buildCommand(method, imagePath, threads, processes, blockSize) {
  const projectRoot = path.resolve(__dirname, '../../');
  
  const threadCount = Math.max(1, parseInt(threads) || 4);
  const processCount = Math.max(1, parseInt(processes) || 1);
  
  const exePaths = {
    'serial': path.join(projectRoot, 'serial', 'denoise_serial_edge'),
    'openmp': path.join(projectRoot, 'openmp', 'denoise_openmp_edge'),
    'pthreads': path.join(projectRoot, 'pthreads', 'denoise_pthreads_edge'),
    'mpi': path.join(projectRoot, 'mpi', 'denoise_mpi_edge'),
    'hybrid': path.join(projectRoot, 'hybrid', 'denoise_hybrid'),
    'cuda': path.join(projectRoot, 'cuda', 'cuda_denoise_edge')
  };

  const exe = exePaths[method] || exePaths['serial'];
  const isMPI = method === 'mpi' || method === 'hybrid';
  let args = [imagePath];
  
  if (method === 'openmp' || method === 'pthreads' || method === 'hybrid' || method === 'cuda') {
    args.push(String(threadCount));
  }

  return { exe, args, isMPI, method, threads: threadCount, processes: processCount };
}

function runCommand(cmdObj) {
  return new Promise((resolve, reject) => {
    const wslExe = cmdObj.exe.replace(/\\/g, '/').replace(/^([A-Z]):/, (match, drive) => `/mnt/${drive.toLowerCase()}`);
    const wslArgs = cmdObj.args.map(arg => arg.replace(/\\/g, '/').replace(/^([A-Z]):/, (match, drive) => `/mnt/${drive.toLowerCase()}`));

    const projectRoot = path.resolve(__dirname, '../../').replace(/\\/g, '/').replace(/^([A-Z]):/, (match, drive) => `/mnt/${drive.toLowerCase()}`);
    
    let command;
    if (cmdObj.isMPI) {
      command = `cd ${projectRoot} && mpirun -np ${cmdObj.processes} ${wslExe} ${wslArgs.join(' ')}`;
    } else {
      command = `cd ${projectRoot} && ${wslExe} ${wslArgs.join(' ')}`;
    }

    console.log(`[EXEC] Running: ${command}`);
    const wslCmd = `wsl.exe -d Ubuntu -- sh -c "${command}"`;

    exec(wslCmd, { maxBuffer: 1024 * 1024 * 20, timeout: 120000 }, (error, stdout, stderr) => {
      if (error) {
        console.error(`[ERROR] Execution failed: ${error.message}`);
        reject(new Error(`${cmdObj.method} execution failed: ${stderr || error.message}`));
      } else {
        console.log(`[SUCCESS] ${cmdObj.method} completed\n${stdout}`);
        resolve(stdout);
      }
    });
  });
}

app.post('/api/process', upload.single('image'), async (req, res) => {
  try {
    const { method, threads, processes } = req.body;

    if (!req.file) {
      return res.status(400).json({ error: 'Image is required' });
    }

    if (!method) {
      return res.status(400).json({ error: 'Method is required' });
    }

    const imagePath = req.file.path;
    const cmdObj = buildCommand(method, imagePath, threads, processes);

    let executionTime = 0;
    let output = '';
    
    // Execute the compiled binary
    try {
      output = await runCommand(cmdObj);
      executionTime = parseExecutionTime(output);
      console.log(`[SUCCESS] ${method} executed in ${executionTime}ms`);
    } catch (cmdError) {
      console.error(`[ERROR] ${cmdError.message}`);
      return res.status(500).json({ error: cmdError.message });
    }

    const files = getOutputFiles(method);
    
    // Verify output files exist
    if (!fs.existsSync(files.denoised) || !fs.existsSync(files.edges)) {
      const missing = [];
      if (!fs.existsSync(files.denoised)) missing.push('denoised');
      if (!fs.existsSync(files.edges)) missing.push('edges');
      return res.status(500).json({ 
        error: `${method} did not produce output: missing ${missing.join(', ')}` 
      });
    }

    // Copy outputs to results folder
    const stamp = Date.now();
    const denoisedUrl = copyResultToPublic(files.denoised, `${method}_denoised_${stamp}.png`);
    const edgesUrl = copyResultToPublic(files.edges, `${method}_edges_${stamp}.png`);

    let rmse = 0;
    
    if (method === 'serial') {
      rmse = 0; // Serial is baseline
      // Save as baseline for comparison
      const projectRoot = path.resolve(__dirname, '../../');
      const baselineEdgesPath = path.join(projectRoot, '_serial_edges_baseline.png');
      fs.copyFileSync(files.edges, baselineEdgesPath);
      console.log(`[BASELINE] Serial baseline saved`);
    } else {
      // Calculate RMSE against serial baseline
      const projectRoot = path.resolve(__dirname, '../../');
      const baselineEdgesPath = path.join(projectRoot, '_serial_edges_baseline.png');
      
      if (!fs.existsSync(baselineEdgesPath)) {
        return res.status(500).json({ error: 'Serial baseline not found. Process serial first.' });
      }
      
      console.log(`[RMSE] Calculating for ${method}...`);
      rmse = await calculateRMSE(baselineEdgesPath, files.edges);
      console.log(`[RMSE] ${method}: ${rmse}`);
    }

    res.json({
      success: true,
      method,
      threads: cmdObj.threads,
      processes: cmdObj.processes,
      executionTime: parseFloat(executionTime.toFixed(4)),
      rmse,
      denoisedImage: denoisedUrl,
      edgeImage: edgesUrl
    });

  } catch (err) {
    console.error(`[ERROR] ${err.message}`);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});
