const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

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

function parseRMSE(outputText) {
  const match = outputText.match(/RMSE:\s*([0-9.]+)/i);
  return match ? parseFloat(match[1]) : null;
}

function calculateRMSE(serialEdgesPath, methodEdgesPath) {
  return new Promise((resolve) => {
    try {
      console.log(`[RMSE] Calculating RMSE between:`);
      console.log(`[RMSE]   Baseline: ${serialEdgesPath}`);
      console.log(`[RMSE]   Method:   ${methodEdgesPath}`);
      
      // Check if files exist
      const serialExists = fs.existsSync(serialEdgesPath);
      const methodExists = fs.existsSync(methodEdgesPath);
      
      console.log(`[RMSE] Baseline exists: ${serialExists}`);
      console.log(`[RMSE] Method exists:   ${methodExists}`);
      
      if (!serialExists || !methodExists) {
        console.log(`[RMSE] One or both files missing, returning RMSE=0`);
        resolve(0);
        return;
      }

      // Use Windows Python with OpenCV (which is installed on Windows)
      const pythonScript = `
import cv2
import numpy as np
import sys

try:
    img1 = cv2.imread('${serialEdgesPath.replace(/\\/g, '\\\\')}', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('${methodEdgesPath.replace(/\\/g, '\\\\')}', cv2.IMREAD_GRAYSCALE)
    
    print(f'DEBUG: img1 loaded = {img1 is not None}', file=sys.stderr)
    print(f'DEBUG: img2 loaded = {img2 is not None}', file=sys.stderr)
    
    if img1 is None or img2 is None:
        print('RMSE:0')
        print(f'DEBUG: Images could not be loaded', file=sys.stderr)
        sys.exit(0)
    
    print(f'DEBUG: img1 shape = {img1.shape}', file=sys.stderr)
    print(f'DEBUG: img2 shape = {img2.shape}', file=sys.stderr)
    
    if img1.shape != img2.shape:
        print(f'DEBUG: Resizing img2 to match img1', file=sys.stderr)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    diff = img1.astype(float) - img2.astype(float)
    mse = np.mean(diff * diff)
    rmse = np.sqrt(mse)
    
    print(f'DEBUG: mse = {mse}, rmse = {rmse}', file=sys.stderr)
    print(f'RMSE:{rmse:.6f}')
except Exception as e:
    print(f'DEBUG: Exception occurred: {e}', file=sys.stderr)
    print('RMSE:0')
`;

      // Write Python script to temp file
      const tmpFile = path.join(path.dirname(methodEdgesPath), `rmse_calc_${Date.now()}.py`);
      fs.writeFileSync(tmpFile, pythonScript);
      console.log(`[RMSE] Python script written to: ${tmpFile}`);
      
      // Execute using Windows Python
      const pythonCmd = `python "${tmpFile}"`;
      console.log(`[RMSE] Executing Python...`);
      
      exec(pythonCmd, { maxBuffer: 1024 * 1024 * 10, timeout: 30000 }, (error, stdout, stderr) => {
        // Clean up temp file
        try {
          fs.unlinkSync(tmpFile);
        } catch (e) {}
        
        const output = stdout.trim();
        const stderrOutput = stderr.trim();
        
        console.log(`[RMSE] Python stdout: "${output}"`);
        if (stderrOutput) {
          console.log(`[RMSE] Python debug output:`);
          stderrOutput.split('\n').forEach(line => {
            if (line.trim()) console.log(`[RMSE]   ${line}`);
          });
        }
        
        const rmse = parseRMSE(output);
        console.log(`[RMSE] Final parsed RMSE: ${rmse}`);
        
        resolve(rmse !== null && !isNaN(rmse) ? rmse : 0);
      });
    } catch (e) {
      console.error('[RMSE] Exception in calculateRMSE:', e.message);
      resolve(0);
    }
  });
}

function buildCommand(method, imagePath, threads, processes) {
  const projectRoot = path.resolve(__dirname, '../../');

  switch (method) {
    case 'serial': {
      const exe = path.join(projectRoot, 'serial', 'denoise_serial_edge');
      return { exe, args: [imagePath] };
    }

    case 'openmp': {
      const exe = path.join(projectRoot, 'openmp', 'denoise_openmp_edge');
      return { exe, args: [imagePath, String(threads || 4)] };
    }

    case 'pthreads': {
      const exe = path.join(projectRoot, 'pthreads', 'denoise_pthreads_edge');
      return { exe, args: [imagePath, String(threads || 4)] };
    }

    case 'mpi': {
      const exe = path.join(projectRoot, 'mpi', 'denoise_mpi_edge');
      return { 
        exe, 
        args: [imagePath],
        isMPI: true,
        processes: processes || 2
      };
    }

    case 'hybrid': {
      const exe = path.join(projectRoot, 'hybrid', 'denoise_hybrid');
      return { 
        exe, 
        args: [imagePath, String(threads || 4)],
        isMPI: true,
        processes: processes || 2
      };
    }

    default:
      throw new Error('Invalid method');
  }
}

function getOutputFiles(method) {
  const projectRoot = path.resolve(__dirname, '../../');

  const map = {
    serial: {
      denoised: path.join(projectRoot, 'serial_denoised.png'),
      edges: path.join(projectRoot, 'serial_edges.png')
    },
    openmp: {
      denoised: path.join(projectRoot, 'openmp_denoised.png'),
      edges: path.join(projectRoot, 'openmp_edges.png')
    },
    pthreads: {
      denoised: path.join(projectRoot, 'pthread_denoised.png'),
      edges: path.join(projectRoot, 'pthread_edges.png')
    },
    mpi: {
      denoised: path.join(projectRoot, 'mpi_denoised.png'),
      edges: path.join(projectRoot, 'mpi_edges.png')
    },
    hybrid: {
      denoised: path.join(projectRoot, 'hybrid_denoised.png'),
      edges: path.join(projectRoot, 'hybrid_edges.png')
    }
  };

  return map[method];
}

function copyResultToPublic(srcPath, targetName) {
  const destination = path.join(resultsDir, targetName);
  fs.copyFileSync(srcPath, destination);
  return `/results/${targetName}`;
}

function runCommand(cmdObj) {
  return new Promise((resolve, reject) => {
    // Convert Windows paths to WSL paths
    const wslExe = cmdObj.exe.replace(/\\/g, '/').replace(/^([A-Z]):/, (match, drive) => `/mnt/${drive.toLowerCase()}`);
    const wslArgs = cmdObj.args.map(arg => 
      arg.replace(/\\/g, '/').replace(/^([A-Z]):/, (match, drive) => `/mnt/${drive.toLowerCase()}`)
    );
    
    // Log the paths for debugging
    console.log(`Windows exe path: ${cmdObj.exe}`);
    console.log(`WSL exe path: ${wslExe}`);
    console.log(`Windows args: ${JSON.stringify(cmdObj.args)}`);
    console.log(`WSL args: ${JSON.stringify(wslArgs)}`);
    
    // Convert project root for WSL
    const projectRoot = path.resolve(__dirname, '../../');
    const wslProjectRoot = projectRoot.replace(/\\/g, '/').replace(/^([A-Z]):/, (match, drive) => `/mnt/${drive.toLowerCase()}`);
    
    let command;
    if (cmdObj.isMPI) {
      // MPI commands - change to project directory first
      command = `cd ${wslProjectRoot} && mpirun -np ${cmdObj.processes} ${wslExe} ${wslArgs.join(' ')}`;
    } else {
      // Non-MPI commands - change to project directory first
      command = `cd ${wslProjectRoot} && ${wslExe} ${wslArgs.join(' ')}`;
    }
    
    console.log(`Executing via WSL: ${command}`);
    
    // First test if file exists in WSL
    const testCmd = `wsl.exe -d Ubuntu -- test -f ${wslExe} && echo "File exists"`;
    console.log(`Testing file existence: ${testCmd}`);
    
    exec(testCmd, { timeout: 5000 }, (testError, testOutput) => {
      if (testError) {
        console.error(`File not found at ${wslExe}`);
        console.error(`Test output: ${testOutput}`);
        // Try listing the parent directory
        const parentDir = wslExe.substring(0, wslExe.lastIndexOf('/'));
        const lsCmd = `wsl.exe -d Ubuntu -- ls -la ${parentDir}`;
        console.log(`Listing directory: ${lsCmd}`);
        exec(lsCmd, { timeout: 5000 }, (lsError, lsOutput) => {
          console.log(`Directory listing:\n${lsOutput}`);
        });
      }
      
      // Execute the command with working directory context
      const wslCmd = `wsl.exe -d Ubuntu -- sh -c "${command}"`;
      console.log(`Final WSL invocation: ${wslCmd}`);
      
      exec(wslCmd, { maxBuffer: 1024 * 1024 * 20, timeout: 120000 }, (error, stdout, stderr) => {
        if (error) {
          console.error(`Command error: ${error.message}`);
          console.error(`stderr: ${stderr}`);
          console.error(`stdout: ${stdout}`);
          reject(stderr || error.message);
        } else {
          console.log(`Command successful`);
          console.log(`Output: ${stdout}`);
          resolve(stdout);
        }
      });
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

    const output = await runCommand(cmdObj);
    const executionTime = parseExecutionTime(output);

    const files = getOutputFiles(method);

    if (!fs.existsSync(files.denoised) || !fs.existsSync(files.edges)) {
      return res.status(500).json({ error: 'Expected output images were not generated' });
    }

    const stamp = Date.now();
    const denoisedUrl = copyResultToPublic(files.denoised, `${method}_denoised_${stamp}.png`);
    const edgesUrl = copyResultToPublic(files.edges, `${method}_edges_${stamp}.png`);

    let rmse = null;

    if (method === 'serial') {
      // Serial is the baseline, RMSE is 0
      rmse = 0;
      // Save the serial edges as the comparison baseline for subsequent methods
      const projectRoot = path.resolve(__dirname, '../../');
      const baselineEdgesPath = path.join(projectRoot, '_serial_edges_baseline.png');
      try {
        fs.copyFileSync(files.edges, baselineEdgesPath);
        console.log(`[BASELINE] Saved serial baseline to: ${baselineEdgesPath}`);
      } catch (e) {
        console.error(`[BASELINE] Error saving baseline: ${e.message}`);
      }
    } else {
      const projectRoot = path.resolve(__dirname, '../../');
      const baselineEdgesPath = path.join(projectRoot, '_serial_edges_baseline.png');
      const methodEdgesPath = files.edges;
      
      if (fs.existsSync(baselineEdgesPath)) {
        rmse = await calculateRMSE(baselineEdgesPath, methodEdgesPath);
        console.log(`[RMSE] Calculated RMSE for ${method}: ${rmse}`);
      } else {
        console.log(`[RMSE] Serial baseline not found at ${baselineEdgesPath}, skipping RMSE`);
        rmse = null;
      }
    }

    res.json({
      success: true,
      method,
      executionTime,
      rmse,
      denoisedImage: denoisedUrl,
      edgeImage: denoisedUrl ? edgesUrl : null,
      rawOutput: output
    });
  } catch (error) {
    res.status(500).json({
      error: typeof error === 'string' ? error : error.message
    });
  }
});

const server = app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});

server.on('error', (err) => {
  console.error('Server error:', err);
});
