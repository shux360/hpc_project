# Extended benchmark script for HPC Project - MPI and Hybrid
# Tests MPI and Hybrid configurations

$imagePath = "images/noisy/1024X1024_noise.jpg"
$trials = 3
$mpiProcesses = @(1, 2, 4, 8)
$hybridConfigs = @(@(1,1), @(2,2), @(4,4), @(2,4), @(4,2))  # (processes, threads)

Write-Host "Starting MPI and Hybrid benchmarks" -ForegroundColor Green
Write-Host ""

$outputFile = "benchmark_results_mpi_hybrid.csv"
"Implementation,Config,Trial,ExecutionTime_ms" | Out-File $outputFile

# Test if MPI is available
try {
    $mpiTest = & mpirun.exe --version 2>$null
    $mpiAvailable = $?
} catch {
    $mpiAvailable = $false
}

if (-not $mpiAvailable) {
    Write-Host "WARNING: MPI not found. Install Intel MPI or Open MPI." -ForegroundColor Yellow
} else {
    # MPI Benchmarks
    Write-Host "=== MPI ===" -ForegroundColor Yellow
    foreach ($procs in $mpiProcesses) {
        Write-Host "Processes: $procs"
        for ($trial = 0; $trial -lt $trials; $trial++) {
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            & mpirun.exe -np $procs "./mpi/denoise_mpi_edge" $imagePath 2>&1 | Out-Null
            $stopwatch.Stop()
            $elapsed = $stopwatch.Elapsed.TotalMilliseconds
            
            Write-Host "  Trial $($trial+1): $([Math]::Round($elapsed, 2)) ms"
            "MPI_$($procs)procs,$procs,$trial,$([Math]::Round($elapsed, 2))" | Add-Content $outputFile
        }
    }
}

# Hybrid Benchmarks
Write-Host ""
Write-Host "=== HYBRID (MPI + OpenMP) ===" -ForegroundColor Yellow
foreach ($config in $hybridConfigs) {
    $procs = $config[0]
    $threads = $config[1]
    Write-Host "Config: $procs processes × $threads threads/process"
    
    for ($trial = 0; $trial -lt $trials; $trial++) {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        & mpirun.exe -np $procs "./hybrid/denoise_hybrid" $imagePath $threads 2>&1 | Out-Null
        $stopwatch.Stop()
        $elapsed = $stopwatch.Elapsed.TotalMilliseconds
        
        Write-Host "  Trial $($trial+1): $([Math]::Round($elapsed, 2)) ms"
        "Hybrid_$($procs)x$($threads),$($procs)x$($threads),$trial,$([Math]::Round($elapsed, 2))" | Add-Content $outputFile
    }
}

Write-Host ""
Write-Host "Results saved to $outputFile" -ForegroundColor Green
