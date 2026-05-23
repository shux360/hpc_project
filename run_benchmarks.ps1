# Benchmark runner script for HPC Project
# Runs all parallel implementations and collects timing data

$imagePath = "images/noisy/1024X1024_noise.jpg"
$trials = 5
$threadCounts = @(1, 2, 4, 8, 16)

# Create output CSV
$outputFile = "benchmark_results.csv"
"Implementation,Threads,Trial,ExecutionTime_ms" | Out-File $outputFile

Write-Host "Starting benchmarks with image: $imagePath" -ForegroundColor Green
Write-Host ""

# Serial Baseline
Write-Host "=== SERIAL BASELINE ===" -ForegroundColor Yellow
for ($trial = 0; $trial -lt $trials; $trial++) {
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    & "./serial/denoise_serial_edge" $imagePath | Out-Null
    $stopwatch.Stop()
    $elapsed = $stopwatch.Elapsed.TotalMilliseconds
    
    Write-Host "Trial $($trial+1): $([Math]::Round($elapsed, 2)) ms"
    "Serial,1,$trial,$([Math]::Round($elapsed, 2))" | Add-Content $outputFile
}

# OpenMP
Write-Host ""
Write-Host "=== OPENMP ===" -ForegroundColor Yellow
foreach ($threads in $threadCounts) {
    Write-Host "Threads: $threads"
    for ($trial = 0; $trial -lt $trials; $trial++) {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        & "./openmp/denoise_openmp_edge" $imagePath $threads 2>&1 | Out-Null
        $stopwatch.Stop()
        $elapsed = $stopwatch.Elapsed.TotalMilliseconds
        
        Write-Host "  Trial $($trial+1): $([Math]::Round($elapsed, 2)) ms"
        "OpenMP,$threads,$trial,$([Math]::Round($elapsed, 2))" | Add-Content $outputFile
    }
}

# Pthreads
Write-Host ""
Write-Host "=== PTHREADS ===" -ForegroundColor Yellow
foreach ($threads in $threadCounts) {
    Write-Host "Threads: $threads"
    for ($trial = 0; $trial -lt $trials; $trial++) {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        & "./pthreads/denoise_pthreads_edge" $imagePath $threads 2>&1 | Out-Null
        $stopwatch.Stop()
        $elapsed = $stopwatch.Elapsed.TotalMilliseconds
        
        Write-Host "  Trial $($trial+1): $([Math]::Round($elapsed, 2)) ms"
        "Pthreads,$threads,$trial,$([Math]::Round($elapsed, 2))" | Add-Content $outputFile
    }
}

Write-Host ""
Write-Host "Benchmarks completed. Results saved to $outputFile" -ForegroundColor Green
