# Extended benchmark script for HPC Project - MPI and Hybrid
# Tests MPI and Hybrid configurations and records only successful executions.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$imagePath = "images/noisy/1024X1024_noise.jpg"
$trials = 3
$mpiProcesses = @(1, 2, 4, 8)
$hybridConfigs = @(@(1, 1), @(2, 2), @(4, 4), @(2, 4), @(4, 2)) # (processes, threads)
$outputFile = "benchmark_results_mpi_hybrid.csv"
$stagingFile = "$outputFile.tmp"

function Get-MpiLauncher {
    foreach ($candidate in @("mpirun", "mpirun.exe", "mpiexec", "mpiexec.exe")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $command) {
            return $command.Source
        }
    }

    throw "No MPI launcher was found. Install MPI or run this benchmark in the Linux environment where mpirun is available."
}

function Assert-RequiredFile([string] $path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required benchmark input does not exist: $path"
    }
}

function Invoke-Benchmark {
    param(
        [string] $Implementation,
        [string] $Config,
        [int] $Processes,
        [int] $Trial,
        [string] $Executable,
        [string[]] $Arguments
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $programOutput = & $mpiLauncher -np $Processes $Executable @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $stopwatch.Stop()

    if ($exitCode -ne 0) {
        throw "$Implementation failed for config $Config (exit code $exitCode).`n$($programOutput -join [Environment]::NewLine)"
    }

    $elapsed = [Math]::Round($stopwatch.Elapsed.TotalMilliseconds, 2)
    Write-Host "  Trial $($Trial + 1): $elapsed ms"
    "$Implementation,$Config,$Trial,$elapsed" | Add-Content -Path $stagingFile -Encoding utf8
}

Write-Host "Starting MPI and Hybrid benchmarks" -ForegroundColor Green

$mpiLauncher = Get-MpiLauncher
Assert-RequiredFile $imagePath
Assert-RequiredFile "./mpi/denoise_mpi_edge"
Assert-RequiredFile "./hybrid/denoise_hybrid"

try {
    "Implementation,Config,Trial,ExecutionTime_ms" | Set-Content -Path $stagingFile -Encoding utf8

    Write-Host ""
    Write-Host "=== MPI ===" -ForegroundColor Yellow
    foreach ($procs in $mpiProcesses) {
        Write-Host "Processes: $procs"
        for ($trial = 0; $trial -lt $trials; $trial++) {
            Invoke-Benchmark -Implementation "MPI_$($procs)procs" -Config "$procs" -Processes $procs -Trial $trial -Executable "./mpi/denoise_mpi_edge" -Arguments @($imagePath)
        }
    }

    Write-Host ""
    Write-Host "=== HYBRID (MPI + OpenMP) ===" -ForegroundColor Yellow
    foreach ($config in $hybridConfigs) {
        $procs = $config[0]
        $threads = $config[1]
        Write-Host "Config: $procs processes x $threads threads/process"

        for ($trial = 0; $trial -lt $trials; $trial++) {
            Invoke-Benchmark -Implementation "Hybrid_$($procs)x$($threads)" -Config "$($procs)x$($threads)" -Processes $procs -Trial $trial -Executable "./hybrid/denoise_hybrid" -Arguments @($imagePath, "$threads")
        }
    }

    Move-Item -LiteralPath $stagingFile -Destination $outputFile -Force
    Write-Host ""
    Write-Host "Results saved to $outputFile" -ForegroundColor Green
} catch {
    if (Test-Path -LiteralPath $stagingFile) {
        Remove-Item -LiteralPath $stagingFile -Force
    }
    throw
}
