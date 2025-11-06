# CoACD Profiler - Runs a single mesh and outputs detailed timing breakdown
param(
    [string]$Mesh = "examples\Octocat-v2.obj",
    [string]$Preset = "balanced"
)

$ErrorActionPreference = "Stop"

$EXECUTABLE = ".\build_vs\Release\main.exe"
$OUTPUT_DIR = ".\profile_outputs"

# Presets
$presets = @{
    "fast"     = @{ threshold = 0.1;  resolution = 1000; max_hulls = 16 }
    "balanced" = @{ threshold = 0.05; resolution = 2000; max_hulls = 32 }
    "detailed" = @{ threshold = 0.02; resolution = 4000; max_hulls = 64 }
}

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "CoACD Performance Profiler" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Executable: $EXECUTABLE"
Write-Host "Mesh:       $Mesh"
Write-Host "Preset:     $Preset"
Write-Host ""

if (-not (Test-Path $EXECUTABLE)) {
    Write-Error "Executable not found: $EXECUTABLE. Run .\build_vs.ps1 first."
    exit 1
}

if (-not (Test-Path $Mesh)) {
    Write-Error "Mesh not found: $Mesh"
    exit 1
}

if (-not $presets.ContainsKey($Preset)) {
    Write-Error "Unknown preset: $Preset. Available: $($presets.Keys -join ', ')"
    exit 1
}

# Create output directory
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

$config = $presets[$Preset]
$meshName = [System.IO.Path]::GetFileNameWithoutExtension($Mesh)
$outputPath = Join-Path $OUTPUT_DIR "${meshName}_${Preset}_profiled.obj"

$params = @(
    "-i", $Mesh
    "-o", $outputPath
    "--threshold", $config.threshold.ToString()
    "--resolution", $config.resolution.ToString()
    "--max-convex-hull", $config.max_hulls.ToString()
)

Write-Host "Command:" -ForegroundColor Yellow
Write-Host "  $EXECUTABLE $($params -join ' ')" -ForegroundColor Gray
Write-Host ""
Write-Host "Running with profiling enabled..." -ForegroundColor Yellow
Write-Host ""

# Set profiling environment variable
$env:COACD_PROFILE = "1"

$startTime = Get-Date

try {
    # Run with profiling enabled
    & $EXECUTABLE $params
    $exitCode = $LASTEXITCODE
    
    $duration = ((Get-Date) - $startTime).TotalSeconds
    
    Write-Host ""
    Write-Host "===========================================" -ForegroundColor Cyan
    
    if ($exitCode -eq 0) {
        Write-Host "SUCCESS!" -ForegroundColor Green
        Write-Host "Total Duration: $([math]::Round($duration, 2))s"
        Write-Host "Output:   $outputPath"
        
        if (Test-Path $outputPath) {
            $fileSize = (Get-Item $outputPath).Length
            Write-Host "Size:     $([math]::Round($fileSize / 1KB, 2)) KB"
        }
        
        Write-Host ""
        Write-Host "The profiling summary above shows the time spent in each major function." -ForegroundColor Cyan
        Write-Host "Focus optimization efforts on the functions with the highest % Total." -ForegroundColor Cyan
    } else {
        Write-Host "FAILED with exit code: $exitCode" -ForegroundColor Red
    }
    
    Write-Host "===========================================" -ForegroundColor Cyan
    
    exit $exitCode
} catch {
    Write-Host ""
    Write-Host "EXCEPTION: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
