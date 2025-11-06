# Test a single mesh to diagnose crashes
param(
    [string]$Mesh = "examples\Octocat-v2.obj",
    [string]$Preset = "fast"
)

$ErrorActionPreference = "Stop"

$EXECUTABLE = ".\build_vs\Release\main.exe"
$OUTPUT_DIR = ".\test_outputs"

# Presets
$presets = @{
    "fast"     = @{ threshold = 0.1;  resolution = 1000; max_hulls = 16 }
    "balanced" = @{ threshold = 0.05; resolution = 2000; max_hulls = 32 }
    "detailed" = @{ threshold = 0.02; resolution = 4000; max_hulls = 64 }
}

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "CoACD Single Test (Diagnostic Mode)" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Executable: $EXECUTABLE"
Write-Host "Mesh:       $Mesh"
Write-Host "Preset:     $Preset"
Write-Host ""

if (-not (Test-Path $EXECUTABLE)) {
    Write-Error "Executable not found: $EXECUTABLE"
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
$outputPath = Join-Path $OUTPUT_DIR "${meshName}_${Preset}_diagnostic.obj"

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
Write-Host "Running..." -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date

try {
    # Run with full output visible
    & $EXECUTABLE $params
    $exitCode = $LASTEXITCODE
    
    $duration = ((Get-Date) - $startTime).TotalSeconds
    
    Write-Host ""
    Write-Host "===========================================" -ForegroundColor Cyan
    
    if ($exitCode -eq 0) {
        Write-Host "SUCCESS!" -ForegroundColor Green
        Write-Host "Duration: $([math]::Round($duration, 2))s"
        Write-Host "Output:   $outputPath"
        
        if (Test-Path $outputPath) {
            $fileSize = (Get-Item $outputPath).Length
            Write-Host "Size:     $([math]::Round($fileSize / 1KB, 2)) KB"
        }
    } else {
        Write-Host "FAILED!" -ForegroundColor Red
        Write-Host "Duration: $([math]::Round($duration, 2))s"
        
        # Translate error code
        $errorDesc = switch ($exitCode) {
            -1073741819 { "ACCESS_VIOLATION (0xC0000005)" }
            -1073740791 { "STATUS_STACK_BUFFER_OVERRUN (0xC0000409)" }
            -1073741571 { "STATUS_STACK_OVERFLOW (0xC00000FD)" }
            -1073741510 { "STATUS_HEAP_CORRUPTION (0xC0000374)" }
            -1073741515 { "STATUS_DLL_NOT_FOUND (0xC0000135)" }
            default { "Exit code: $exitCode" }
        }
        
        Write-Host "Error:    $errorDesc" -ForegroundColor Red
        Write-Host ""
        Write-Host "This indicates a crash in the C++ code." -ForegroundColor Yellow
        Write-Host "Common causes:" -ForegroundColor Yellow
        Write-Host "  - Null pointer dereference" -ForegroundColor Gray
        Write-Host "  - Buffer overflow" -ForegroundColor Gray
        Write-Host "  - Stack corruption" -ForegroundColor Gray
        Write-Host "  - Use of freed memory" -ForegroundColor Gray
        Write-Host "  - Thread safety issues" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Suggestion: Run with a debugger or enable crash dumps" -ForegroundColor Yellow
    }
    
    Write-Host "===========================================" -ForegroundColor Cyan
    
    exit $exitCode
} catch {
    Write-Host ""
    Write-Host "EXCEPTION: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
