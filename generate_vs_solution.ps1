# Generate Visual Studio solution for faster iterative development
# Stop on errors
$ErrorActionPreference = "Stop"

Write-Host "Generating Visual Studio 2022 solution for CoACD..."

# Verify CMake is available
$cmakeCheck = Get-Command cmake -ErrorAction SilentlyContinue
if ($null -eq $cmakeCheck) {
    Write-Error "CMake not found. Please install CMake and add it to PATH"
    exit 1
}

$cmakeVersion = (cmake --version | Select-Object -First 1).ToString()
Write-Host "Using: $cmakeVersion"

# Create build directory for VS solution
$BUILD_DIR = ".\build_vs"
if (-not (Test-Path $BUILD_DIR)) {
    Write-Host "Creating build directory: $BUILD_DIR"
    New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
}

# Configure with CMake for Visual Studio
Write-Host "`nConfiguring CMake project..."
Push-Location $BUILD_DIR

try {
    $cmakeArgs = @(
        ".."
        "-G", "Visual Studio 17 2022"
        "-A", "x64"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
        "-DOPENVDB_CORE_SHARED=OFF"
        "-DTBB_TEST=OFF"
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
        "-DCMAKE_CXX_FLAGS=/MT /EHsc"
    )
    
    Write-Host "Running CMake to GENERATE solution (not building)..."
    & cmake $cmakeArgs
    
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    
    Write-Host "`n==================================="
    Write-Host "Visual Studio solution generated successfully!"
    Write-Host "==================================="
    Write-Host "`nVerifying solution file..."
    
    $slnPath = Join-Path (Get-Location) "CoACD.sln"
    if (Test-Path $slnPath) {
        $slnInfo = Get-Item $slnPath
        Write-Host "[OK] Solution file created: CoACD.sln"
        Write-Host "     Size: $([math]::Round($slnInfo.Length/1KB, 2)) KB"
        Write-Host "     Full path: $slnPath"
    } else {
        Write-Warning "Solution file not found at expected location!"
    }
    
    # List generated files
    Write-Host "`nGenerated files:"
    Get-ChildItem -Filter "*.sln" | ForEach-Object { Write-Host "  - $($_.Name)" }
    Get-ChildItem -Filter "*.vcxproj" | Select-Object -First 5 | ForEach-Object { Write-Host "  - $($_.Name)" }
    $vcxprojCount = (Get-ChildItem -Filter "*.vcxproj").Count
    if ($vcxprojCount -gt 5) {
        Write-Host "  ... and $($vcxprojCount - 5) more .vcxproj files"
    }
    
    Write-Host "`nLocation: $BUILD_DIR\"
    Write-Host "`nTo build:"
    Write-Host "1. Open build_vs\CoACD.sln in Visual Studio 2022"
    Write-Host "2. Or run: cmake --build $BUILD_DIR --config Release -j"
    Write-Host "`nTargets:"
    Write-Host "- coacd: Static library"
    Write-Host "- _coacd: Python module"
    Write-Host "- main: Standalone executable"
    
    # Ask if user wants to open the solution
    $openSolution = Read-Host "`nOpen solution in Visual Studio now? (y/n)"
    if ($openSolution -eq 'y') {
        $slnPath = Resolve-Path (Join-Path $BUILD_DIR "CoACD.sln")
        if (Test-Path $slnPath) {
            Write-Host "Opening: $slnPath"
            try {
                Start-Process -FilePath $slnPath -ErrorAction Stop
                Write-Host "Visual Studio should be launching..."
            } catch {
                Write-Warning "Could not open solution automatically: $_"
                Write-Host "Please open manually: $slnPath"
            }
        } else {
            Write-Warning "Solution file not found at: $slnPath"
        }
    }
    
} catch {
    Write-Error "Failed to generate solution: $_"
    exit 1
} finally {
    Pop-Location
}
