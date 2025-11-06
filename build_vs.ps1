# Quick build script using existing Visual Studio solution
# Much faster than rebuild through setup.py
$ErrorActionPreference = "Stop"

$BUILD_DIR = ".\build_vs"

function Stop-CoACDLocks {
    Write-Host "Ensuring no running processes are locking build outputs..." -ForegroundColor Yellow
    $killed = @()

    # Kill running main.exe instances
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name='main.exe'" -ErrorAction SilentlyContinue
        foreach ($p in ($procs | Where-Object { $_ -ne $null })) {
            try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; $killed += "main.exe (PID=$($p.ProcessId))" } catch {}
        }
    } catch {}

    # Kill python processes likely holding lib_coacd.dll (heuristic: command line contains project path or 'coacd')
    try {
        $projPath = (Get-Location).Path
        $pys = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(python|pythonw)\.exe$' }
        foreach ($p in ($pys | Where-Object { $_.CommandLine -and (($_.CommandLine -match '(?i)coacd') -or ($_.CommandLine -like "*$projPath*")) })) {
            try { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; $killed += "$($p.Name) (PID=$($p.ProcessId))" } catch {}
        }
    } catch {}

    if ($killed.Count -gt 0) {
        Write-Host ("Killed: " + ($killed -join ", ")) -ForegroundColor DarkYellow
        Start-Sleep -Milliseconds 500
    }
}

# Check if solution exists
if (-not (Test-Path "$BUILD_DIR\CoACD.sln")) {
    Write-Host "Visual Studio solution not found. Generating first..."
    & .\generate_vs_solution.ps1
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Building CoACD with Visual Studio..."
Write-Host "Build directory: $BUILD_DIR"
Write-Host "Build type: Incremental (only changed files will recompile)"

# Stop any locking processes before building
Stop-CoACDLocks

# Build with CMake (uses Visual Studio behind the scenes)
$TARGET = $args[0]
if ($null -eq $TARGET -or $TARGET -eq "") {
    $TARGET = "ALL_BUILD"
}

Write-Host "Target: $TARGET"
Write-Host "`nStarting build..."
$buildStart = Get-Date

cmake --build $BUILD_DIR --config Release --target $TARGET -j

$buildEnd = Get-Date
$buildTime = ($buildEnd - $buildStart).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==================================="
    Write-Host "Build completed successfully!"
    Write-Host "Build time: $([math]::Round($buildTime, 2)) seconds"
    Write-Host "==================================="
    
    if ($TARGET -eq "_coacd" -or $TARGET -eq "ALL_BUILD") {
        Write-Host "`nPython module location:"
        if (Test-Path "$BUILD_DIR\Release\lib_coacd.dll") {
            $moduleSize = [math]::Round((Get-Item "$BUILD_DIR\Release\lib_coacd.dll").Length / 1MB, 2)
            Write-Host "  $BUILD_DIR\Release\lib_coacd.dll ($moduleSize MB)"
        }
    }
    
    if ($TARGET -eq "main" -or $TARGET -eq "ALL_BUILD") {
        Write-Host "`nExecutable location:"
        if (Test-Path "$BUILD_DIR\Release\main.exe") {
            $exeSize = [math]::Round((Get-Item "$BUILD_DIR\Release\main.exe").Length / 1MB, 2)
            Write-Host "  $BUILD_DIR\Release\main.exe ($exeSize MB)"
        }
    }
} else {
    Write-Error "Build failed with exit code $LASTEXITCODE"
    Write-Host "Attempting to free file locks and retry once..." -ForegroundColor Yellow
    Stop-CoACDLocks
    Start-Sleep -Milliseconds 500
    cmake --build $BUILD_DIR --config Release --target $TARGET -j
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Retry succeeded." -ForegroundColor Green
        exit 0
    }
    exit $LASTEXITCODE
}
