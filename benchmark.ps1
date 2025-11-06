# CoACD Performance Benchmark
# Tests all example meshes with all presets and logs timing results
$ErrorActionPreference = "Stop"

function Stop-ZombieProcesses {
    Write-Host "Cleaning up any zombie/stuck processes..." -ForegroundColor Yellow
    $killed = @()

    # Kill any running main.exe instances
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name='main.exe'" -ErrorAction SilentlyContinue
        foreach ($p in ($procs | Where-Object { $_ -ne $null })) {
            try {
                Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
                $killed += "main.exe (PID=$($p.ProcessId))"
            } catch {}
        }
    } catch {}

    if ($killed.Count -gt 0) {
        Write-Host ("Killed zombie processes: " + ($killed -join ", ")) -ForegroundColor DarkYellow
        Start-Sleep -Milliseconds 500
    } else {
        Write-Host "No zombie processes found." -ForegroundColor Green
    }
}

$EXECUTABLE = Resolve-Path ".\build_vs\Release\main.exe" | Select-Object -ExpandProperty Path
$EXAMPLES_DIR = Resolve-Path ".\examples" | Select-Object -ExpandProperty Path
$OUTPUT_DIR = Join-Path (Get-Location) "benchmark_outputs"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_FILE = "benchmark_$TIMESTAMP.log"
$MAX_PARALLEL = [Environment]::ProcessorCount  # Run tests in parallel

# Set OpenMP threads per process (total cores / parallel jobs)
# This prevents oversubscription when running multiple instances
$env:OMP_NUM_THREADS = [math]::Max(1, [math]::Floor([Environment]::ProcessorCount / $MAX_PARALLEL))

# Clean up any stuck processes before starting
Stop-ZombieProcesses

# Create output directory with absolute path
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

# Presets
$presets = @{
    "fast"     = @{ threshold = 0.1;  resolution = 1000; max_hulls = 16 }
    "balanced" = @{ threshold = 0.05; resolution = 2000; max_hulls = 32 }
    "detailed" = @{ threshold = 0.02; resolution = 4000; max_hulls = 64 }
}

# Check executable
if (-not (Test-Path $EXECUTABLE)) {
    Write-Error "Executable not found: $EXECUTABLE. Run .\build_vs.ps1 first."
    exit 1
}

# Get all .obj files
$meshFiles = Get-ChildItem -Path $EXAMPLES_DIR -Filter "*.obj"
if ($meshFiles.Count -eq 0) {
    Write-Error "No .obj files found in $EXAMPLES_DIR"
    exit 1
}

# Start log
$logContent = @()
$logContent += "CoACD Performance Benchmark"
$logContent += "==========================="
$logContent += "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$logContent += "Meshes: $($meshFiles.Count)"
$logContent += "Presets: $($presets.Keys -join ', ')"
$logContent += "Total tests: $($meshFiles.Count * $presets.Count)"
$logContent += ""

Write-Host $logContent[-6..-1] -join "`n"
Write-Host "Parallel jobs: $MAX_PARALLEL (OpenMP threads per job: $env:OMP_NUM_THREADS)"
Write-Host ""

$totalTests = $meshFiles.Count * $presets.Count

# Build test queue
$tests = @()
foreach ($mesh in $meshFiles) {
    $meshName = [System.IO.Path]::GetFileNameWithoutExtension($mesh.Name)
    $meshPath = $mesh.FullName
    
    foreach ($presetName in $presets.Keys | Sort-Object) {
        $config = $presets[$presetName]
        $tests += @{
            MeshName = $meshName
            MeshPath = $meshPath
            Preset = $presetName
            Config = $config
        }
    }
}

# Run tests in parallel
$jobs = @()
$completed = 0
$results = @{}

Write-Host "Starting parallel execution..."
Write-Host ""

foreach ($test in $tests) {
    # Wait if we hit max parallel jobs
    while (($jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $MAX_PARALLEL) {
        Start-Sleep -Milliseconds 100
        
        # Check for completed jobs
        $finished = $jobs | Where-Object { $_.State -eq 'Completed' }
        foreach ($job in $finished) {
            $result = Receive-Job -Job $job
            $completed++
            $statusLine = "[$completed/$totalTests] $($result.MeshName) - $($result.Preset) - $($result.Duration)s [$($result.Status)]"
            if ($result.Status -eq "FAIL" -and $result.Error) {
                $statusLine += " - $($result.Error)"
            }
            Write-Host $statusLine
            
            $results["$($result.MeshName)_$($result.Preset)"] = $result
            Remove-Job -Job $job
        }
        $jobs = $jobs | Where-Object { $_.State -eq 'Running' }
    }
    
    # Start new job
    $job = Start-Job -ScriptBlock {
        param($exe, $meshPath, $meshName, $presetName, $config, $outDir)
        
        $ErrorActionPreference = "Continue"
        $outputPath = Join-Path $outDir "${meshName}_${presetName}.obj"
        $params = @(
            "-i", $meshPath
            "-o", $outputPath
            "--threshold", $config.threshold.ToString()
            "--resolution", $config.resolution.ToString()
            "--max-convex-hull", $config.max_hulls.ToString()
        )
        
        $startTime = Get-Date
        $success = $false
        $errorMsg = ""
        
        try {
            # Test if executable exists
            if (-not (Test-Path $exe)) {
                throw "Executable not found: $exe"
            }
            
            # Run the command and capture output
            $output = & $exe $params 2>&1
            $exitCode = $LASTEXITCODE
            $success = $exitCode -eq 0
            
            if (-not $success) {
                # Translate common Windows error codes
                $errorDesc = switch ($exitCode) {
                    -1073741819 { "ACCESS_VIOLATION (0xC0000005) - Segmentation fault/null pointer" }
                    -1073740791 { "STATUS_STACK_BUFFER_OVERRUN (0xC0000409) - Stack overflow" }
                    -1073741571 { "STATUS_STACK_OVERFLOW (0xC00000FD) - Stack overflow" }
                    -1073741510 { "STATUS_HEAP_CORRUPTION (0xC0000374) - Heap corruption" }
                    -1073741515 { "STATUS_DLL_NOT_FOUND (0xC0000135) - Missing DLL" }
                    3221225477 { "APPLICATION_ERROR (0xC0000005) - Application error" }
                    default { "Exit code: $exitCode (0x$([Convert]::ToString($exitCode, 16).ToUpper()))" }
                }
                $errorMsg = $errorDesc
            }
        } catch {
            $errorMsg = $_.Exception.Message
        }
        
        $duration = ((Get-Date) - $startTime).TotalSeconds
        
        return @{
            MeshName = $meshName
            Preset = $presetName
            Duration = [math]::Round($duration, 2)
            Status = if ($success) { "OK" } else { "FAIL" }
            Error = $errorMsg
        }
    } -ArgumentList $EXECUTABLE, $test.MeshPath, $test.MeshName, $test.Preset, $test.Config, $OUTPUT_DIR
    
    $jobs += $job
}

# Wait for remaining jobs
while ($jobs.Count -gt 0) {
    Start-Sleep -Milliseconds 100
    $finished = $jobs | Where-Object { $_.State -eq 'Completed' }
    foreach ($job in $finished) {
        $result = Receive-Job -Job $job
        $completed++
        $statusLine = "[$completed/$totalTests] $($result.MeshName) - $($result.Preset) - $($result.Duration)s [$($result.Status)]"
        if ($result.Status -eq "FAIL" -and $result.Error) {
            $statusLine += " - $($result.Error)"
        }
        Write-Host $statusLine
        
        $results["$($result.MeshName)_$($result.Preset)"] = $result
        Remove-Job -Job $job
    }
    $jobs = $jobs | Where-Object { $_.State -eq 'Running' }
}

# Build log from results
foreach ($test in $tests) {
    $key = "$($test.MeshName)_$($test.Preset)"
    $r = $results[$key]
    $logContent += "$($r.MeshName),$($r.Preset),$($r.Duration),$($r.Status)"
}

# Calculate summary
$logContent += ""
$logContent += "Summary by Preset:"
$logContent += "-----------------"
foreach ($presetName in $presets.Keys | Sort-Object) {
    $presetLines = $logContent | Where-Object { $_ -match "^[^,]+,$presetName,\d+\.?\d*,OK$" }
    if ($presetLines.Count -gt 0) {
        $times = $presetLines | ForEach-Object { [double]($_ -split ',')[2] }
        $avg = ($times | Measure-Object -Average).Average
        $logContent += "$presetName : avg $([math]::Round($avg, 2))s"
    }
}

$logContent += ""
$logContent += "Summary by Mesh:"
$logContent += "----------------"
foreach ($mesh in $meshFiles) {
    $meshName = [System.IO.Path]::GetFileNameWithoutExtension($mesh.Name)
    $meshLines = $logContent | Where-Object { $_ -match "^$meshName,[^,]+,\d+\.?\d*,OK$" }
    if ($meshLines.Count -gt 0) {
        $times = $meshLines | ForEach-Object { [double]($_ -split ',')[2] }
        $avg = ($times | Measure-Object -Average).Average
        $logContent += "$meshName : avg $([math]::Round($avg, 2))s"
    }
}

# Save log
$logContent | Out-File -FilePath $LOG_FILE -Encoding UTF8

Write-Host ""
Write-Host "==========================="
Write-Host "Benchmark Complete!"
Write-Host "==========================="
Write-Host "Log saved: $LOG_FILE"
Write-Host "Outputs: $OUTPUT_DIR\"
Write-Host ""
Write-Host ($logContent[-20..-1] -join "`n")
