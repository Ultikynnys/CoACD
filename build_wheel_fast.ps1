param(
    [string]$Python = "python",
    [switch]$Clean,
    [switch]$CopyToAddon,
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$WheelDir = Join-Path $PSScriptRoot "dist"
$AddonDir = Join-Path (Join-Path $PSScriptRoot "..") "wheels"

Write-Host "== CoACD Windows Wheel Build ==" -ForegroundColor Cyan

if ($Clean) {
    Write-Host "Cleaning..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force (Join-Path $PSScriptRoot "build") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $WheelDir -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

Push-Location $PSScriptRoot

Write-Host "Building wheel..." -ForegroundColor Green
& $Python -m pip wheel . --wheel-dir $WheelDir --no-deps -v

$wheel = Get-ChildItem -Path $WheelDir -Filter "coacd_u-*-win_amd64.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $wheel) {
    Write-Host "ERROR: No wheel found" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host "Built: $($wheel.Name)" -ForegroundColor Green

if ($CopyToAddon) {
    New-Item -ItemType Directory -Force -Path $AddonDir | Out-Null
    $dest = Join-Path $AddonDir $wheel.Name
    Copy-Item -Force $wheel.FullName $dest
    Write-Host "Copied to: $dest" -ForegroundColor Green
}

if ($Install) {
    Write-Host "Installing..." -ForegroundColor Yellow
    & $Python -m pip uninstall -y coacd_u 2>&1 | Out-Null
    & $Python -m pip install --force-reinstall $wheel.FullName
}

Pop-Location
Write-Host "Done!" -ForegroundColor Cyan
