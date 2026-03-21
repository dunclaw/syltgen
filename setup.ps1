<#
.SYNOPSIS
    Set up the syltgen Python virtual environment.

.DESCRIPTION
    Creates a .venv using Python 3.10-3.12, installs PyTorch (CPU or CUDA),
    then installs all remaining dependencies from requirements.txt.

.PARAMETER Cuda
    PyTorch CUDA version to target. Use "cpu" for CPU-only.
    Common values: "cu128" (CUDA 12.8), "cu121" (CUDA 12.1), "cu118" (CUDA 11.8), "cpu"
    Default: "cu128"

.EXAMPLE
    # GPU setup (CUDA 12.8 – RTX 40/50 series)
    .\setup.ps1

    # CPU-only setup
    .\setup.ps1 -Cuda cpu

    # CUDA 12.1
    .\setup.ps1 -Cuda cu121
#>
param(
    [string]$Cuda = "cu128"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------------
# 1. Locate Python 3.10-3.12
# -----------------------------------------------------------------------
$pythonExe = $null

# Try py launcher first (preferred on Windows)
foreach ($ver in @("3.12", "3.11", "3.10")) {
    try {
        $out = & py -$ver --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = "py -$ver"
            Write-Host "Using Python $ver via py launcher." -ForegroundColor Cyan
            break
        }
    } catch { }
}

if (-not $pythonExe) {
    Write-Error @"
Python 3.10, 3.11, or 3.12 is required but was not found.
WhisperX does NOT support Python 3.14+.

Download Python 3.12 from: https://www.python.org/downloads/release/python-3127/
Make sure to check 'Add to PATH' and install for all users.

After installing, re-run this script.
"@
    exit 1
}

# -----------------------------------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------------------------------
if (Test-Path ".venv") {
    Write-Host ".venv already exists – skipping creation." -ForegroundColor Yellow
} else {
    Write-Host "Creating .venv…" -ForegroundColor Green
    Invoke-Expression "$pythonExe -m venv .venv"
}

$pip = ".\.venv\Scripts\python.exe -m pip"

# -----------------------------------------------------------------------
# 3. Upgrade pip
# -----------------------------------------------------------------------
Write-Host "Upgrading pip…" -ForegroundColor Green
Invoke-Expression "$pip install --upgrade pip setuptools wheel" | Out-Null

# -----------------------------------------------------------------------
# 4. Install PyTorch
# -----------------------------------------------------------------------
$torchUrl = if ($Cuda -eq "cpu") {
    "https://download.pytorch.org/whl/cpu"
} else {
    "https://download.pytorch.org/whl/$Cuda"
}

Write-Host "Installing PyTorch ($Cuda)… (this may take a few minutes)" -ForegroundColor Green
Invoke-Expression "$pip install torch torchaudio --index-url $torchUrl"

# -----------------------------------------------------------------------
# 5. Install FFmpeg (required by audio-separator and whisperx)
# -----------------------------------------------------------------------
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "FFmpeg already on PATH – skipping." -ForegroundColor Yellow
} else {
    Write-Host "Installing FFmpeg via winget…" -ForegroundColor Green
    $wingetAvailable = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetAvailable) {
        winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements
        # Refresh PATH in this session so subsequent commands can find ffmpeg
        $env:PATH = [System.Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' +
                    [System.Environment]::GetEnvironmentVariable('PATH', 'User')
    } else {
        Write-Warning @"
winget not found.  Please install FFmpeg manually:
  - Download from https://ffmpeg.org/download.html  OR
  - Run:  choco install ffmpeg
Then re-open your terminal and re-run this script.
"@
    }
}

# -----------------------------------------------------------------------
# 6. Install remaining dependencies
# -----------------------------------------------------------------------
Write-Host "Installing project dependencies from requirements.txt…" -ForegroundColor Green
Invoke-Expression "$pip install -r requirements.txt"

# -----------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Activate the environment with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Then run: python -m syltgen.main ./input ./output" -ForegroundColor Cyan
