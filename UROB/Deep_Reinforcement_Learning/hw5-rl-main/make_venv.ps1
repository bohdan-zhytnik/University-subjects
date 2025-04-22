# Default Python version
$pythonVersion = 11
$venvName = "rl-homework-venv"

# Check if argument is provided
if ($args.Length -eq 1) {
    if ($args[0] -match "^(10|11|12)$") {
        $pythonVersion = [int]$args[0]
    } else {
        Write-Error "Error: Python version must be 10, 11, or 12"
        exit 1
    }
}

# Check if `py` command is available
if (-not (Get-Command "py" -ErrorAction SilentlyContinue)) {
    Write-Error "Error: Python launcher 'py' not found. Make sure Python is installed and added to PATH."
    exit 1
}

# Create virtual environment
py -3.$pythonVersion -m venv $venvName

if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to create virtual environment"
    exit 1
}

# Activate virtual environment
$activateScript = ".\$venvName\Scripts\activate.ps1"
& $activateScript

# Run install_deps.ps1
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Failed to install dependencies"
    exit 1
}

Write-Host "Virtual environment $venvName created and activated with Python 3.$pythonVersion"