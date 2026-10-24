$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Create the Python 3.12 x64 virtual environment and install requirements first; see README.md.'
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonExe app.py --port 8797
} finally {
    Pop-Location
}
