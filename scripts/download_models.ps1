# Download the model weights this project runs on.
#
# The heavy lifting lives in ``manga_localizer.model_manager``: fixed URLs,
# SHA-256 checks and atomic writes.  This wrapper only finds a Python that has
# the package installed, so the weights land in the same place the UI looks.
#
#   .\scripts\download_models.ps1              # only what the current settings need
#   .\scripts\download_models.ps1 -All         # every weight in MODEL_REGISTRY
#   .\scripts\download_models.ps1 -Model lama  # one of: paddleocr manga-ocr lama hy-mt2
[CmdletBinding()]
param(
    [switch]$All,
    [ValidateSet("paddleocr", "manga-ocr", "lama", "hy-mt2")]
    [string]$Model = "lama"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "[setup] No .venv yet - running bootstrap first (dependency install only)." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "bootstrap.ps1") -SkipModels
}

# Keep the weights inside the checkout, exactly where start.bat points MLS_HOME.
$env:MLS_HOME = Join-Path $ProjectRoot "data"
$env:PADDLE_PDX_CACHE_HOME = Join-Path $ProjectRoot "data\models\paddlex"
$env:PADDLE_PDX_MODEL_SOURCE = "modelscope"
$env:MODELSCOPE_CACHE = Join-Path $ProjectRoot "data\models\modelscope-cache"
$env:HF_HOME = Join-Path $ProjectRoot "data\models\huggingface"

$Target = if ($All) { "all" } else { $Model }
Write-Host "Downloading model: $Target" -ForegroundColor Cyan
& $PythonPath -m manga_localizer.cli models download $Target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonPath -m manga_localizer.cli models status
Write-Host ""
Write-Host "GGUF models are not fetched here: drop them into data\llama-models\<tag>.gguf (see README 'Model weights')." -ForegroundColor Gray
