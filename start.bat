@echo off
rem Manga Localizer Studio - local launcher (AMD/CPU, no GPU torch)
rem Lives on D:\ with a pure-ASCII path, so no ASCII junction is needed.
setlocal
cd /d "%~dp0"

set "PY=%~dp0app\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Virtualenv missing: "%PY%"
  echo Reinstall with: uv venv --python 3.12 "%~dp0app\.venv"
  pause
  exit /b 1
)

set "MLS_HOME=%~dp0data"
set "PADDLE_PDX_CACHE_HOME=%~dp0data\models\paddlex"
set "PADDLE_PDX_MODEL_SOURCE=modelscope"
set "MODELSCOPE_CACHE=%~dp0data\models\modelscope-cache"
set "HF_HOME=%~dp0data\models\huggingface"
set "HF_ENDPOINT=https://hf-mirror.com"

rem --- paddle/torch write to %%USERPROFILE%%; keep them off the C: drive ---
set "USERPROFILE=%~dp0_home"
set "HOME=%~dp0_home"
set "TEMP=%~dp0_home\tmp"
set "TMP=%~dp0_home\tmp"
if not exist "%~dp0_home\tmp" mkdir "%~dp0_home\tmp"
rem --- The shell folder picker reads its default location from %USERPROFILE%.
rem --- Without a Desktop folder there it pops "位置不可用 ...\_home\Desktop".
for %%D in (Desktop Documents Downloads Pictures Music Videos Favorites Links Searches Contacts) do (
  if not exist "%~dp0_home\%%~D" mkdir "%~dp0_home\%%~D"
)

rem --- Ollama: DeepSeek-V4-Pro-Qwen3.5-9B (i1-Q6_K + f16 mmproj), Vulkan on RX 9070 XT ---
set "OLLAMA_MODELS=%~dp0ollama-models"
set "OLLAMA_HOST=127.0.0.1:11434"
rem --- Serve the Hy-MT2 candidate pass from the GPU too.  Drop this line to
rem --- fall back to the CPU torch build (much slower, same weights).
set "MLS_HY_MT2_OLLAMA=hy-mt2-1.8b:q8"
if exist "%~dp0_dl\ollama\ollama.exe" (
  powershell -NoProfile -Command "try{Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/tags' -UseBasicParsing -TimeoutSec 2|Out-Null}catch{Start-Process -FilePath '%~dp0_dl\ollama\ollama.exe' -ArgumentList 'serve' -WindowStyle Hidden}"
)

if "%~1"=="" (
  "%PY%" -m manga_localizer.cli ui
) else (
  "%PY%" -m manga_localizer.cli %*
)
if errorlevel 1 pause
