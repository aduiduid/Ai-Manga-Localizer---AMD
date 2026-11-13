@echo off
rem Manga Localizer Studio - launcher for a fresh clone.
rem Creates .venv on first run, installs dependencies, then starts the UI.
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [setup] No .venv found - bootstrapping. This downloads several GB and may take a while.
  echo [setup] Requires Python 3.11 or 3.12 on PATH ^(or the uv package manager^).
  echo [setup] Model weights are left to the UI's "prepare missing models" button.
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" -SkipModels
  if errorlevel 1 (
    echo.
    echo [ERROR] Bootstrap failed. See the messages above, or follow README.md manually.
    pause
    exit /b 1
  )
)

rem Keep every cache and every downloaded weight inside this folder, so the
rem checkout is self-contained and nothing lands on C:.
set "MLS_HOME=%~dp0data"
set "PADDLE_PDX_CACHE_HOME=%~dp0data\models\paddlex"
set "PADDLE_PDX_MODEL_SOURCE=modelscope"
set "MODELSCOPE_CACHE=%~dp0data\models\modelscope-cache"
set "HF_HOME=%~dp0data\models\huggingface"
set "HF_ENDPOINT=https://hf-mirror.com"

set "USERPROFILE=%~dp0_home"
set "HOME=%~dp0_home"
set "TEMP=%~dp0_home\tmp"
set "TMP=%~dp0_home\tmp"
if not exist "%~dp0_home\tmp" mkdir "%~dp0_home\tmp"
for %%D in (Desktop Documents Downloads Pictures Music Videos Favorites Links Searches Contacts) do (
  if not exist "%~dp0_home\%%~D" mkdir "%~dp0_home\%%~D"
)

if "%~1"=="" (
  "%PY%" -m manga_localizer.cli ui
) else (
  "%PY%" -m manga_localizer.cli %*
)
if errorlevel 1 pause
