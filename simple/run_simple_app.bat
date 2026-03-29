@echo off
setlocal
cd /d "%~dp0"

set "PYEXE=%~dp0st_env\python.exe"
if not exist "%PYEXE%" set "PYEXE=%USERPROFILE%\.conda\envs\st\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo Using Python: %PYEXE%
"%PYEXE%" -m streamlit run "%~dp0app.py" --server.headless true --server.fileWatcherType none

endlocal
