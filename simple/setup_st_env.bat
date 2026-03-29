@echo off
setlocal
cd /d "%~dp0"

set "LOCAL_PY=%CD%\st_env\python.exe"
if exist "%LOCAL_PY%" (
  echo Existing local env found: %LOCAL_PY%
  goto END
)

set "CONDAEXE=%ProgramData%\Anaconda3\Scripts\conda.exe"
if not exist "%CONDAEXE%" set "CONDAEXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not exist "%CONDAEXE%" (
  echo conda.exe not found. Please install dependencies manually.
  goto END
)

echo Creating local env at %CD%\st_env
"%CONDAEXE%" env create --prefix "%CD%\st_env" --file "%CD%\st_env.yml"

:END
endlocal
