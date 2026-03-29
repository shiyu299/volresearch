@echo off
setlocal
set "APP_DIR=%~dp0"
set "CONDAEXE=C:\ProgramData\Anaconda3\Scripts\conda.exe"
if not exist "%CONDAEXE%" (
  echo conda.exe not found at %CONDAEXE%
  exit /b 1
)
pushd "%APP_DIR%"
"%CONDAEXE%" env update -n st -f st_env.yml --prune
popd
endlocal
