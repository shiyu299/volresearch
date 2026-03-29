@echo off
setlocal
cd /d "%~dp0"

set "PYEXE=%~dp0st_env\python.exe"
if not exist "%PYEXE%" set "PYEXE=%USERPROFILE%\.conda\envs\st\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo.
set /p CSV_INPUT=Raw csv filename or path (example: TA605320.csv):
set /p EXPIRY=Expiry date (example: 2026-04-13):
set /p SPREAD=Spread limit (example: 15):

set "CSV_PATH=%CSV_INPUT%"

if exist "%CSV_PATH%" goto RUN_PROCESS
if exist "data\raw\%CSV_INPUT%" set "CSV_PATH=data\raw\%CSV_INPUT%"
if exist "%CSV_PATH%" goto RUN_PROCESS

for /r "data\raw" %%F in (%CSV_INPUT%) do (
  set "CSV_PATH=%%F"
  goto RUN_PROCESS
)

echo Raw csv not found: %CSV_INPUT%
echo Put the file under data\raw\ or input a valid relative path.
goto END

:RUN_PROCESS
echo Using raw csv: %CSV_PATH%
"%PYEXE%" "%~dp0process_raw_to_derived.py" --csv-path "%CSV_PATH%" --expiry-date "%EXPIRY%" --spread-limit %SPREAD%
if errorlevel 1 goto END

"%PYEXE%" -m streamlit run "%~dp0app.py" --server.headless true --server.fileWatcherType none

:END
endlocal
