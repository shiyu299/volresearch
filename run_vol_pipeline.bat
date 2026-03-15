@echo off
setlocal
cd /d "%~dp0"

set "PYEXE=%USERPROFILE%\.conda\envs\st\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
set "EXITCODE=0"

echo.
echo [1] Run modified v4 + factor materials
echo [2] Run factor materials only
set /p MODE=Choose mode [1/2]:

if "%MODE%"=="1" goto RUN_ALL
if "%MODE%"=="2" goto RUN_FACTOR_ONLY
echo Invalid mode.
set "EXITCODE=1"
goto END

:RUN_ALL
set /p RAW_FILE=Raw file under data\raw (example: PL26.csv):
set /p UNDERLYING=Underlying symbol (example: PL605):
set /p EXPIRY=Expiry date (example: 2026-04-13):
set /p SPREAD=Spread limit (example: 25):
set /p DERIVED_FILE=Derived output file under data\derived (example: PL60526.parquet):
set /p PREVIEW_FILE=Preview csv under data\derived (example: PL60526preview5000.csv):
set /p BASE_RULE=Factor base rule (example: 1S):

"%PYEXE%" timeseries\marketvolseries_modified_v4.py ^
  --csv-path "data/raw/%RAW_FILE%" ^
  --underlying "%UNDERLYING%" ^
  --expiry-date "%EXPIRY%" ^
  --spread-limit %SPREAD% ^
  --out-path "data/derived/%DERIVED_FILE%" ^
  --out-csv-preview-path "data/derived/%PREVIEW_FILE%"
if errorlevel 1 (
  set "EXITCODE=%ERRORLEVEL%"
  goto END
)

"%PYEXE%" timeseries\iv_inspector_refactor_toggle\iv_inspector_refactor\precompute_factor_materials.py ^
  --input "data/derived/%DERIVED_FILE%" ^
  --base-rule "%BASE_RULE%"
if errorlevel 1 (
  set "EXITCODE=%ERRORLEVEL%"
  goto END
)

goto DONE

:RUN_FACTOR_ONLY
set /p DERIVED_FILE=Derived input file under data\derived (example: PL60526.parquet):
set /p BASE_RULE=Factor base rule (example: 1S):

"%PYEXE%" timeseries\iv_inspector_refactor_toggle\iv_inspector_refactor\precompute_factor_materials.py ^
  --input "data/derived/%DERIVED_FILE%" ^
  --base-rule "%BASE_RULE%"
if errorlevel 1 (
  set "EXITCODE=%ERRORLEVEL%"
  goto END
)

:DONE
echo.
echo Finished.
:END
echo.
echo Exit code: %EXITCODE%
pause
endlocal
