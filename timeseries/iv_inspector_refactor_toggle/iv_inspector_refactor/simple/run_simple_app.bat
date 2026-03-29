@echo off
setlocal
set "APP_DIR=%~dp0"
set "PYEXE=C:\Users\admin\.conda\envs\st\python.exe"
if not exist "%PYEXE%" (
  echo st environment not found. Run setup_simple_env.bat first.
  exit /b 1
)
pushd "%APP_DIR%"
"%PYEXE%" -m streamlit run app.py --server.fileWatcherType none
popd
endlocal
