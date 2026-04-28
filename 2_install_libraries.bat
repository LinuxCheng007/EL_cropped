@echo off
if not "%1"=="RUN" ( cmd /k "%~f0" RUN & exit )
title EL Tool - Step 2: Install Libraries
set "ROOT=%~dp0"
set "PY=%ROOT%runtime\python.exe"
if not exist "%PY%" (
    echo [ERROR] Python not found. Run 1_download_python.bat first.
    pause & exit
)
echo Running library installer...
echo.
"%PY%" "%ROOT%install_libs.py"
pause & exit
