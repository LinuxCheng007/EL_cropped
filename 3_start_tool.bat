@echo off
if not "%1"=="RUN" ( cmd /k "%~f0" RUN & exit )
title EL Perspective Crop Tool v3
set "ROOT=%~dp0"
set "PY=%ROOT%runtime\python.exe"
set "APP=%ROOT%app\app.py"
if not exist "%PY%" (
    echo [ERROR] Python not found. Run 1_download_python.bat first.
    pause & exit /b 1
)
if not exist "%APP%" (
    echo [ERROR] app\app.py not found.
    pause & exit /b 1
)
netstat -ano 2>nul | find "15789" | find "LISTEN" >nul 2>&1
if not errorlevel 1 (
    echo Already running. Opening browser...
    start "" http://127.0.0.1:15789
    exit /b 0
)
echo Adding firewall rule for port 15789...
netsh advfirewall firewall show rule name="EL Crop Tool 15789" >nul 2>&1
if errorlevel 1 (
    netsh advfirewall firewall add rule name="EL Crop Tool 15789" dir=in action=allow protocol=TCP localport=15789 >nul 2>&1
    echo [OK] Firewall rule added.
) else (
    echo [OK] Firewall rule already exists.
)
echo.
echo Starting EL Crop Tool...
echo.
echo Local   : http://127.0.0.1:15789
echo LAN     : http://192.168.3.119:15789
echo.
echo [Close this window to exit]
echo.
start /B "" "%PY%" "%APP%"
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:15789
REM Keep window alive (close it to stop the server)
:keep
ping -n 30 127.0.0.1 >nul 2>&1
goto keep
