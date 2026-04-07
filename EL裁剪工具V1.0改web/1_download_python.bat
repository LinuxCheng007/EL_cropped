@echo off
if not "%1"=="RUN" ( cmd /k "%~f0" RUN & exit )
title EL Tool - Step 1: Download Python
set "ROOT=%~dp0"
set "PYDIR=%ROOT%runtime"
set "PY=%PYDIR%\python.exe"
echo.
echo =============================================
echo  EL Tool - Step 1: Download Python 3.8
echo =============================================
echo.
if exist "%PY%" (
    echo [OK] Python already downloaded.
    echo Now run: 2_install_libraries.bat
    pause & exit
)
if not exist "%PYDIR%" mkdir "%PYDIR%"
set ARCH=win32
if exist "%SYSTEMROOT%\SysWOW64\cmd.exe" set ARCH=amd64
echo Arch: %ARCH%
if "%ARCH%"=="amd64" (
    set PY_URL=https://www.python.org/ftp/python/3.8.10/python-3.8.10-embed-amd64.zip
) else (
    set PY_URL=https://www.python.org/ftp/python/3.8.10/python-3.8.10-embed-win32.zip
)
set "PY_ZIP=%PYDIR%\py.zip"
echo Downloading Python 3.8.10 (~8MB)...
curl -k -L --retry 3 --progress-bar -o "%PY_ZIP%" "%PY_URL%"
if exist "%PY_ZIP%" goto :unzip
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$c=New-Object Net.WebClient;[System.Net.ServicePointManager]::ServerCertificateValidationCallback={$true};[Net.ServicePointManager]::SecurityProtocol=3072;$c.DownloadFile('%PY_URL%','%PY_ZIP%')"
if exist "%PY_ZIP%" goto :unzip
bitsadmin /transfer dl /download /priority normal "%PY_URL%" "%PY_ZIP%"
if exist "%PY_ZIP%" goto :unzip
echo [ERROR] Download failed. Check internet connection.
pause & exit
:unzip
echo Extracting...
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%PY_ZIP%' -DestinationPath '%PYDIR%' -Force" >nul 2>&1
if not exist "%PY%" tar -xf "%PY_ZIP%" -C "%PYDIR%" >nul 2>&1
del "%PY_ZIP%" >nul 2>&1
if not exist "%PY%" ( echo [ERROR] Extraction failed. & pause & exit )
echo.
echo [OK] Python ready. Now run: 2_install_libraries.bat
echo.
pause & exit
