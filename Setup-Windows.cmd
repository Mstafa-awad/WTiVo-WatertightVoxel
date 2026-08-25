@echo off
REM SPDX-License-Identifier: GPL-3.0-or-later
setlocal EnableExtensions
cd /d "%~dp0"

set "NO_WINGET="
set "SKIP_VCPKG="
set "ALLOW_MSVC="

:parse
if "%~1"=="" goto run
if /I "%~1"=="--no-winget" set "NO_WINGET=-NoWinget"& shift& goto parse
if /I "%~1"=="--skip-vcpkg-install" set "SKIP_VCPKG=-SkipVcpkgInstall"& shift& goto parse
if /I "%~1"=="--allow-unsupported-msvc" set "ALLOW_MSVC=-AllowUnsupportedMSVC"& shift& goto parse
echo [ERROR] Unknown setup argument: %~1
exit /b 2

:run
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1" %NO_WINGET% %SKIP_VCPKG% %ALLOW_MSVC%
exit /b %ERRORLEVEL%
