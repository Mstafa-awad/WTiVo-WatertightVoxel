@echo off
REM SPDX-License-Identifier: GPL-3.0-or-later
setlocal EnableExtensions
cd /d "%~dp0"

if exist "%~dp0.wtivo-env.cmd" call "%~dp0.wtivo-env.cmd"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv not found. Run Setup-Windows.cmd first.
  exit /b 1
)
if not exist "%~dp0build\wtivo_core*.pyd" (
  echo [ERROR] wtivo_core Python extension not found. Run Setup-Windows.cmd first.
  exit /b 1
)
if not exist "%~dp0build\wtivo_vdb*.pyd" (
  echo [ERROR] wtivo_vdb Python extension not found. Run Setup-Windows.cmd first.
  exit /b 1
)
if not exist "%~dp0build\wtivo_gpupr*.pyd" (
  echo [ERROR] wtivo_gpupr Python extension not found. Run Setup-Windows.cmd first.
  exit /b 1
)

if defined WTIVO_VCPKG_BIN if exist "%WTIVO_VCPKG_BIN%" set "PATH=%WTIVO_VCPKG_BIN%;%PATH%"
if defined WTIVO_CUDA_HOME if exist "%WTIVO_CUDA_HOME%\bin" set "PATH=%WTIVO_CUDA_HOME%\bin;%PATH%"

"%PY%" -I "%~dp0wtivo.py" %*
exit /b %ERRORLEVEL%
