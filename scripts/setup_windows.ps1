# SPDX-License-Identifier: GPL-3.0-or-later
param(
    [switch]$NoWinget,
    [switch]$SkipVcpkgInstall,
    [switch]$AllowUnsupportedMSVC
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$VcpkgTag = '2026.07.29'
$KnownGoodVc = '14.38'

function Fail([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red; exit 1 }
function Info([string]$Message) { Write-Host "[WTiVo] $Message" -ForegroundColor Cyan }
function Success([string]$Message) { Write-Host "[SUCCESS] $Message" -ForegroundColor Green }

if ($env:OS -ne 'Windows_NT') { Fail 'WTiVo v1.0 setup currently supports Windows only.' }
if (-not [Environment]::Is64BitOperatingSystem) { Fail 'WTiVo requires 64-bit Windows.' }

Write-Host '======================================================='
Write-Host '  WTiVo: WatertightVoxel Optimizer - Windows Setup'
Write-Host '======================================================='
Write-Host 'Windows x64 + NVIDIA CUDA'
Write-Host 'Known-good build: Python 3.12 / Torch 2.8 cu128 / CUDA 12.8 / MSVC 14.38'
Write-Host ''

function Find-Python312 {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $p = (& $py.Source -3.12 -c "import sys;print(sys.executable)" 2>$null | Select-Object -First 1).Trim()
            if ($p -and (Test-Path $p)) { return $p }
        } catch {}
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        try {
            $ver = (& $python.Source -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
            if ($ver -eq '3.12') { return $python.Source }
        } catch {}
    }
    return $null
}

function Install-WithWinget([string]$Id, [string[]]$ExtraArgs=@()) {
    if ($NoWinget) { return $false }
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) { return $false }
    Info "winget install: $Id"
    $args = @('install','--id',$Id,'--exact','--accept-package-agreements','--accept-source-agreements') + $ExtraArgs
    & $winget.Source @args
    return ($LASTEXITCODE -eq 0)
}

# Python 3.12
$Python = Find-Python312
if (-not $Python) { [void](Install-WithWinget 'Python.Python.3.12'); $Python = Find-Python312 }
if (-not $Python) { Fail 'Python 3.12 x64 was not found. Install it from python.org and rerun setup.' }
Info "Python: $Python"

$Venv = Join-Path $Root '.venv'
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    Info 'Creating .venv...'
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail 'Could not create .venv.' }
}

Info 'Installing Python dependencies...'
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Fail 'pip bootstrap failed.' }
& $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
if ($LASTEXITCODE -ne 0) { Fail 'PyTorch 2.8.0 cu128 installation failed.' }
& $VenvPython -m pip install -r (Join-Path $Root 'requirements-runtime.txt') -r (Join-Path $Root 'requirements-build.txt')
if ($LASTEXITCODE -ne 0) { Fail 'Python dependencies failed.' }

& $VenvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print('[GPU]',torch.cuda.get_device_name(0),'CC',torch.cuda.get_device_capability(0),'Torch',torch.__version__,'CUDA',torch.version.cuda)"
if ($LASTEXITCODE -ne 0) { Fail 'PyTorch cannot see an NVIDIA CUDA GPU. Update/install the NVIDIA driver and retry.' }

# CUDA Toolkit 12.8
$CudaHome = $env:WTIVO_CUDA_HOME
if (-not $CudaHome) { $CudaHome = $env:CUDA_HOME }
if (-not $CudaHome) { $CudaHome = $env:CUDA_PATH }
$PreferredCuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
if ((-not $CudaHome) -and (Test-Path (Join-Path $PreferredCuda 'bin\nvcc.exe'))) { $CudaHome = $PreferredCuda }
if (-not $CudaHome -or -not (Test-Path (Join-Path $CudaHome 'bin\nvcc.exe'))) {
    Fail "CUDA Toolkit 12.8 with nvcc was not found. Install it from NVIDIA, then rerun Setup-Windows.cmd."
}
$nvccText = (& (Join-Path $CudaHome 'bin\nvcc.exe') --version | Out-String)
if ($nvccText -notmatch 'release 12\.8') {
    Fail "WTiVo v1.0 release setup requires CUDA Toolkit 12.8. Found: $CudaHome"
}
$env:WTIVO_CUDA_HOME=$CudaHome; $env:CUDA_HOME=$CudaHome; $env:CUDA_PATH=$CudaHome
$env:PATH=(Join-Path $CudaHome 'bin')+';'+$env:PATH
Info "CUDA Toolkit: $CudaHome"

# Visual Studio 2022 + known-good MSVC 14.38.
function Find-VsWhere {
    $p=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if(Test-Path $p){return $p}; return $null
}
function Find-VS2022 {
    $vw=Find-VsWhere; if(-not $vw){return $null}
    $p=(& $vw -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
    if($p -and (Test-Path $p)){return $p}; return $null
}
function Has-Vc1438([string]$VsPath) {
    if(-not $VsPath){return $false}
    return [bool](Get-ChildItem (Join-Path $VsPath 'VC\Tools\MSVC') -Directory -ErrorAction SilentlyContinue | Where-Object {$_.Name -like '14.38.*'} | Select-Object -First 1)
}

$VsPath=Find-VS2022
if((-not $VsPath) -or -not (Has-Vc1438 $VsPath)) {
    Info 'Visual Studio 2022 MSVC 14.38 is missing; attempting installation...'
    [void](Install-WithWinget 'Microsoft.VisualStudio.2022.BuildTools' @(
      '--override', '--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.14.38.17.8.x86.x64 --includeRecommended'
    ))
    $VsPath=Find-VS2022
}
if($VsPath -and -not (Has-Vc1438 $VsPath)) {
    $vsSetup=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\setup.exe'
    if((Test-Path $vsSetup) -and -not $NoWinget) {
        Info 'Adding the MSVC 14.38 component to the existing VS 2022 installation...'
        & $vsSetup modify --installPath "$VsPath" --passive --norestart --add Microsoft.VisualStudio.Component.VC.14.38.17.8.x86.x64 --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
        $VsPath=Find-VS2022
    }
}
if((-not $VsPath) -or -not (Has-Vc1438 $VsPath)) {
    Fail 'Visual Studio 2022 with MSVC v14.38 (17.8) is required. Add component Microsoft.VisualStudio.Component.VC.14.38.17.8.x86.x64 in Visual Studio Installer.'
}
$VcVarsAll=Join-Path $VsPath 'VC\Auxiliary\Build\vcvarsall.bat'
if(-not (Test-Path $VcVarsAll)){Fail 'vcvarsall.bat missing from VS 2022.'}
Info "Visual Studio 2022: $VsPath"
Info 'Loading MSVC 14.38 environment...'
$envLines = cmd /s /c "`"$VcVarsAll`" x64 -vcvars_ver=14.38 >nul && set"
foreach($line in $envLines){$i=$line.IndexOf('=');if($i -gt 0){[Environment]::SetEnvironmentVariable($line.Substring(0,$i),$line.Substring($i+1),'Process')}}
$env:DISTUTILS_USE_SDK='1'; $env:MSSdk='1'
if($AllowUnsupportedMSVC){$env:WTIVO_ALLOW_UNSUPPORTED_MSVC='1'}
$cl=Get-Command cl.exe -ErrorAction SilentlyContinue
if(-not $cl){Fail 'cl.exe unavailable after vcvarsall.'}
$clText=(& $cl.Source 2>&1 | Out-String)
if(($clText -notmatch '19\.38\.') -and -not $AllowUnsupportedMSVC){Fail "Expected cl 19.38, but active compiler differs. Active: $($cl.Source)"}
Info "Compiler: $($cl.Source)"

# private pinned vcpkg
$Deps=Join-Path $Root '.deps'; $VcpkgRoot=Join-Path $Deps 'vcpkg'; $VcpkgExe=Join-Path $VcpkgRoot 'vcpkg.exe'
New-Item -ItemType Directory -Force -Path $Deps | Out-Null
if(-not (Test-Path $VcpkgExe)){
    Info "Downloading pinned vcpkg $VcpkgTag..."
    $zip=Join-Path $Deps "vcpkg-$VcpkgTag.zip"; $tmp=Join-Path $Deps 'vcpkg-extract'
    if(Test-Path $tmp){Remove-Item $tmp -Recurse -Force}
    Invoke-WebRequest -UseBasicParsing "https://github.com/microsoft/vcpkg/archive/refs/tags/$VcpkgTag.zip" -OutFile $zip
    Expand-Archive $zip $tmp -Force
    $extracted=Get-ChildItem $tmp -Directory | Select-Object -First 1
    if(-not $extracted){Fail 'vcpkg extraction failed.'}
    if(Test-Path $VcpkgRoot){Remove-Item $VcpkgRoot -Recurse -Force}
    Move-Item $extracted.FullName $VcpkgRoot
    Remove-Item $tmp -Recurse -Force; Remove-Item $zip -Force
    & (Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
    if($LASTEXITCODE -ne 0){Fail 'vcpkg bootstrap failed.'}
}
$env:VCPKG_DISABLE_METRICS='1'
if(-not $SkipVcpkgInstall){
    Info 'Installing manifest native dependencies (CGAL/Eigen/OpenVDB/oneTBB)... first run can be long.'
    & $VcpkgExe install --triplet x64-windows --x-manifest-root=$Root --disable-metrics
    if($LASTEXITCODE -ne 0){Fail 'vcpkg dependency installation failed.'}
}
$VcpkgBin=Join-Path $VcpkgRoot 'installed\x64-windows\bin'
if(-not (Test-Path $VcpkgBin)){Fail 'vcpkg x64-windows runtime directory missing.'}
$env:WTIVO_VCPKG_BIN=$VcpkgBin; $env:PATH=$VcpkgBin+';'+$env:PATH

# build CPU/OpenVDB modules
$Cmake=Join-Path $Venv 'Scripts\cmake.exe'; $Ninja=Join-Path $Venv 'Scripts\ninja.exe'
$PybindDir=(& $VenvPython -m pybind11 --cmakedir | Select-Object -First 1).Trim()
$NativeBuild=Join-Path $Root '.build\native'; New-Item -ItemType Directory -Force -Path $NativeBuild | Out-Null
$Toolchain=Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
Info 'Configuring wtivo_core + wtivo_vdb...'
& $Cmake -S $Root -B $NativeBuild -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_MAKE_PROGRAM=$Ninja" "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" -DVCPKG_TARGET_TRIPLET=x64-windows "-DPython3_EXECUTABLE=$VenvPython" "-Dpybind11_DIR=$PybindDir" "-DWTIVO_OUTPUT_DIR=$(Join-Path $Root 'build')"
if($LASTEXITCODE -ne 0){Fail 'CMake configure failed.'}
& $Cmake --build $NativeBuild --config Release -j ([Math]::Min(6,[Environment]::ProcessorCount))
if($LASTEXITCODE -ne 0){Fail 'Native CPU/OpenVDB build failed.'}

# build CUDA module for detected GPU
Info 'Building wtivo_gpupr for the detected GPU architecture...'
& $VenvPython (Join-Path $Root 'scripts\build_gpupr.py')
if($LASTEXITCODE -ne 0){Fail 'WTiVo CUDA extension build failed. See docs/TROUBLESHOOTING.md.'}

# Persist machine-local paths for later Run-WTiVo.cmd sessions.
$LocalEnv=Join-Path $Root '.wtivo-env.cmd'
@(
  '@echo off',
  "set `"WTIVO_CUDA_HOME=$CudaHome`"",
  "set `"CUDA_HOME=$CudaHome`"",
  "set `"CUDA_PATH=$CudaHome`"",
  "set `"WTIVO_VCPKG_BIN=$VcpkgBin`""
) | Set-Content -Encoding ASCII $LocalEnv

Info 'Running source audit...'
& $VenvPython (Join-Path $Root 'scripts\source_audit.py')
if($LASTEXITCODE -ne 0){Fail 'Source audit failed.'}
Info 'Running native install verification...'
& $VenvPython (Join-Path $Root 'scripts\verify_install.py')
if($LASTEXITCODE -ne 0){Fail 'Native verification failed.'}
Info 'Collecting exact installed dependency license/NOTICE files...'
& $VenvPython (Join-Path $Root 'scripts\collect_dependency_licenses.py')
if($LASTEXITCODE -ne 0){Fail 'Dependency license collection failed.'}

Success 'WTiVo setup completed.'
Write-Host ''
Write-Host 'Run:'
Write-Host '  Run-WTiVo.cmd --input "model.glb" --output "model_watertight.glb"'
