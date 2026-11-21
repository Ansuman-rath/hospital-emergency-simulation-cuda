@echo off
rem Configure and build with MSVC + Ninja (+ CUDA if installed).
rem Usage: build.bat [Release^|Debug] ["-DOPTION=VALUE" ...]   (quote options containing '=')
setlocal EnableDelayedExpansion
set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=Release

set EXTRA=
:argloop
shift
if "%~1"=="" goto argsdone
set EXTRA=!EXTRA! %1
goto argloop
:argsdone

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`call "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
  echo Visual Studio C++ build tools not found.
  exit /b 1
)
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

set GEN=-G "NMake Makefiles"
where ninja >nul 2>nul && set GEN=-G Ninja

cmake -S "%~dp0." -B "%~dp0build" %GEN% -DCMAKE_BUILD_TYPE=%CONFIG% %EXTRA% || exit /b 1
cmake --build "%~dp0build" || exit /b 1
echo.
echo Build finished: %~dp0build\bin
endlocal
