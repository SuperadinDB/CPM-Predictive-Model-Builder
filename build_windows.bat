@echo off
setlocal
cd /d "%~dp0"

if not exist ".build-venv\Scripts\python.exe" (
  py -3.12 -m venv .build-venv
  if errorlevel 1 goto :failed
)
set "PYTHON=.build-venv\Scripts\python.exe"

%PYTHON% -m pip install --upgrade pip
if errorlevel 1 goto :failed
%PYTHON% -m pip install -r requirements.txt -r requirements-models.txt pyinstaller
if errorlevel 1 goto :failed

%PYTHON% -m PyInstaller --noconfirm --clean CPM_Model_Builder.spec
if errorlevel 1 goto :failed

set "ISCC="
where ISCC >nul 2>nul
if not errorlevel 1 set "ISCC=ISCC"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 7\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 7\ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 7\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 7\ISCC.exe"
if not defined ISCC (
  echo PyInstaller completed, but Inno Setup 6 or 7 is required to create the Windows installer.
  echo Install Inno Setup and run this script again.
  goto :failed
)

"%ISCC%" CPM_Model_Builder.iss
if errorlevel 1 goto :failed
echo Installer created: installer\CPM_Model_Builder_Setup.exe
exit /b 0

:failed
echo Build failed. Review the command output above.
exit /b 1
