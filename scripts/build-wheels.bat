@echo off

echo Building wheels

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"

REM Install build dependencies
echo Installing build dependencies...
python -m pip install --upgrade pip build wheel

REM Build source distribution and wheel
echo Building source distribution and wheel...
python -m build

echo.
echo Build complete!
pause
