@echo off
REM Build the rehab_engine pybind11 module on Windows (MSVC, stub mode)
echo === Setting up Visual Studio Build Tools environment ===
call "c:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: vcvarsall.bat failed. Check VS Build Tools installation.
    exit /b 1
)

echo === Configuring CMake (stub mode) ===
cd /d "%~dp0"
rmdir /s /q build 2>nul
cmake -S . -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DSTROKE_ENGINE_STUB=ON ^
    -DPython_EXECUTABLE=c:/Python314/python.exe ^
    -Dpybind11_DIR=C:/Users/FP/AppData/Roaming/Python/Python314/site-packages/pybind11/share/cmake/pybind11
if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configure failed.
    exit /b 1
)

echo === Building ===
cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo === Build successful! ===
echo Output:
dir build\rehab_engine*.pyd 2>nul
dir build\Release\rehab_engine*.pyd 2>nul
echo.
echo To test:
echo   python -c "import sys; sys.path.insert(0, 'build'); import rehab_engine; print(rehab_engine.PipelineConfig())"