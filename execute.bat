@echo off
REM ==============================================================================
REM execute.bat — 1-Command Execution Script for New Satellite GeoTIFF (.tif) Images
REM
REM Usage:
REM   execute.bat "path\to\image.tif" "CASE a4"
REM ==============================================================================

if "%~1"=="" (
    echo [ERROR] Missing image path.
    echo Usage: execute.bat "path\to\image.tif" "CASE_NAME"
    exit /b 1
)

if "%~2"=="" (
    echo [ERROR] Missing case name.
    echo Usage: execute.bat "path\to\image.tif" "CASE_NAME"
    exit /b 1
)

set IMAGE_PATH=%~1
set CASE_NAME=%~2

echo ==============================================================================
echo  EXECUTING NEW GEOTIFF INGESTION PIPELINE
echo  Image : %IMAGE_PATH%
echo  Case  : %CASE_NAME%
echo ==============================================================================

python process_new_tif.py --image "%IMAGE_PATH%" --case "%CASE_NAME%" --force

if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Pipeline failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Processing complete for %CASE_NAME%!
echo Check my_dashboard_data\ for the frontend upload JSON.
echo.
