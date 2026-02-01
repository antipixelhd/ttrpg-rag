@echo off
setlocal enabledelayedexpansion

echo Select evaluation type:
echo   1) Retrieval Evaluation (search --evaluate)
echo   2) Response Evaluation (chat --evaluate)
echo.
set /p EVAL_TYPE="Enter your choice (1 or 2): "

if "%EVAL_TYPE%"=="1" (
    set COMMAND=search
    echo.
    echo Selected: Retrieval Evaluation
) else if "%EVAL_TYPE%"=="2" (
    set COMMAND=chat
    echo.
    echo Selected: Response Evaluation
) else (
    echo.
    echo ERROR: Invalid choice. Please enter 1 or 2.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.

set /p CONFIG_FOLDER="Enter the folder path containing .yaml config files: "

set CONFIG_FOLDER=%CONFIG_FOLDER:"=%

if not exist "%CONFIG_FOLDER%" (
    echo.
    echo ERROR: Folder does not exist: %CONFIG_FOLDER%
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo.

set COUNT=0
for %%f in ("%CONFIG_FOLDER%\*.yaml") do (
    set /a COUNT+=1
)

if %COUNT%==0 (
    echo ERROR: No .yaml files found in %CONFIG_FOLDER%
    pause
    exit /b 1
)

echo Found %COUNT% configuration file(s) in: %CONFIG_FOLDER%
echo.
echo Starting evaluation...
echo.
echo ================================================================================
echo.

set PROCESSED=0
for %%f in ("%CONFIG_FOLDER%\*.yaml") do (
    set /a PROCESSED+=1
    echo.
    echo [!PROCESSED!/%COUNT%] Processing: %%~nxf
    echo --------------------------------------------------------------------------------
    
    python main.py %COMMAND% --evaluate -c "%%f"
    
    if errorlevel 1 (
        echo.
        echo WARNING: Evaluation failed for %%~nxf
        echo.
    ) else (
        echo.
        echo SUCCESS: Completed %%~nxf
        echo.
    )
    
    echo ================================================================================
)

echo.
echo All evaluations complete!
echo Processed %PROCESSED% configuration(s)
echo.
pause
