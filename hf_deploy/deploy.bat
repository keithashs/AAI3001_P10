@echo off
REM ============================================
REM Hugging Face Spaces Deployment Script
REM Fashion Intelligence Suite - AAI3001 Group 10
REM ============================================

echo.
echo ========================================
echo  Fashion Intelligence Suite Deployment
echo ========================================
echo.

REM Check if huggingface-cli is installed
where huggingface-cli >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [!] huggingface-cli not found. Installing...
    pip install huggingface_hub
)

REM Login to Hugging Face (if not already logged in)
echo [1/4] Checking Hugging Face login...
huggingface-cli whoami >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Please log in to Hugging Face:
    huggingface-cli login
)

REM Set your Space name
set SPACE_NAME=fashion-intelligence-suite
set /p USERNAME="Enter your Hugging Face username: "

echo.
echo [2/4] Creating Space: %USERNAME%/%SPACE_NAME%
huggingface-cli repo create %SPACE_NAME% --type space --space_sdk gradio

echo.
echo [3/4] Cloning and uploading files...
cd /d "%~dp0"

REM Initialize git if needed
if not exist ".git" (
    git init
    git lfs install
)

REM Add remote
git remote remove origin 2>nul
git remote add origin https://huggingface.co/spaces/%USERNAME%/%SPACE_NAME%

REM Add all files
git add .
git commit -m "Initial deployment of Fashion Intelligence Suite"

echo.
echo [4/4] Pushing to Hugging Face Spaces...
git push -u origin main --force

echo.
echo ========================================
echo  Deployment Complete!
echo ========================================
echo.
echo Your Space is now live at:
echo https://huggingface.co/spaces/%USERNAME%/%SPACE_NAME%
echo.
echo Note: First build may take 5-10 minutes.
echo.
pause
