@echo off
REM ============================================================================
REM run.bat — single-command install + start for Loadshare RCA Agent
REM
REM What it does:
REM   1. Verifies prerequisites (Python, Node, .env with NVIDIA key)
REM   2. Installs Python deps if missing (pip install -r requirements.txt)
REM   3. Installs Node deps for the MCP server if missing (npm install)
REM   4. Builds the SQLite DB from CSV if missing
REM   5. Opens 3 terminals: MCP server, backend, frontend
REM
REM Re-run is safe; only re-installs what's missing.
REM ============================================================================

setlocal EnableDelayedExpansion
title Loadshare RCA Agent
cls

echo ===================================================
echo Loadshare RCA Agent
echo ===================================================
echo.

REM --- Prerequisite checks -----------------------------------------------------

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.10+ from https://python.org and try again.
    pause
    exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found in PATH.
    echo         Install Node 18+ from https://nodejs.org and try again.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [ERROR] .env file missing.
    echo.
    echo         Copy .env.example to .env and set your NVIDIA_API_KEY:
    echo           copy .env.example .env
    echo           notepad .env
    echo.
    pause
    exit /b 1
)

findstr /B "NVIDIA_API_KEY=nvapi-" .env >nul 2>&1
if errorlevel 1 (
    echo [WARN] NVIDIA_API_KEY in .env does not look like a real key.
    echo        Edit .env and set NVIDIA_API_KEY=nvapi-...
    echo        Continuing anyway in case you set it via environment.
    echo.
)

echo Prerequisites OK.
echo.

REM --- Python deps -------------------------------------------------------------

echo [1/3] Checking Python dependencies...
python -c "import fastapi, langgraph, mcp, openai, pydantic, streamlit" >nul 2>&1
if errorlevel 1 (
    echo        Installing Python packages...
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed.
        pause
        exit /b 1
    )
)
echo        Python OK.
echo.

REM --- Node deps for MCP -------------------------------------------------------

echo [2/3] Checking MCP server dependencies...
if not exist "mcp-servers\node_modules" (
    echo        Installing Node packages for MCP server...
    pushd mcp-servers
    call npm install --silent
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        popd
        pause
        exit /b 1
    )
    popd
)
echo        MCP server OK.
echo.

REM --- Database build ----------------------------------------------------------

echo [3/3] Checking database...
if not exist "data\loadshare.db" (
    echo        Building SQLite database from CSV...
    python scripts\load_csv_to_sqlite.py
    if errorlevel 1 (
        echo [ERROR] Database build failed.
        pause
        exit /b 1
    )
)
echo        Database OK.
echo.

REM --- Start services ---------------------------------------------------------

echo ===================================================
echo Starting services in 3 separate windows...
echo ===================================================
echo.
echo   [1] SQLite MCP Server  :3002
echo   [2] FastAPI Backend    :8000
echo   [3] Streamlit Frontend :8501
echo.
echo Login: demo / demo at http://localhost:8501
echo Close all 3 windows to stop.
echo.

start "MCP Server" cmd /k "cd mcp-servers && npm start"
timeout /t 5 /nobreak >nul

start "Backend"    cmd /k "python -m uvicorn app.main:app --port 8000 --reload"
timeout /t 8 /nobreak >nul

start "Frontend"   cmd /k "streamlit run frontend\app.py"

echo.
echo All services launched. Browser should open shortly.
echo.
pause
