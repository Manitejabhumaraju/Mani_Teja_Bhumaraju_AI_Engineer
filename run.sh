#!/usr/bin/env bash
# =============================================================================
# run.sh — single-command install + start for Loadshare RCA Agent
#
# What it does:
#   1. Verifies prerequisites (Python, Node, .env with NVIDIA key)
#   2. Installs Python deps if missing
#   3. Installs Node deps for the MCP server if missing
#   4. Builds the SQLite DB from CSV if missing
#   5. Starts MCP server, backend, frontend (in background; logs to ./logs/)
#
# Re-run is safe; only re-installs what's missing.
# Stops cleanly on Ctrl+C.
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# --- Prerequisite checks ----------------------------------------------------

command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 not in PATH. Install Python 3.10+."; exit 1; }
command -v node    >/dev/null 2>&1 || { echo "[ERROR] node not in PATH. Install Node 18+."; exit 1; }

if [[ ! -f .env ]]; then
    echo "[ERROR] .env missing. Run: cp .env.example .env && \$EDITOR .env"
    exit 1
fi

if ! grep -q "^NVIDIA_API_KEY=nvapi-" .env; then
    echo "[WARN] NVIDIA_API_KEY in .env doesn't look like a real key — continuing anyway."
fi

# --- Install steps (idempotent) --------------------------------------------

echo "[1/3] Python deps..."
if ! python3 -c "import fastapi, langgraph, mcp, openai, pydantic, streamlit" 2>/dev/null; then
    pip install -q -r requirements.txt
fi

echo "[2/3] MCP Node deps..."
if [[ ! -d mcp-servers/node_modules ]]; then
    (cd mcp-servers && npm install --silent)
fi

echo "[3/3] Database..."
if [[ ! -f data/loadshare.db ]]; then
    python3 scripts/load_csv_to_sqlite.py
fi

# --- Start services in background -------------------------------------------

mkdir -p logs

cleanup() {
    echo ""
    echo "Stopping services..."
    [[ -n "${MCP_PID:-}" ]] && kill "$MCP_PID" 2>/dev/null || true
    [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
    [[ -n "${UI_PID:-}"  ]] && kill "$UI_PID"  2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

echo ""
echo "===================================================="
echo "Starting:"
echo "  MCP server  :3002   (logs/mcp.log)"
echo "  Backend     :8000   (logs/backend.log)"
echo "  Frontend    :8501   (logs/frontend.log)"
echo ""
echo "Login: demo / demo at http://localhost:8501"
echo "Ctrl+C to stop everything."
echo "===================================================="

(cd mcp-servers && npm start) > logs/mcp.log 2>&1 &
MCP_PID=$!
sleep 4

python3 -m uvicorn app.main:app --port 8000 --reload > logs/backend.log 2>&1 &
API_PID=$!
sleep 6

streamlit run frontend/app.py > logs/frontend.log 2>&1 &
UI_PID=$!

echo ""
echo "All started. Tailing backend log (Ctrl+C to stop):"
echo ""
tail -f logs/backend.log
