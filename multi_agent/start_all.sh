#!/bin/bash
# start_all.sh — 启动所有 Agent 服务（独立进程）
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="/usr/bin/python3"

echo "============================================"
echo "  启动多 Agent 电商智能客服（独立部署）"
echo "============================================"

# 1. RAG 服务（知识库检索）
echo "[1/4] 启动 RAG 服务 (端口 8001)..."
$VENV "$ROOT/../RAG/app.py" &
RAG_PID=$!

# 2. 路由 Agent
echo "[2/4] 启动 Router Agent (端口 8002)..."
$VENV "$ROOT/router_agent/service.py" &
ROUTER_PID=$!

# 3. 售前 Agent
echo "[3/4] 启动 Pre-Service Agent (端口 8003)..."
$VENV "$ROOT/pre_service_agent/service.py" &
PRE_PID=$!

# 4. 售后 Agent
echo "[4/4] 启动 After-Service Agent (端口 8004)..."
$VENV "$ROOT/after_service_agent/service.py" &
AFTER_PID=$!

echo ""
echo "所有服务已启动："
echo "  RAG 服务:          http://127.0.0.1:8001 (PID $RAG_PID)"
echo "  Router Agent:      http://127.0.0.1:8002 (PID $ROUTER_PID)"
echo "  Pre-Service Agent: http://127.0.0.1:8003 (PID $PRE_PID)"
echo "  After-Service:     http://127.0.0.1:8004 (PID $AFTER_PID)"
echo ""
echo "交互客户端: python main.py"
echo "API 网关:    python main_api.py (端口 8000)"
echo ""
echo "按 Ctrl+C 停止所有服务"

cleanup() {
    echo ""
    echo "正在停止所有服务..."
    kill $RAG_PID $ROUTER_PID $PRE_PID $AFTER_PID 2>/dev/null
    wait
    echo "已停止。"
}
trap cleanup EXIT INT TERM

wait
