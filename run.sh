#!/bin/bash

# Define script names
BOT_SCRIPT="run_nifty_only.py"
LOG_FILE="nifty_trading.log"

# Check if bot is running
PID=$(pgrep -f "python3 $BOT_SCRIPT")

if [ -n "$PID" ]; then
    echo "✅ Trading Bot is ALREADY running (PID $PID)"
else
    echo "🚀 Starting NIFTY Trading Bot in background..."
    nohup python3 $BOT_SCRIPT > $LOG_FILE 2>&1 &
    PID=$!
    echo "✅ Bot started (PID $PID)"
    echo "📄 Logs redirected to $LOG_FILE"
    echo "⏳ Waiting for initialization..."
    sleep 3
fi

echo "==================================================="
echo "📊 Launching Monitor (Ctrl+C to exit monitor only)"
echo "==================================================="

python3 monitor.py

echo ""
echo "👋 Monitor closed."
# Check if bot is still running
if ps -p $PID > /dev/null; then
    echo "ℹ️  Trading Bot is STILL running in background (PID $PID)."
    echo "👉 To stop it, run: pkill -f $BOT_SCRIPT"
else
    echo "⚠️  Trading Bot has stopped."
fi
