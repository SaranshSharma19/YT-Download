#!/usr/bin/env python3
"""
Real-Time Trading Monitor
Monitors the trading system and provides live performance metrics
"""

import sqlite3
import time
from datetime import datetime, timedelta
import os

def get_db_stats(db_path="trading_performance.db"):
    """Get current database statistics"""
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {}
    
    # Total trades
    cursor.execute("SELECT COUNT(*) FROM trades")
    stats['total_trades'] = cursor.fetchone()[0]
    
    # Open trades
    cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NULL")
    stats['open_trades'] = cursor.fetchone()[0]
    
    # Closed trades
    cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL")
    stats['closed_trades'] = cursor.fetchone()[0]
    
    # Win rate
    if stats['closed_trades'] > 0:
        cursor.execute("""
            SELECT 
                AVG(CASE WHEN outcome='win' THEN 1.0 ELSE 0.0 END) as win_rate,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl
            FROM trades 
            WHERE outcome IS NOT NULL
        """)
        row = cursor.fetchone()
        stats['win_rate'] = row[0] if row[0] else 0
        stats['total_pnl'] = row[1] if row[1] else 0
        stats['avg_pnl'] = row[2] if row[2] else 0
    else:
        stats['win_rate'] = 0
        stats['total_pnl'] = 0
        stats['avg_pnl'] = 0
    
    # Recent trades (last 10)
    cursor.execute("""
        SELECT timestamp, index_name, direction, entry_price, confidence, regime
        FROM trades 
        ORDER BY timestamp DESC 
        LIMIT 10
    """)
    stats['recent_trades'] = cursor.fetchall()
    
    conn.close()
    return stats

def check_process_status():
    """Check if trading.py is running"""
    import subprocess
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True
        )
        return 'python3 trading.py' in result.stdout or 'python trading.py' in result.stdout
    except:
        return False

def monitor_logs(log_file=None, lines=20):
    """Get last N lines from log file"""
    # Search priority: specific file -> nifty log -> standard log
    candidates = ["nifty_trading.log", "trading_bot.log", "logs/trading_bot.log"]
    
    if log_file:
        candidates.insert(0, log_file)
    
    target_file = None
    for f in candidates:
        if os.path.exists(f):
            target_file = f
            break
            
    if not target_file:
        return [f"⚠️  Log file not found. Checked: {', '.join(candidates)}"]
    
    try:
        with open(target_file, 'r') as f:
            all_lines = f.readlines()
            return [f"--- Tail of {target_file} ---\n"] + all_lines[-lines:]
    except:
        return ["Error reading log file"]

def main():
    print("="*80)
    print("TRADING SYSTEM MONITOR")
    print("="*80)
    print()
    
    # Check if process is running
    is_running = check_process_status()
    print(f"🔄 Trading System Status: {'RUNNING ✅' if is_running else 'STOPPED ❌'}")
    print()
    
    # Check database stats
    stats = get_db_stats()
    
    if stats:
        print("📊 PERFORMANCE METRICS")
        print("-" * 80)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Open Trades: {stats['open_trades']}")
        print(f"Closed Trades: {stats['closed_trades']}")
        
        if stats['closed_trades'] > 0:
            print(f"Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"Total P&L: ₹{stats['total_pnl']:+,.2f}")
            print(f"Avg P&L/Trade: ₹{stats['avg_pnl']:+,.2f}")
        
        print()
        
        if stats['recent_trades']:
            print("📈 RECENT TRADES")
            print("-" * 80)
            print(f"{'Time':<20} {'Index':<12} {'Dir':<4} {'Price':<10} {'Conf':<6} {'Regime':<15}")
            print("-" * 80)
            for trade in stats['recent_trades']:
                timestamp, index_name, direction, entry_price, confidence, regime = trade
                print(f"{timestamp:<20} {index_name:<12} {direction:<4} {entry_price:<10.2f} {confidence:<6.3f} {regime:<15}")
        else:
            print("⚠️  No trades logged yet")
        
        print()
    else:
        print("⚠️  No database found - system may still be initializing")
        print()
    
    # Show recent logs
    print("📝 RECENT LOG ENTRIES (Last 15 lines)")
    print("-" * 80)
    logs = monitor_logs(lines=15)
    for log in logs:
        print(log.rstrip())
    
    print()
    print("="*80)
    print(f"Monitor run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
