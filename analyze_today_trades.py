#!/usr/bin/env python3
"""
Extract today's trades from the database for TradingView analysis
"""
import sqlite3
import pandas as pd
from datetime import datetime, date
import json

def get_today_trades():
    """Extract all trades from today"""
    conn = sqlite3.connect('trading_performance.db')
    
    # Get today's date
    today = date.today()
    
    # Query trades
    query = """
    SELECT 
        id,
        timestamp,
        index_name,
        direction,
        entry_price,
        exit_price,
        position_size,
        confidence,
        validation_score,
        regime,
        time_of_day,
        stop_loss,
        take_profit_1,
        take_profit_2,
        exit_reason,
        pnl,
        pnl_percent,
        features_json
    FROM trades
    WHERE DATE(timestamp) = DATE('now')
    ORDER BY timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def get_rejected_signals():
    """Extract rejected signals from logs"""
    import re
    from pathlib import Path
    
    log_file = Path('logs/trading_bot.log')
    if not log_file.exists():
        return []
    
    rejected = []
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    with open(log_file, 'r') as f:
        for line in f:
            if today_str in line:
                # Look for rejected signals
                if 'REJECTED' in line or 'FILTERED' in line:
                    # Extract index, reason, time
                    match = re.search(r'(\d{2}:\d{2}:\d{2}).*?(NIFTY|BANKNIFTY|FINNIFTY).*?(REJECTED|FILTERED).*?Reason[:\s]+([^|]+)', line)
                    if match:
                        rejected.append({
                            'time': match.group(1),
                            'index': match.group(2),
                            'status': match.group(3),
                            'reason': match.group(4).strip()
                        })
    
    return rejected

if __name__ == '__main__':
    print("="*80)
    print("TODAY'S TRADING ACTIVITY ANALYSIS")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get executed trades
    trades_df = get_today_trades()
    
    print(f"EXECUTED TRADES: {len(trades_df)}")
    print("-"*80)
    
    if not trades_df.empty:
        for idx, trade in trades_df.iterrows():
            print(f"\nTrade #{trade['id']}")
            print(f"  Index: {trade['index_name']}")
            print(f"  Direction: {trade['direction']}")
            print(f"  Entry Time: {trade['timestamp']}")
            print(f"  Entry Price: {trade['entry_price']:.2f}")
            print(f"  Exit Price: {trade['exit_price']:.2f}" if pd.notna(trade['exit_price']) else "  Exit Price: OPEN")
            print(f"  Position Size: {trade['position_size']}")
            print(f"  Confidence: {trade['confidence']:.3f}")
            print(f"  Validation Score: {trade['validation_score']:.1f}" if pd.notna(trade['validation_score']) else "  Validation Score: N/A")
            print(f"  Regime: {trade['regime']}")
            print(f"  Time of Day: {trade['time_of_day']}")
            print(f"  Stop Loss: {trade['stop_loss']:.2f}" if pd.notna(trade['stop_loss']) else "  Stop Loss: N/A")
            print(f"  Take Profit 1: {trade['take_profit_1']:.2f}" if pd.notna(trade['take_profit_1']) else "  Take Profit 1: N/A")
            print(f"  P&L: {trade['pnl']:.2f}" if pd.notna(trade['pnl']) else "  P&L: OPEN")
            print(f"  Exit Reason: {trade['exit_reason']}" if pd.notna(trade['exit_reason']) else "  Exit Reason: OPEN")
            
            # Calculate R:R if available
            if pd.notna(trade['stop_loss']) and pd.notna(trade['take_profit_1']):
                risk = abs(trade['entry_price'] - trade['stop_loss'])
                reward = abs(trade['take_profit_1'] - trade['entry_price'])
                if risk > 0:
                    rr = reward / risk
                    print(f"  Risk:Reward: 1:{rr:.2f}")
    else:
        print("No executed trades found for today.")
    
    # Get rejected signals
    print("\n" + "="*80)
    print("REJECTED SIGNALS (from logs)")
    print("="*80)
    rejected = get_rejected_signals()
    print(f"Found {len(rejected)} rejected signals")
    
    if rejected:
        for sig in rejected:
            print(f"\n  Time: {sig['time']} | Index: {sig['index']} | Status: {sig['status']}")
            print(f"  Reason: {sig['reason']}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if not trades_df.empty:
        total_trades = len(trades_df)
        closed_trades = trades_df[trades_df['exit_price'].notna()]
        open_trades = trades_df[trades_df['exit_price'].isna()]
        
        print(f"Total Trades: {total_trades}")
        print(f"Closed Trades: {len(closed_trades)}")
        print(f"Open Trades: {len(open_trades)}")
        
        if len(closed_trades) > 0:
            total_pnl = closed_trades['pnl'].sum()
            winning_trades = closed_trades[closed_trades['pnl'] > 0]
            losing_trades = closed_trades[closed_trades['pnl'] <= 0]
            
            print(f"\nClosed Trades P&L: ₹{total_pnl:.2f}")
            print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(closed_trades)*100:.1f}%)")
            print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(closed_trades)*100:.1f}%)")
            
            if len(winning_trades) > 0:
                avg_win = winning_trades['pnl'].mean()
                print(f"Average Win: ₹{avg_win:.2f}")
            
            if len(losing_trades) > 0:
                avg_loss = losing_trades['pnl'].mean()
                print(f"Average Loss: ₹{avg_loss:.2f}")
            
            if len(losing_trades) > 0 and len(winning_trades) > 0:
                profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')
                print(f"Profit Factor: {profit_factor:.2f}")
    
    # Export to JSON for easy reference
    output = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'trades': trades_df.to_dict('records') if not trades_df.empty else [],
        'rejected_signals': rejected,
        'summary': {
            'total_trades': len(trades_df) if not trades_df.empty else 0,
            'closed_trades': len(trades_df[trades_df['exit_price'].notna()]) if not trades_df.empty else 0,
            'open_trades': len(trades_df[trades_df['exit_price'].isna()]) if not trades_df.empty else 0,
        }
    }
    
    with open('today_trades_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n\nAnalysis exported to: today_trades_analysis.json")
    print("\n" + "="*80)
    print("TRADINGVIEW ANALYSIS INSTRUCTIONS")
    print("="*80)
    print("""
1. Open TradingView and navigate to:
   - NIFTY: NSE:NIFTY (5-minute chart)
   - BANKNIFTY: NSE:BANKNIFTY (5-minute chart)
   - FINNIFTY: NSE:FINNIFTY (5-minute chart)

2. For each trade, go to the entry time and analyze:
   - Market structure (trending/sideways/choppy)
   - Price action (breakout/retest/reversal)
   - Key indicators: RSI, MACD, ADX, VWAP, Volume
   - Support/Resistance levels
   - Risk:Reward ratio

3. For rejected signals, check why they were rejected and if:
   - The rejection was justified (false positive avoided)
   - A valid setup was missed (false negative)

4. Document findings in the analysis report.
    """)
