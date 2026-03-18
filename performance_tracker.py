"""
Performance Tracking & Feedback Loop System
Tracks all trades, calculates metrics, and provides insights for optimization
"""

import sqlite3
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path


class PerformanceTracker:
    """
    Comprehensive performance tracking system with:
    - Trade logging to SQLite database
    - Performance metrics calculation
    - Regime-specific analysis
    - Time-of-day analysis
    - Model performance monitoring
    """
    
    def __init__(self, db_path: str = "trading_performance.db"):
        self.logger = logging.getLogger('PerformanceTracker')
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                index_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                position_size REAL NOT NULL,
                confidence REAL NOT NULL,
                validation_score REAL,
                outcome TEXT,
                pnl REAL,
                pnl_percent REAL,
                regime TEXT,
                time_of_day TEXT,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                exit_reason TEXT,
                features_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                index_name TEXT NOT NULL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                total_pnl REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, index_name)
            )
        """)
        
        # Daily summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL UNIQUE,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                largest_win REAL,
                largest_loss REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def log_trade(
        self,
        trade_id: int, # Accept trade_id passed from bot
        index_name: str,
        direction: str,
        entry_price: float,
        position_size: float,
        confidence: float,
        regime: str,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: Optional[float] = None,
        validation_score: Optional[float] = None,
        features: Optional[Dict] = None,
        symbol: Optional[str] = None,
        size: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> int:
        """
        Log a new trade entry.
        Accepts flexible arguments for compatibility.
        """
        # Normalize arguments
        actual_index = index_name if index_name else symbol
        actual_size = position_size if position_size else size
        actual_sl = stop_loss if stop_loss else sl
        actual_tp = take_profit_1 if take_profit_1 else tp
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now()
        time_of_day = self._classify_time_of_day(timestamp)
        features_json = json.dumps(features) if features else None
        
        # Check if trade_id allows text or int. The DB schema says INTEGER PRIMARY KEY AUTOINCREMENT
        # but the bot is passing a string ID. We should probably adjust to use the DB's ID 
        # OR update the bot to use the DB's ID. 
        # For now, let's insert and let the DB generate the ID, and we return it.
        # But wait, the bot generates a string ID "NIFTY_2023..."
        # The schema definition has 'id INTEGER PRIMARY KEY'.
        # We need to add a 'trade_id' text column to the schema if we want to store the bot's ID,
        # or just ignore the bot's ID and return the DB ID.
        # Let's check the schema in _initialize_database.
        # It has `id INTEGER PRIMARY KEY`. It does NOT have a separate text `trade_id`.
        # Strategy: We will modify the schema to add `trade_ref` for the bot's string ID if needed,
        # but for now, let's just insert and return the new DB ID which the bot will use.
        
        cursor.execute("""
            INSERT INTO trades (
                timestamp, index_name, direction, entry_price, position_size,
                confidence, validation_score, regime, time_of_day,
                stop_loss, take_profit_1, take_profit_2, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, actual_index, direction, entry_price, actual_size,
            confidence, validation_score, regime, time_of_day,
            actual_sl, actual_tp, take_profit_2, features_json
        ))
        
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.logger.info(
            f"Trade logged: DB_ID={new_id} | {actual_index} {direction} @ {entry_price:.2f} "
            f"| Confidence: {confidence:.3f} | Regime: {regime}"
        )
        
        return new_id
    
    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        pnl: Optional[float] = None,
        exit_reason: str = "manual"
    ) -> Optional[Dict]:
        """
        Update trade with exit information and calculate P&L
        
        Returns:
            Dict with trade outcome details or None if trade not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get trade details
        cursor.execute("""
            SELECT entry_price, position_size, direction
            FROM trades WHERE id = ?
        """, (trade_id,))
        
        result = cursor.fetchone()
        if not result:
            self.logger.warning(f"Trade ID {trade_id} not found")
            conn.close()
            return None
        
        entry_price, position_size, direction = result
        
        # Calculate P&L if not provided
        if pnl is None:
            if direction == 'CE':
                pnl = (exit_price - entry_price) * position_size
            else:  # PE
                pnl = (entry_price - exit_price) * position_size
        
        pnl_percent = (pnl / (entry_price * position_size)) * 100
        outcome = 'win' if pnl > 0 else ('loss' if pnl < 0 else 'breakeven')
        
        # Update trade
        cursor.execute("""
            UPDATE trades
            SET exit_price = ?, pnl = ?, pnl_percent = ?, outcome = ?, exit_reason = ?
            WHERE id = ?
        """, (exit_price, pnl, pnl_percent, outcome, exit_reason, trade_id))
        
        conn.commit()
        conn.close()
        
        trade_result = {
            'trade_id': trade_id,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'outcome': outcome
        }
        
        self.logger.info(
            f"Trade updated: ID={trade_id} | {outcome.upper()} | "
            f"P&L: {pnl:+.2f} ({pnl_percent:+.2f}%)"
        )
        
        # Update daily summary
        self._update_daily_summary(datetime.now().date())
        
        return trade_result
    
    def _classify_time_of_day(self, timestamp: datetime) -> str:
        """Classify time into trading session periods"""
        hour = timestamp.hour
        minute = timestamp.minute
        time_val = hour + minute / 60.0
        
        if 9.25 <= time_val < 10.5:
            return 'opening'
        elif 10.5 <= time_val < 14.0:
            return 'mid_session'
        elif 14.0 <= time_val < 15.5:
            return 'closing'
        else:
            return 'after_hours'
    
    def _update_daily_summary(self, date):
        """Update daily summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get daily stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                AVG(CASE WHEN outcome = 'win' THEN pnl_percent END) as avg_win,
                AVG(CASE WHEN outcome = 'loss' THEN pnl_percent END) as avg_loss,
                MAX(pnl) as largest_win,
                MIN(pnl) as largest_loss
            FROM trades
            WHERE DATE(timestamp) = ? AND outcome IS NOT NULL
        """, (date,))
        
        stats = cursor.fetchone()
        
        if stats and stats[0] > 0:
            total_trades, winning_trades, losing_trades, total_pnl, avg_win, avg_loss, largest_win, largest_loss = stats
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO daily_summary (
                    date, total_trades, winning_trades, losing_trades,
                    total_pnl, win_rate, avg_win, avg_loss,
                    largest_win, largest_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date, total_trades, winning_trades, losing_trades,
                total_pnl, win_rate, avg_win or 0, avg_loss or 0,
                largest_win or 0, largest_loss or 0
            ))
            
            conn.commit()
        
        conn.close()
    
    def get_recent_stats(
        self,
        index_name: Optional[str] = None,
        lookback_days: int = 30
    ) -> Dict:
        """
        Get recent performance statistics
        
        Args:
            index_name: Filter by specific index (None for all)
            lookback_days: Number of days to look back
        
        Returns:
            Dict with win_rate, avg_win, avg_loss, etc.
        """
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN outcome = 'win' THEN pnl_percent END) as avg_win,
                AVG(CASE WHEN outcome = 'loss' THEN pnl_percent END) as avg_loss,
                SUM(pnl) as total_pnl,
                MAX(pnl) as max_win,
                MIN(pnl) as max_loss
            FROM trades
            WHERE timestamp >= ? AND outcome IS NOT NULL
        """
        
        params = [cutoff_date]
        if index_name:
            query += " AND index_name = ?"
            params.append(index_name)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty or df['total_trades'].iloc[0] == 0:
            return {
                'win_rate': 0.5,
                'avg_win': 0.01,
                'avg_loss': 0.01,
                'total_trades': 0,
                'total_pnl': 0,
                'profit_factor': 1.0
            }
        
        row = df.iloc[0]
        win_rate = row['wins'] / row['total_trades'] if row['total_trades'] > 0 else 0.5
        avg_win = abs(row['avg_win']) if pd.notna(row['avg_win']) else 0.01
        avg_loss = abs(row['avg_loss']) if pd.notna(row['avg_loss']) else 0.01
        
        # Calculate profit factor
        total_wins = row['wins'] * avg_win if row['wins'] > 0 else 0
        total_losses = (row['total_trades'] - row['wins']) * avg_loss if row['total_trades'] > row['wins'] else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 1.0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win / 100,  # Convert to decimal
            'avg_loss': avg_loss / 100,
            'total_trades': int(row['total_trades']),
            'total_pnl': float(row['total_pnl']) if pd.notna(row['total_pnl']) else 0,
            'profit_factor': profit_factor,
            'max_win': float(row['max_win']) if pd.notna(row['max_win']) else 0,
            'max_loss': float(row['max_loss']) if pd.notna(row['max_loss']) else 0
        }
    
    def get_performance_by_regime(self, lookback_days: int = 30) -> pd.DataFrame:
        """Get performance breakdown by market regime"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        query = """
            SELECT 
                regime,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN outcome = 'win' THEN pnl_percent END) as avg_win,
                AVG(CASE WHEN outcome = 'loss' THEN pnl_percent END) as avg_loss,
                SUM(pnl) as total_pnl
            FROM trades
            WHERE timestamp >= ? AND outcome IS NOT NULL
            GROUP BY regime
        """
        
        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()
        
        if not df.empty:
            df['win_rate'] = df['wins'] / df['total_trades']
            df = df.sort_values('total_trades', ascending=False)
        
        return df
    
    def get_performance_by_time(self, lookback_days: int = 30) -> pd.DataFrame:
        """Get performance breakdown by time of day"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        query = """
            SELECT 
                time_of_day,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN outcome = 'win' THEN pnl_percent END) as avg_win,
                AVG(CASE WHEN outcome = 'loss' THEN pnl_percent END) as avg_loss,
                SUM(pnl) as total_pnl
            FROM trades
            WHERE timestamp >= ? AND outcome IS NOT NULL
            GROUP BY time_of_day
        """
        
        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()
        
        if not df.empty:
            df['win_rate'] = df['wins'] / df['total_trades']
            # Order by session
            session_order = {'opening': 1, 'mid_session': 2, 'closing': 3, 'after_hours': 4}
            df['order'] = df['time_of_day'].map(session_order)
            df = df.sort_values('order').drop('order', axis=1)
        
        return df
    
    def get_summary_report(self, lookback_days: int = 30) -> str:
        """Generate a comprehensive performance summary report"""
        stats = self.get_recent_stats(lookback_days=lookback_days)
        regime_perf = self.get_performance_by_regime(lookback_days=lookback_days)
        time_perf = self.get_performance_by_time(lookback_days=lookback_days)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         PERFORMANCE REPORT (Last {lookback_days} Days)                    ║
╚══════════════════════════════════════════════════════════════╝

📊 OVERALL STATISTICS
  Total Trades:     {stats['total_trades']}
  Win Rate:         {stats['win_rate']*100:.1f}%
  Avg Win:          {stats['avg_win']*100:+.2f}%
  Avg Loss:         {stats['avg_loss']*100:.2f}%
  Profit Factor:    {stats['profit_factor']:.2f}
  Total P&L:        {stats['total_pnl']:+.2f}

📈 PERFORMANCE BY REGIME
"""
        
        if not regime_perf.empty:
            for _, row in regime_perf.iterrows():
                report += f"  {row['regime']:15} | Trades: {int(row['total_trades']):3} | Win Rate: {row['win_rate']*100:5.1f}% | P&L: {row['total_pnl']:+8.2f}\n"
        else:
            report += "  No data available\n"
        
        report += "\n⏰ PERFORMANCE BY TIME OF DAY\n"
        
        if not time_perf.empty:
            for _, row in time_perf.iterrows():
                report += f"  {row['time_of_day']:15} | Trades: {int(row['total_trades']):3} | Win Rate: {row['win_rate']*100:5.1f}% | P&L: {row['total_pnl']:+8.2f}\n"
        else:
            report += "  No data available\n"
        
        report += "\n" + "="*64
        
        return report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    tracker = PerformanceTracker("test_performance.db")
    
    # Log a trade
    trade_id = tracker.log_trade(
        index_name="NIFTY",
        direction="CE",
        entry_price=19500.0,
        position_size=100,
        confidence=0.68,
        regime="trending_up",
        stop_loss=19450.0,
        take_profit_1=19600.0,
        take_profit_2=19650.0,
        validation_score=75.0
    )
    
    # Simulate exit
    tracker.update_trade_exit(trade_id, exit_price=19580.0, exit_reason="tp1_hit")
    
    # Get stats
    stats = tracker.get_recent_stats(lookback_days=30)
    print(f"\nRecent Stats: {stats}")
    
    # Get summary report
    print(tracker.get_summary_report(lookback_days=30))
