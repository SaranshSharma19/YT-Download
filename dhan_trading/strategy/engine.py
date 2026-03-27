import time
import datetime
import sys
from dhan_trading.core.logger import system_logger, error_logger, trade_logger
from dhan_trading.core.config import Config
from dhan_trading.strategy.expiry_utils import is_stt_trap_time
from dhan_trading.strategy.bridge import MonolithBridge

class TradingEngine:
    """
    Core heart-beat engine for running live/paper trading.
    Monitors connections, manages open positions, checks limits, and executes signals.
    """
    def __init__(self, broker, data_feed, risk_manager, validator, instrument_mapper, regime_detector):
        self.broker = broker
        self.data_feed = data_feed
        self.risk_manager = risk_manager
        self.validator = validator
        self.mapper = instrument_mapper
        self.regime_detector = regime_detector
        self.running = False
        self.open_trades = {}
        self._poll_timer = 0
        # Inject self.broker so the bridge can place paper orders
        self.bridge = MonolithBridge(broker=broker) if Config.USE_MONOLITH_BRIDGE else None
        
    def _get_current_time(self):
        return datetime.datetime.now()

    def _is_square_off_time(self) -> bool:
        now = self._get_current_time()
        sq_time_str = Config.SQUARE_OFF_TIME # "15:10"
        sq_hour, sq_minute = map(int, sq_time_str.split(':'))
        
        # Check if current time is roughly >= 15:10
        if now.hour > sq_hour or (now.hour == sq_hour and now.minute >= sq_minute):
            return True
        return False
        
    def _check_stt_trap(self) -> bool:
        """Check if any position is an option on expiry day nearing 15:20"""
        now = self._get_current_time()
        for symbol, pos in list(self.open_trades.items()):
            # Simplified logic mapping generic base index from symbol
            base_index = "NIFTY" if "NIFTY" in symbol else "BANKNIFTY"
            if is_stt_trap_time(base_index, now) and pos['qty'] < 0:
                trade_logger.warning(f"STT Trap Avoidance: Squaring off short options {symbol}")
                self.close_position(symbol, "STT Trap Auto-SquareOff")

    def _square_off_all(self):
        system_logger.info("Initiating strict 15:10 Intraday Square-Off for all positions.")
        for symbol in list(self.open_trades.keys()):
            self.close_position(symbol, reason="Daily Auto Square-Off")
        self.running = False
        
    def close_position(self, symbol: str, reason: str = "Manual"):
        pos = self.open_trades.get(symbol)
        if not pos: return
        
        qty = pos['qty']
        action = "SELL" if qty > 0 else "BUY"
        # Simulate market fetch price for simplicity (assume close price)
        # In a real tick architecture, this would use active WebSocket LTP.
        # Placing order cancels the internal state
        res = self.broker.place_order(symbol, "NSE", action, abs(qty), "MARKET", 0)
        
        if res.get('status') == 'success':
            trade_logger.info(f"Closed {symbol} | Reason: {reason}")
            del self.open_trades[symbol]
        else:
            error_logger.critical(f"Failed to close position {symbol}: {res}")

    def execute_signal(self, signal: dict):
        base_index = signal['index']
        # Lookup realistic token
        # E.g., getting NIFTY ATM CE for 20 JUN. Simulation uses index name directly.
        symbol = self.mapper.get_security_id(base_index, "NSE") or base_index 
        
        direction = signal['direction']
        price = signal.get('price', 100) # Proxy price
        atr = signal.get('atr', 20)
        
        # Use signal position size if available, otherwise calculate using risk manager
        qty = signal.get('position_size', 0)
        risk = 0
        if qty <= 0:
            qty, risk = self.risk_manager.calculate_position_size(signal, price, atr)
        
        if qty <= 0:
            return
            
        action = "BUY" # Let's assume strategy only buys options
        # Fire to broker
        res = self.broker.place_order(symbol, "NSE", action, qty, "MARKET", price)
        
        if res.get('status') == 'success':
            # Use signal SL/TP if provided by the monolith brain, otherwise fallback to risk manager
            sl = signal.get('stop_loss', signal.get('sl'))
            tp1 = signal.get('take_profit_1', signal.get('tp1'))
            
            if sl is None:
                sl = self.risk_manager.calculate_exits(price, direction, atr, signal.get('regime', 'unknown'))['stop_loss']
            if tp1 is None:
                tp1 = self.risk_manager.calculate_exits(price, direction, atr, signal.get('regime', 'unknown'))['tp1']
            
            self.open_trades[symbol] = {
                'qty': qty,
                'entry': price,
                'sl': sl,
                'tp1': tp1
            }
            system_logger.info(f"Entered trade: {symbol} Qty: {qty} @ {price}")

    def manage_positions(self):
        """Heartbeat check for SL/TP hits using realistic LTP prices."""
        for symbol, pos in list(self.open_trades.items()):
            try:
                # Fetch latest price for SL/TP monitoring
                # E.g. using yfinance or whatever data_feed provides
                # In paper trading, we use the fallback or live data feed
                today_str = datetime.date.today().strftime('%Y-%m-%d')
                df = self.data_feed.fetch_historical_candles(symbol, "NSE", "1", today_str, today_str)
                if df is None or df.empty:
                    current_price = pos['entry'] # Fallback if no fresh data
                else:
                    current_price = df['close'].iloc[-1]
                
                # Check Stop Loss
                if current_price <= pos['sl']:
                    trade_logger.warning(f"STOP LOSS HIT for {symbol} @ {current_price} | SL: {pos['sl']}")
                    self.close_position(symbol, "Stop Loss")
                
                # Check Take Profit
                elif current_price >= pos['tp1']:
                    trade_logger.info(f"TAKE PROFIT HIT for {symbol} @ {current_price} | TP1: {pos['tp1']}")
                    self.close_position(symbol, "Take Profit 1")
            except Exception as e:
                error_logger.error(f"Error managing position for {symbol}: {e}")

    def _poll_market(self):
        """Run the market scan. Delegates to MonolithBridge (trading.py) if enabled."""
        if Config.USE_MONOLITH_BRIDGE and self.bridge:
            # === Full Monolith Mode ===
            # Let trading.py drive everything: data, regime, ML, entry/exit, paper orders
            for name in Config.CORE_INDICES.keys():
                yf_ticker = Config.YFINANCE_TICKERS.get(name) or name
                try:
                    result = self.bridge.run_lifecycle(name, yf_ticker, self.open_trades)
                    status = result.get('status', '-')
                    price = result.get('price', 0)
                    regime = result.get('regime', 'N/A')
                    signal = result.get('signal', '-')
                    system_logger.info(
                        f"LIFECYCLE | {name} | Status: {status} | Price: {price:.1f} | "
                        f"Regime: {regime} | Signal: {signal}"
                    )
                except Exception as e:
                    error_logger.error(f"Lifecycle error for {name}: {e}")
            return
        
        # === Modular Mode (fallback when bridge is OFF) ===
        if not self.data_feed:
            return

        for name in Config.CORE_INDICES.keys():
            system_logger.debug(f"Polling market data for {name}...")
            try:
                today_str = datetime.date.today().strftime('%Y-%m-%d')
                security_id = self.mapper.get_security_id(name) or name
                df = self.data_feed.fetch_historical_candles(
                    security_id, "NSE", "5", today_str, today_str
                )
                
                if df is None or df.empty:
                    system_logger.info(f"SCAN | {name} | No historical data received. Skipping.")
                    continue
                
                # 1. Detect Regime
                regime = self.regime_detector.detect_regime(df)
                regime_params = self.regime_detector.get_regime_parameters(regime)
                context = {'regime': regime, 'parameters': regime_params}
                
                # 2. Run Validation Score for CE and PE directions
                for direction in ['CE', 'PE']:
                    temp_sig = {'direction': direction, 'index': name, 'price': df['close'].iloc[-1]}
                    is_valid, score, msg = self.validator.validate_trade_setup(temp_sig, df, context)
                    
                    system_logger.info(f"SCAN | {name} | {direction} | LTP: {df['close'].iloc[-1]:.1f} | Regime: {regime.upper()} | Score: {score}/60")
                    
                    if is_valid:
                        atr = df['high'].iloc[-1] - df['low'].iloc[-1]
                        self.execute_signal({
                            'index': name, 
                            'direction': direction, 
                            'price': df['close'].iloc[-1], 
                            'regime': regime,
                            'atr': atr
                        })
                        
            except Exception as e:
                error_logger.error(f"Polling error for {name}: {e}")

    def run(self):
        system_logger.info("Trading Engine started.")
        self.running = True
        
        # Initial poll to avoid waiting 30s
        self._poll_market()
        self._poll_timer = time.time()

        while self.running:
            try:
                # 1. Market Hours Check
                if self._is_square_off_time():
                    self._square_off_all()
                    break
                    
                # 2. Daily Loss Limit Safety
                if self.risk_manager.check_daily_limits() == 0.0:
                    self._square_off_all()
                    system_logger.critical("Engine halting due to max daily loss limit.")
                    break
                    
                # 3. Handle STT Expiry Trap
                self._check_stt_trap()
                
                # 4. Manage active stops
                self.manage_positions()
                
                # 5. Poll Market every 30 seconds
                if time.time() - self._poll_timer > 30:
                    self._poll_market()
                    self._poll_timer = time.time()
                
                # Sleep heavily for 1 second real heartbeat
                time.sleep(1)
                
            except KeyboardInterrupt:
                system_logger.warning("KeyboardInterrupt caught. Squaring off and shutting down safely.")
                self._square_off_all()
                break
            except Exception as e:
                error_logger.critical(f"Engine Loop Error: {e}")
                # Crash safety
                self._square_off_all()
                break
        
        system_logger.info("Trading Engine shutdown complete.")
