import sys
import os
import logging
from dhan_trading.core.logger import system_logger

class MonolithBridge:
    """
    Connects the monolithic TradingBot (from trading.py) to the modular TradingEngine.
    
    When USE_MONOLITH_BRIDGE=True, the bridge delegates the FULL trading lifecycle
    (data fetch, regime detection, ML signal, validation, entry/exit) to trading.py.
    The modular engine then only handles the broker order placement.
    """
    def __init__(self, broker=None):
        self.bot = None
        self.broker = broker  # Injected PaperBroker or DhanBrokerAdapter
        
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if root_dir not in sys.path:
                sys.path.append(root_dir)
            
            from trading import TradingBot
            self.bot = TradingBot()
            self.bot.load_or_train_models()
            
            system_logger.info("MonolithBridge: Successfully integrated TradingBot (monolith) algorithm.")
        except Exception as e:
            system_logger.error(f"MonolithBridge: Failed to initialize: {e}")
            import traceback
            system_logger.error(traceback.format_exc())

    def _sync_positions(self, open_trades: dict):
        """Sync the modular engine's open_trades into the monolith's internal state."""
        monolith_positions = {}
        for s, p in open_trades.items():
            if "BANKNIFTY" in s:
                monolith_positions["BANKNIFTY"] = p
            elif "FINNIFTY" in s:
                monolith_positions["FINNIFTY"] = p
            elif "NIFTY" in s:
                monolith_positions["NIFTY"] = p
        self.bot.state.state['open_positions'] = monolith_positions

    def run_lifecycle(self, name: str, symbol: str, open_trades: dict) -> dict:
        """
        Run the FULL trade lifecycle from trading.py:
          - Fetches data using trading.py's data fetcher
          - Detects regime using trading.py's regime detector (proper ML-based)
          - Runs ML model and validates signal
          - Handles position monitoring and exits
        
        Returns lifecycle result dict for logging.
        If the monolith approves an entry, this also places a PAPER ORDER via broker.
        """
        if not self.bot:
            return {'index': name, 'status': 'Bridge Not Ready'}
        
        try:
            self._sync_positions(open_trades)
            
            # Call the FULL lifecycle — this handles everything internally:
            # data → regime → ML → validation → _execute_entry / _monitor_position
            result = self.bot.execute_trade_lifecycle(name, symbol)
            
            # If the monolith approved an entry, also execute a paper order
            if result.get('status') == 'APPROVED' and self.broker:
                direction = result.get('signal', '-')
                price = result.get('price', 0)
                
                if price > 0 and direction in ('CE', 'PE'):
                    lot_sizes = {'NIFTY': 25, 'BANKNIFTY': 15, 'FINNIFTY': 40}
                    lot = lot_sizes.get(name, 25)
                    
                    try:
                        res = self.broker.place_order(
                            security_id=symbol,
                            exchange_segment="NSE",
                            action="BUY",
                            quantity=lot,
                            order_type="MARKET",
                            price=price
                        )
                        if res.get('status') == 'success':
                            system_logger.info(
                                f"MonolithBridge: PAPER ORDER PLACED | {name} {direction} | "
                                f"Qty: {lot} @ ₹{price:.2f} | Fill: ₹{res.get('fill_price', price):.2f}"
                            )
                    except Exception as e:
                        system_logger.error(f"MonolithBridge: Broker order failed for {name}: {e}")
            
            return result
        except Exception as e:
            system_logger.error(f"MonolithBridge: Error in lifecycle for {name}: {e}")
            import traceback
            system_logger.error(traceback.format_exc())
            return {'index': name, 'status': 'Error'}

    def get_signal(self, name: str, symbol: str, open_trades: dict):
        """Returns only the signal without placing order (used for inspection)."""
        if not self.bot:
            return None
        try:
            self._sync_positions(open_trades)
            return self.bot.generate_signal(name, symbol)
        except Exception as e:
            system_logger.error(f"MonolithBridge: Error generating signal for {name}: {e}")
            return None
