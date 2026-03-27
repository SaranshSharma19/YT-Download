import time
import uuid
from dhan_trading.broker.base import BaseBroker
from dhan_trading.core.logger import trade_logger, system_logger
from dhan_trading.core.config import Config
from dhan_trading.core.constants import TRANSACTION_COSTS

class PaperBroker(BaseBroker):
    """
    Simulates trading through Dhan. Handles slippage, transaction costs,
    virtual margin blocking, and local position accounting.
    """
    def __init__(self, start_balance: float = Config.ACCOUNT_BALANCE):
        self.balance = start_balance
        self.used_margin = 0.0
        self.positions = {}
        self.orders = {}
        system_logger.info(f"Initialized PaperBroker with virtual balance: ₹{self.balance:.2f}")

    def get_funds(self) -> float:
        return self.balance - self.used_margin

    def _calculate_total_cost(self, val: float, is_buy: bool, is_option: bool = True) -> float:
        # Brokerage + GST
        brokerage = TRANSACTION_COSTS['brokerage_per_order']
        gst = (brokerage + (val * TRANSACTION_COSTS['exchange_txn_charge'])) * TRANSACTION_COSTS['gst']
        
        # STT (Only on sell for options, applied broadly here)
        stt = 0
        if not is_buy:
            stt = val * (TRANSACTION_COSTS['stt_options_sell'] if is_option else TRANSACTION_COSTS['stt_futures'])
        
        stamp_duty = (val * TRANSACTION_COSTS['stamp_duty']) if is_buy else 0
        exch = val * TRANSACTION_COSTS['exchange_txn_charge']
        sebi = val * TRANSACTION_COSTS['sebi_turnover']
        
        return brokerage + gst + stt + stamp_duty + exch + sebi

    def place_order(self, security_id: str, exchange: str, tx_type: str, qty: int, order_type: str, price: float) -> dict:
        """Simulate Realistic Order Placing"""
        
        # Simulate slippage (Assuming MARKET orders slip by 0.05%)
        # For a literal paper trade with historical data we take exactly the `price` passed in 
        # but realistically there's some slippage.
        simulated_fill = price * (1.0005 if tx_type == "BUY" else 0.9995) if order_type == "MARKET" else price
        
        trade_value = simulated_fill * qty
        is_opt = "CE" in security_id or "PE" in security_id or "OPT" in exchange
        
        cost = self._calculate_total_cost(trade_value, tx_type == "BUY", is_opt)
        
        # Margin Check
        if tx_type == "BUY" and self.get_funds() < (trade_value + cost):
            trade_logger.warning(f"MARGIN SHORTFALL: Required {trade_value+cost:.2f}, Available {self.get_funds():.2f}")
            return {"status": "failure", "remarks": "Insufficient Virtual Margin"}

        # Execute
        order_id = str(uuid.uuid4())
        
        if tx_type == "BUY":
            self.balance -= cost
            if security_id in self.positions:
                self.positions[security_id]['qty'] += qty
                self.positions[security_id]['avg_price'] = ((self.positions[security_id]['avg_price'] * (self.positions[security_id]['qty']-qty)) + trade_value) / self.positions[security_id]['qty']
            else:
                self.positions[security_id] = {'qty': qty, 'avg_price': simulated_fill}
        else:
            self.balance -= cost
            if security_id in self.positions:
                # Calculate Realized PnL
                pnl = (simulated_fill - self.positions[security_id]['avg_price']) * qty
                self.balance += (self.positions[security_id]['avg_price'] * qty) + pnl
                
                self.positions[security_id]['qty'] -= qty
                if self.positions[security_id]['qty'] <= 0:
                    del self.positions[security_id]
            else:
                # Creating Short Position (Requires heavy margin typically, simplifying here)
                self.positions[security_id] = {'qty': -qty, 'avg_price': simulated_fill}

        trade_logger.info(f"PAPER FILL: {tx_type} {qty} {security_id} @ {simulated_fill:.2f} | Cost: ₹{cost:.2f}")

        return {
            "status": "success",
            "orderId": order_id,
            "orderStatus": "TRADED",
            "tradedPrice": simulated_fill
        }

    def cancel_order(self, order_id: str) -> dict:
        return {"status": "success", "orderStatus": "CANCELLED"}

    def get_positions(self) -> dict:
        return {"status": "success", "data": self.positions}
