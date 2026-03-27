from dhanhq import dhanhq
from dhan_trading.core.config import Config
from dhan_trading.core.logger import system_logger, error_logger

class DhanBrokerAdapter:
    """
    Wrapper around the official DhanHQ API SDK.
    Handles authentication, live order placement, and live portfolio fetch.
    """
    def __init__(self):
        self.client_id = Config.DHAN_CLIENT_ID
        self.token = Config.DHAN_ACCESS_TOKEN
        
        if not self.client_id or not self.token:
            system_logger.warning("Dhan Credentials not set! Paper trading only allowed without live market feed validation.")
            self.dhan = None
        else:
            self.dhan = dhanhq(self.client_id, self.token)
            system_logger.info("Successfully authenticated with DhanHQ API.")
            
    def get_funds(self):
        """Fetch available margin from Dhan"""
        if not self.dhan: return Config.ACCOUNT_BALANCE
        try:
            response = self.dhan.get_fund_limits()
            if response['status'] == 'success':
                return response['data']['availabelBalance']
            else:
                error_logger.error(f"Failed to fetch funds. Status: {response['status']}")
                return Config.ACCOUNT_BALANCE
        except Exception as e:
            error_logger.error(f"Error fetching funds: {e}")
            return Config.ACCOUNT_BALANCE

    def place_order(self, security_id: str, exchange_segment: str, transaction_type: str, quantity: int, order_type: str, product_type: str, price: float = 0):
        """Place live order to Dhan"""
        if not self.dhan:
            error_logger.error("Dhan credentials missing. Cannot execute live order.")
            return {"status": "failure", "remarks": "No API Credentials"}
            
        try:
            order_res = self.dhan.place_order(
                security_id=security_id,
                exchange_segment=exchange_segment,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product_type=product_type,
                price=price
            )
            return order_res
        except Exception as e:
            error_logger.error(f"Dhan placing order failed: {e}")
            return {"status": "failure", "remarks": str(e)}

    def cancel_order(self, order_id: str):
        if self.dhan:
            return self.dhan.cancel_order(order_id)
        return {"status": "failure"}

    def get_positions(self):
        if self.dhan:
            return self.dhan.get_positions()
        return {"status": "failure"}
