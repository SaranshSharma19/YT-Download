from abc import ABC, abstractmethod

class BaseBroker(ABC):
    @abstractmethod
    def get_funds(self) -> float:
        pass
        
    @abstractmethod
    def place_order(self, security_id: str, exchange: str, tx_type: str, qty: int, order_type: str, price: float) -> dict:
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        pass
        
    @abstractmethod
    def get_positions(self) -> dict:
        pass
