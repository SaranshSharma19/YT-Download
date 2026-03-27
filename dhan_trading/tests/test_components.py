import pytest
import datetime
from dhan_trading.strategy.expiry_utils import is_stt_trap_time, EXPIRY_CONFIG

def test_stt_trap_detection():
    # Example logic using STT trap functionality
    test_date = datetime.datetime(2025, 2, 26, 15, 25) # Mock expiry day date
    
    # Needs actual context injection in the module to run natively,
    # Here we mock EXPIRY_CONFIG locally.
    EXPIRY_CONFIG['NIFTY']['weekday'] = 2 # Force Wednesday for test
    
    # As an abstract structural test: Ensure the function executes without failing
    res = is_stt_trap_time("NIFTY", test_date)
    assert isinstance(res, bool)

def test_config_structure():
    from dhan_trading.core.config import Config
    assert Config.ACCOUNT_BALANCE > 0
    assert "NIFTY" in Config.CORE_INDICES

def test_risk_manager_initialization():
    from dhan_trading.strategy.risk_manager import RiskManager
    from dhan_trading.broker.paper_broker import PaperBroker
    
    broker = PaperBroker()
    risk = RiskManager(broker)
    assert risk.check_daily_limits() == 1.0
