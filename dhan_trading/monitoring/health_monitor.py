import time
import os
import psutil
from dhan_trading.core.logger import system_logger, error_logger

class HealthMonitor:
    """
    Monitors system resources, broker connectivity, and application health.
    Designed to trigger fail-safes if the environment becomes unstable.
    """
    def __init__(self, broker):
        self.broker = broker
        
    def check_system_resources(self) -> bool:
        """Check CPU and RAM usage"""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            
            if cpu > 95.0:
                system_logger.warning(f"HIGH CPU USAGE: {cpu}%. (Expected during startup, proceeding...)")
            if mem > 90.0:
                system_logger.warning(f"HIGH MEMORY USAGE: {mem}%")
                return False
            return True
        except Exception as e:
            error_logger.error(f"HealthMonitor resource check failed: {e}")
            return True # Fail open to avoid blocking trading falsely

    def check_broker_latency(self) -> float:
        """Ping broker to check latency. Under 200ms is ideal."""
        start = time.time()
        try:
            # A simple lightweight call to check responsiveness
            self.broker.get_funds()
            latency = (time.time() - start) * 1000
            if latency > 500:
                system_logger.warning(f"HIGH BROKER LATENCY: {latency:.2f}ms")
            return latency
        except Exception as e:
            error_logger.error(f"Broker latency check failed. Connection might be dead: {e}")
            return -1.0
            
    def run_diagnostics(self):
        res_ok = self.check_system_resources()
        lat = self.check_broker_latency()
        
        system_logger.info(f"Diagnostics - Resources OK: {res_ok} | Dhan API Latency: {lat:.2f}ms")
        return res_ok and lat >= 0
