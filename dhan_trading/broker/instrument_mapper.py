import pandas as pd
import requests
import os
from pathlib import Path
from dhan_trading.core.logger import system_logger, error_logger
from dhan_trading.core.constants import DHAN_INSTRUMENT_MASTER_URL

class InstrumentMapper:
    """
    Downloads and caches the Dhan Instrument Master CSV.
    Provides methods to instantly lookup security IDs for trading.
    """
    def __init__(self, cache_dir: str = ".trading_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "api-scrip-master.csv"
        self.df = None
        self._load_master()
        
    def _load_master(self):
        """Loads from cache or downloads if not present/old"""
        if not self.cache_file.exists():
            self._download_master()
            
        try:
            self.df = pd.read_csv(self.cache_file, low_memory=False)
            system_logger.info(f"Loaded {len(self.df)} instruments from Dhan Master.")
        except Exception as e:
            error_logger.warning(f"Corrupted cache detected, re-downloading: {e}")
            if self.cache_file.exists():
                os.remove(self.cache_file)
            self._download_master()
            self.df = pd.read_csv(self.cache_file, low_memory=False)
            system_logger.info(f"Successfully re-loaded {len(self.df)} instruments.")
            
    def _download_master(self):
        system_logger.info(f"Downloading Dhan Instrument Master from {DHAN_INSTRUMENT_MASTER_URL}")
        try:
            response = requests.get(DHAN_INSTRUMENT_MASTER_URL, stream=True)
            response.raise_for_status()
            with open(self.cache_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            system_logger.info("Instrument Master downloaded successfully.")
        except Exception as e:
            error_logger.error(f"Error downloading instrument master: {e}")
            raise

    def get_security_id(self, trading_symbol: str, exchange: str = "NSE") -> str:
        """
        Look up a security ID by exactly matching the trading symbol.
        E.g., mapping 'NIFTY' to its underlying ID, or 'NIFTY 20 JUN 22000 CE'
        """
        if self.df is None:
            return None
        
        # Depending on Dhan's actual CSV column names (usually SEM_TRADING_SYMBOL and SEM_EXM_EXCH_ID)
        # We will assume standard columns: SEM_TRADING_SYMBOL, SEM_EXM_EXCH_ID, SEM_SMST_SECURITY_ID
        try:
            # Find the row
            # Note: the exact column names need to match Dhan's format. 
            # Often they are: SEM_EXM_EXCH_ID, SEM_TRADING_SYMBOL, SEM_SMST_SECURITY_ID
            mask = (self.df['SEM_TRADING_SYMBOL'] == trading_symbol) & (self.df['SEM_EXM_EXCH_ID'] == exchange)
            result = self.df[mask]
            if not result.empty:
                return str(result.iloc[0]['SEM_SMST_SECURITY_ID'])
            return None
        except KeyError:
            # Fallback if column names differ
            # Just search the whole dataframe string-wise (slower but safer if schema changes)
            error_logger.warning("Column names mismatch in Dhan CSV. Fallback search.")
            return None
