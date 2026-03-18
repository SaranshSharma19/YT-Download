import os
import time
import json
import logging
import warnings
import requests
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from functools import wraps

import numpy as np
import pandas as pd
import pytz
from bs4 import BeautifulSoup

# --- Import Libraries with Error Handling ---
# This makes it clearer if a required library is missing
try:
    import yfinance as yf
except ImportError:
    print("Warning: 'yfinance' library not found. Install with: pip install yfinance")
    raise

try:
    # Technical Analysis
    from ta import add_all_ta_features
    from ta.volatility import BollingerBands, KeltnerChannel, AverageTrueRange
    from ta.trend import ADXIndicator, PSARIndicator, CCIIndicator, MACD
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.volume import OnBalanceVolumeIndicator
except ImportError:
    print("Warning: 'ta' library not found. Install with: pip install ta")
    raise

try:
    # Machine Learning
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    from sklearn.base import BaseEstimator, ClassifierMixin
except ImportError:
    print("Warning: 'sklearn' library not found. Install with: pip install scikit-learn")
    raise

# Gradient Boosting Models - Import with fallbacks/checks
LGBMClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("Warning: 'lightgbm' not found. Ensemble will use available models.")

XGBClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    print("Warning: 'xgboost' not found. Ensemble will use available models.")

CatBoostClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    print("Warning: 'catboost' not found. Ensemble will use available models.")

try:
    import joblib
except ImportError:
    print("Warning: 'joblib' not found. Install with: pip install joblib")
    raise

# Import new modules for enhanced functionality
# Import enhanced modules
try:
    from risk_manager import RiskManager
    from performance_tracker import PerformanceTracker
    from regime_detector import RegimeDetector
    from tradingview_validator import TradingViewValidator
    from signal_validator import SignalValidator
    
except ImportError as e:
    print(f"Warning: Could not import enhanced modules: {e}")
    print("Some features may be limited. Ensure all module files are in the same directory.")
    RiskManager = None
    PerformanceTracker = None
    RegimeDetector = None
    TradingViewValidator = None
    SignalValidator = None

try:
    from expiry_utils import (
        is_expiry_day, is_expiry_week, get_expiry_adjustment, get_next_expiry
    )
    # get_days_to_expiry may not exist in older expiry_utils; handle gracefully
    try:
        from expiry_utils import get_days_to_expiry
    except ImportError:
        def get_days_to_expiry(index_name, from_date=None):
            """Fallback: compute DTE from get_next_expiry."""
            from datetime import date as _date
            ref = from_date or _date.today()
            expiry = get_next_expiry(index_name, ref)
            return (expiry - ref).days
    from enhanced_features import add_enhanced_features
    from improved_targets import calculate_improved_targets
except ImportError as e:
    print(f"Warning: Supplemental ML modules not fully available: {e}. Some features disabled.")
    is_expiry_day = is_expiry_week = get_expiry_adjustment = None
    get_next_expiry = get_days_to_expiry = None
    add_enhanced_features = calculate_improved_targets = None

# ================== State Management ==================
class TradingState:
    """
    Manages persistent state of the trading bot, including open positions.
    Saves to/loads from JSON file.
    """
    def __init__(self, state_file: str = "trading_state.json"):
        self.state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), state_file)
        self.state = {
            "open_positions": {},
            "last_update": None,
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "daily_pnl_by_index": {
                "NIFTY": 0.0,
                "BANKNIFTY": 0.0,
                "FINNIFTY": 0.0
            },
            "last_trade_close_time": {}  # A-7: Persist cooldown across restarts
        }
        self.load_state()

    def load_state(self):
        """Load state from JSON file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                    logging.info(f"Loaded trading state from {self.state_file}")
            except Exception as e:
                logging.error(f"Failed to load state: {e}")

    def save_state(self):
        """Save state to JSON file."""
        try:
            self.state["last_update"] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save state: {e}")

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get open position for a symbol."""
        return self.state["open_positions"].get(symbol)

    def add_position(self, symbol: str, data: Dict):
        """Add a new open position."""
        self.state["open_positions"][symbol] = data
        self.save_state()

    def remove_position(self, symbol: str):
        """Remove an open position."""
        if symbol in self.state["open_positions"]:
            del self.state["open_positions"][symbol]
            self.save_state()

    def update_daily_stats(self, pnl: float, index_name: str = None):
        """Update daily P&L and trade count globally and per-index."""
        # Check if it's a new day to reset
        last_update = self.state.get("last_update")
        if last_update:
            last_date = datetime.fromisoformat(last_update).date()
            if datetime.now().date() > last_date:
                self.state["daily_pnl"] = 0.0
                self.state["daily_trades"] = 0
                self.state["daily_pnl_by_index"] = {
                    "NIFTY": 0.0,
                    "BANKNIFTY": 0.0,
                    "FINNIFTY": 0.0
                }
        
        self.state["daily_pnl"] += pnl
        self.state["daily_trades"] += 1
        
        # C-2: Update per-index P&L
        if index_name:
            if "daily_pnl_by_index" not in self.state:
                self.state["daily_pnl_by_index"] = {"NIFTY": 0.0, "BANKNIFTY": 0.0, "FINNIFTY": 0.0}
            if index_name in self.state["daily_pnl_by_index"]:
                self.state["daily_pnl_by_index"][index_name] += pnl
            else:
                self.state["daily_pnl_by_index"][index_name] = pnl
        
        self.save_state()

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ================== Configuration ==================
class Config:
    """Configuration parameters for the bot."""
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "market_data") # Not actively used for saving, mostly for structure
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    # Use a persistent cache directory
    CACHE_DIR = os.path.join(BASE_DIR, ".trading_cache")

    # Create directories upon import
    for d in [DATA_DIR, MODEL_DIR, LOG_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)

    # Trading Parameters
    INDICES = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "NIFTY_FIN_SERVICE.NS"
    }

    # Backup symbols are less reliable and can be complex.
    # Relying on robust multiple sources for the main symbol is often better.
    # Keeping this commented out for now to simplify and rely on source fallbacks.
    # BACKUP_SYMBOLS = {
    #     "^NSEI": ["NIFTY50_IND.NS", "NIFTY.NS"],
    #     "^NSEBANK": ["BANKNIFTY.NS", "BANKNIFTY_IND.NS"],
    #     "NIFTY_FIN_SERVICE.NS": ["FINNIFTY.NS", "FINNIFTY_IND.NS"]
    # }

    # Alternative data source IDs (Investing.com)
    INVESTING_COM_IDS = {
        "^NSEI": "17940", # Nifty 50
        "^NSEBANK": "17950", # Bank Nifty
        "NIFTY_FIN_SERVICE.NS": "17954" # Fin Nifty
    }

    # Market timing (IST)
    TIMEZONE = pytz.timezone("Asia/Kolkata")
    MARKET_START = "09:15"
    MARKET_END = "15:30"

    # Model Configuration
    CONFIDENCE_THRESHOLD = 0.51 # Lowered from 0.52
    FORCE_RETRAIN = False  # Set to True to force model retraining
    MIN_VOLUME_RATIO = 0.5   # Minimum volume ratio vs 20-day MA for signal consideration
    LOOKBACK_WINDOWS = [5, 10, 20, 50, 100] # Intraday-friendly windows (bars)
    PREDICTION_HORIZON = 12    # Predict movement over the next N bars (5m -> ~1 hour)
    PRICE_MOVEMENT_THRESHOLD = 0.0015 # 0.15% move threshold for intraday target


    # Behavior toggles
    RETRAIN_ON_FEATURE_MISMATCH = False  # If True, retrain when saved model expects features not present

    # Signal/logic parameters

    USE_EOD_SIGNALS = False         # If True, only act on closed daily candles (after market close)
    ADX_MIN = 12.0                  # Minimal ADX to accept a trend-following trade (more relaxed)
    USE_FALLBACK_TREND_SIGNALS = True  # If ML is neutral, use trend/RSI fallback to emit a signal
    # E-1: Removed AGREEMENT_STD_MAX, MIN_ATR_RATIO (orphaned, unused)

    # Confirmation/cooldown toggles
    CONFIRMATION_FILTERS = True     # Use RSI/SMA/MACD/VWAP confirmations
    ENABLE_VWAP_CONFIRMATION = True
    ENABLE_MACD_CONFIRMATION = True
    USE_BAR_CLOSE_ONLY = True       # Only act on closed bars (for 15m intraday)
    COOLDOWN_MINUTES = 30           # A-7/E-1: Cooldown minutes after trade close (replaces COOLDOWN_BARS)


    # Data fetching parameters
    YFINANCE_TIMEOUT = 10
    INVESTING_TIMEOUT = 15
    # Interval for OHLCV (e.g., '1d', '15m', '5m'). Switching to 5m intraday
    INTERVAL = "5m"
    DATA_PERIOD_TRAINING = "60d"   # For 5m, Yahoo supports ~60 days max
    DATA_PERIOD_SIGNAL = "30d"     # Signals use recent window to stay fast
    CACHE_FRESHNESS_SECONDS = 120 # Adjusted from 60s to 120s for realism
    MAX_FETCH_RETRIES = 5         # Retries for data fetching
    POLL_INTERVAL = 120           # Adjusted from 60s to 120s for realism
    MONITOR_INTERVAL = 15         # Seconds between FAST position-only checks
    EMERGENCY_SL_MULTIPLIER = 1.5 # If price exceeds SL by 1.5× risk, treat as emergency
    TRANSACTION_COST_PER_LOT = 150  # Estimated round-trip cost per lot (brokerage + STT + exchange)
    
    # ML Model Configuration
    TARGET_MODE = 'three_class'   # 'binary' or 'three_class'
    USE_ENHANCED_FEATURES = True
    SELECTED_FEATURES_ONLY = True  # Prune noise by using only high-impact features
    
    # ===== NEW: Enhanced Features Configuration =====
    # Risk Management
    USE_RISK_MANAGER = True
    # A-10: Load ACCOUNT_BALANCE from environment variable so balance can be changed without editing code
    ACCOUNT_BALANCE = float(os.getenv('ACCOUNT_BALANCE', '100000'))  # Initial capital
    RISK_PER_TRADE = 0.015      # 1.5% risk per trade
    MAX_DAILY_LOSS = 0.03       # 3% max daily loss
    
    # Performance Tracking
    USE_PERFORMANCE_TRACKER = True
    PERFORMANCE_DB_PATH = os.path.join(BASE_DIR, "trading_performance.db")
    
    # Regime Detection
    USE_REGIME_DETECTION = True
    REGIME_ADX_THRESHOLD = 25.0
    # B-1: Increased from 20 to 60. 60 bars at 5m = 300 min ≈ full intraday session ATR context
    REGIME_LOOKBACK = 60  # 60 bars at 5m = 300 min ≈ full intraday session ATR context
    
    # TradingView Integration
    USE_TRADINGVIEW_VALIDATION = True
    AUTO_OPEN_CHARTS = True
    CHART_ANALYSIS_DIR = os.path.join(BASE_DIR, "chart_analysis")
    MIN_VALIDATION_SCORE = 60  # Minimum manual validation score (0-100)
    
    # Simplified Confirmation Filters (reduced from 7 to 3)
    USE_SIMPLIFIED_FILTERS = True
    CORE_FILTERS = ['ml_confidence', 'trend_alignment', 'volatility_check']

# ================== Enhanced Data Fetcher ==================
class EnhancedDataFetcher:
    """Fetches historical data with multiple sources and caching."""
    def __init__(self):
        self.logger = logging.getLogger('DataFetcher')
        self.session = requests.Session()

        # Configure session with robust headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })

    def _fetch_yf_download(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Try fetching data using yfinance.download."""
        try:
            df = yf.download(
                tickers=symbol,
                period=period,
                interval=Config.INTERVAL,
                auto_adjust=False, # Keep Open, High, Low, Close, Adj Close
                progress=False,
                threads=False,
                timeout=Config.YFINANCE_TIMEOUT
            )
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                
                # Robustly flatten yfinance multi-index (often seen with index tickers)
                if isinstance(df.columns, pd.MultiIndex):
                    # Level 0 is typically Price type (Close, High, etc.), Level 1 is Ticker
                    df.columns = df.columns.get_level_values(0)
                
                # Standardize column names (remove spaces)
                df.columns = [str(col).replace(' ', '_') for col in df.columns]
                
                # Explicitly drop 'Adj Close' if present, as we use 'Close'
                if 'Adj_Close' in df.columns:
                     df = df.drop(columns=['Adj_Close'])
                
                # Final column selection ensuring OHLCV availability
                available_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                return df[available_cols] 
        except Exception as e:
            self.logger.debug(f"YFinance download failed for {symbol}: {type(e).__name__} - {e}")
        return None

    def _fetch_yf_direct(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Try fetching data using direct Yahoo Finance API call."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "range": period,
                "interval": Config.INTERVAL,
                "includePrePost": False
            }
            response = self.session.get(url, params=params, timeout=Config.YFINANCE_TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                    result = data['chart']['result'][0]
                    timestamps = result.get('timestamp', [])
                    quotes = result.get('indicators', {}).get('quote', [{}])[0]

                    if timestamps and quotes:
                        df = pd.DataFrame({
                            'Open': quotes.get('open', []),
                            'High': quotes.get('high', []),
                            'Low': quotes.get('low', []),
                            'Close': quotes.get('close', []),
                            'Volume': quotes.get('volume', [])
                        }, index=pd.to_datetime(timestamps, unit='s'))

                        # Remove rows with all NaN values that might appear
                        df = df.dropna(how='all')
                        # Ensure numeric types
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        return df[['Open', 'High', 'Low', 'Close', 'Volume']] # Standardize output columns
        except (requests.exceptions.RequestException, ValueError, TypeError) as e:
            self.logger.debug(f"Direct Yahoo API fail for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in _fetch_yf_direct for {symbol}: {type(e).__name__} - {e}")
        return None

    def _fetch_investing_com(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data from Investing.com."""
        try:
            investing_id = Config.INVESTING_COM_IDS.get(symbol)
            if not investing_id:
                self.logger.debug(f"No Investing.com ID for symbol {symbol}")
                return None

            # Investing.com needs specific headers and sometimes a cookie
            url = f"https://www.investing.com/indices/{investing_id}-historical-data"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            # Might need to visit the page first to get cookies, or simulate it
            # Simple get might work for table scraping
            response = self.session.get(url, headers=headers, timeout=Config.INVESTING_TIMEOUT)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find the main historical data table - ID might change, look for attributes
                table = soup.find('table', {'class': 'common-table historical-data-table'}) or \
                        soup.find('table', {'id': 'curr_table'}) # Fallback to older ID

                if table:
                    data = []
                    # Skip header row and potential "show more" footer
                    for row in table.select('tbody tr'):
                        cols = row.find_all('td')
                        if len(cols) >= 6:
                            try:
                                # Clean and convert data
                                date_str = cols[0].text.strip()
                                close_str = cols[1].text.replace(',', '').strip()
                                open_str = cols[2].text.replace(',', '').strip()
                                high_str = cols[3].text.replace(',', '').strip()
                                low_str = cols[4].text.replace(',', '').strip()
                                volume_str = cols[5].text.replace(',', '').strip() # Handle '-' for volume

                                # Basic validation and conversion
                                close = float(close_str) if close_str else np.nan
                                open_ = float(open_str) if open_str else np.nan
                                high = float(high_str) if high_str else np.nan
                                low = float(low_str) if low_str else np.nan
                                volume = float(volume_str) if volume_str and volume_str != '-' else 0

                                # Only append if we have valid price data
                                if not any(pd.isna([close, open_, high, low])):
                                    data.append({
                                        'Date': pd.to_datetime(date_str),
                                        'Close': close,
                                        'Open': open_,
                                        'High': high,
                                        'Low': low,
                                        'Volume': volume
                                    })
                            except (ValueError, AttributeError, pd.errors.DateParseError) as e:
                                self.logger.debug(f"Investing.com parsing error row: {e}")
                                continue # Skip row if parsing fails

                    if data:
                        df = pd.DataFrame(data)
                        df.set_index('Date', inplace=True)
                        df = df.sort_index() # Ensure chronological order
                        # Basic data cleaning - drop rows with missing essential price data
                        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
                        return df[['Open', 'High', 'Low', 'Close', 'Volume']] # Standardize output columns
            else:
                self.logger.debug(f"Investing.com request failed with status {response.status_code} for {symbol}")

        except Exception as e:
            self.logger.debug(f"Investing.com fetch failed for {symbol}: {type(e).__name__} - {e}")
        return None

    def _get_cache_filepath(self, symbol: str) -> str:
         """Generates a safe filename for the cache."""
         safe_symbol = symbol.replace('^', '').replace(':', '_').replace('.', '_').replace('/', '_')
         return os.path.join(Config.CACHE_DIR, f"{safe_symbol}_cache.pkl")

    def _save_cache(self, symbol: str, df: pd.DataFrame):
        """Save DataFrame to cache."""
        try:
            cache_file = self._get_cache_filepath(symbol)
            cache_data = {
                'timestamp': datetime.now(),
                'data': df
            }
            joblib.dump(cache_data, cache_file)
            self.logger.debug(f"Saved cache for {symbol}")
        except Exception as e:
            self.logger.warning(f"Cache save failed for {symbol}: {e}")

    def _load_cache(self, symbol: str, ignore_freshness: bool = False) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache. Returns None if not fresh, unless ignore_freshness is True."""
        try:
            cache_file = self._get_cache_filepath(symbol)
            if os.path.exists(cache_file):
                cache_data = joblib.load(cache_file)
                age = (datetime.now() - cache_data['timestamp']).total_seconds()
                if ignore_freshness or age < Config.CACHE_FRESHNESS_SECONDS:
                    self.logger.debug(f"Loaded cache for {symbol}, age {age:.1f}s")
                    return cache_data['data']
                else:
                    self.logger.debug(f"Cache for {symbol} is stale (age {age:.1f}s)")
        except Exception as e:
            self.logger.warning(f"Cache load failed for {symbol}: {e}")
        return None

    def fetch_data(self, symbol: str, period: str = Config.DATA_PERIOD_SIGNAL) -> pd.DataFrame:
        """Main fetch method with multiple fallbacks and caching."""
        if not symbol or not symbol.strip():
            self.logger.warning("Empty symbol passed to fetch_data.")
            return pd.DataFrame()

        # Try cache first
        df = self._load_cache(symbol, ignore_freshness=False)
        if df is not None and not df.empty and len(df) > 10:
            self.logger.info(f"Using fresh cached data for {symbol}")
            return df

        self.logger.info(f"Fetching fresh data for {symbol} ({period})...")

        fetch_methods = [
            self._fetch_yf_download,
            self._fetch_yf_direct,
            self._fetch_investing_com # Investing.com usually provides longer history
        ]

        fetched_df = None
        for attempt in range(Config.MAX_FETCH_RETRIES):
            for fetch_method in fetch_methods:
                self.logger.debug(f"Attempt {attempt+1}: Trying {fetch_method.__name__} for {symbol}")

                # For Investing.com, period is less relevant as it scrapes tables
                current_period = period if fetch_method != self._fetch_investing_com else None

                try:
                    df = fetch_method(symbol, current_period)
                    if df is not None and not df.empty:
                        # Check for minimum data points required for features + target
                        # If 1d data is requested (monitoring), relax requirement to 10 bars
                        # For signal generation, we still need full lookback windows
                        is_monitoring = (period == "1d")
                        min_data_points = 10 if is_monitoring else max(Config.LOOKBACK_WINDOWS) + Config.PREDICTION_HORIZON + 10
                        
                        if len(df) >= min_data_points:
                             if is_monitoring:
                                 self.logger.debug(f"Successfully fetched {len(df)} bars for monitor using {fetch_method.__name__}")
                             else:
                                 self.logger.info(f"Successfully fetched {len(df)} days for {symbol} using {fetch_method.__name__}")
                             self._save_cache(symbol, df)
                             return df
                        else:
                            self.logger.warning(f"Fetched only {len(df)} bars from {fetch_method.__name__}, insufficient for processing (need {min_data_points}).")
                            fetched_df = df # Keep the insufficient data just in case, maybe combine later? (Complex, skip for now)

                except Exception as e:
                    self.logger.debug(f"Fetch method {fetch_method.__name__} failed: {e}")
                    continue # Try next method

            if fetched_df is not None and not fetched_df.empty:
                 # If we got *some* data but not enough, log it and maybe try again?
                 # Or just break and use stale cache/empty? Let's break and try stale cache.
                 break

            if attempt < Config.MAX_FETCH_RETRIES - 1:
                sleep_time = 2 ** attempt
                self.logger.warning(f"All primary methods failed for {symbol}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time) # Exponential backoff between full cycles

        # If all attempts fail, do NOT use outdated cache for trading signals based on user feedback
        # Stale data is dangerous. Better to skip the signal than trade on old prices.
        self.logger.error(f"Failed to fetch fresh data for {symbol} after all attempts. Stale cache ignored.")
        return pd.DataFrame() # Return empty frame if everything fails

# ================== Feature Engineering ==================
class FeatureEngine:
    """Creates technical and price action features from historical data."""
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Creates features from the raw price/volume data."""

        # Minimum data length required for most TA indicators + lookback windows
        min_required_len = max(Config.LOOKBACK_WINDOWS) + 20 # Need history for MAs, etc.
        if df.empty or len(df) < min_required_len:
            logging.warning(f"Insufficient data ({len(df)} days) for feature creation. Requires at least {min_required_len}.")
            return pd.DataFrame()

        try:
            df = df.copy()
            
            # --- Add Baseline TA features ---
            try:
                df = df.sort_index()
                ta_df = add_all_ta_features(
                    df, open="Open", high="High",
                    low="Low", close="Close",
                    volume="Volume", fillna=False  # A-5: fillna=True fills NaN using future bars for early indicators — data leak
                )
                df = ta_df
            except Exception as e:
                logging.warning(f"TA library feature creation failed: {e}")

            # --- Add Custom Features ---
            df['range'] = df['High'] - df['Low']
            df['tr'] = np.maximum(df['High'] - df['Low'], np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            ))
            df['range_ratio'] = (df['range'] / df['Close'].shift(1)).fillna(0)
            
            # --- Enhanced Microstructure Features (The Edge) ---
            if Config.USE_ENHANCED_FEATURES and add_enhanced_features:
                df = add_enhanced_features(df)
            
            # --- Feature Selection (Noise Pruning) ---
            if Config.SELECTED_FEATURES_ONLY:
                # Curated list of high-impact features identified in Step 6
                SELECTED_FEATURES = [
                    # Trend & Momentum
                    'trend_adx', 'trend_ema_fast', 'trend_ema_slow', 'trend_macd_diff', 
                    'momentum_rsi', 'momentum_stoch_rsi', 'momentum_wr',
                    # Volatility
                    'atr', 'atr_ratio', 'bb_width', 'volatility_bbli',
                    # Volume
                    'volume_obv', 'volume_cmf', 'volume_surge_ratio',
                    # Microstructure (The new edge)
                    'vwap_deviation_pct', 'dist_from_high_pct', 'dist_from_low_pct',
                    'orb_position', 'volatility_squeeze', 'candle_strength',
                    'range_expansion', 'session_progress',
                    'range_ratio', 'intraday_momentum'
                ]
                # Filter to only those that exist
                existing_cols = [c for c in SELECTED_FEATURES if c in df.columns]
                # Also keep basic OHLC for safety in some logic
                for ohlc in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if ohlc not in existing_cols: existing_cols.append(ohlc)
                
                df = df[existing_cols]

            # A-1: Final cleanup — MUST be before return (was dead code previously)
            df = df.replace([np.inf, -np.inf], np.nan)

            # A-5: Forward-fill only; never backward-fill (bfill leaks future data)
            df = df.ffill()

            # A-5: Drop columns that are entirely NaN before dropping rows
            # If an indicator fails completely, it shouldn't kill all rows in the dataset
            all_nan_cols = [c for c in df.columns if df[c].isna().all()]
            if all_nan_cols:
                logging.warning(f"[FeatureEngine] Dropping all-NaN columns (data failed or insufficient history): {all_nan_cols}")
                df = df.drop(columns=all_nan_cols)

            # A-5: Drop leading rows where feature-fill is impossible (no past data)
            feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            if feature_cols:
                df = df.dropna(subset=feature_cols)
            
            logging.debug(f"[FeatureEngine] Rows after NaN drop: {len(df)}")

            # Prune constant features
            non_preserved_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            constant_cols = [col for col in non_preserved_cols if df[col].nunique() <= 1]
            df = df.drop(columns=constant_cols)

            # Update feature_cols to exclude dropped constant columns
            feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']]

            # A-1: Verify all NaNs cleared
            nan_count = df[feature_cols if feature_cols else df.columns].isna().sum().sum()
            logging.debug(f"[FeatureEngine] NaN count post-cleanup: {nan_count}")
            assert nan_count == 0, "NaN values remain after cleanup"

            logging.info(f"[FeatureEngine] Feature creation successful. Final DataFrame shape: {df.shape}")
            return df
        except AssertionError as ae:
            logging.error(f"[FeatureEngine] Assertion failed: {ae}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"[FeatureEngine] Overall feature creation failed: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return pd.DataFrame()

# ================== Meta Probability Estimator ==================
class PlattCalibratedClassifier:
    """
    Manual Platt scaling (sigmoid calibration) wrapper.
    Chains: meta_clf.predict_proba(X) -> sigmoid scaler -> calibrated probability.
    Bypasses CalibratedClassifierCV which has sklearn version issues.
    """
    def __init__(self, meta_clf, platt_scaler):
        self.meta_clf = meta_clf
        self.platt_scaler = platt_scaler

    def predict_proba(self, X):
        """Get calibrated probabilities: meta_clf -> platt sigmoid -> output."""
        raw_proba = self.meta_clf.predict_proba(X)
        if raw_proba.shape[1] == 2:
            # Binary case
            p_pos = self.platt_scaler.predict_proba(raw_proba[:, 1].reshape(-1, 1))[:, 1]
            return np.column_stack([1 - p_pos, p_pos])
        else:
            # Multiclass case: calibrating each class OVR (simplified)
            # For 3-class, we return the meta_clf probs directly if platt is complex,
            # or apply platt to each if we trained OVR calibrators.
            # For now, return meta_clf probs for multiclass to avoid complexity.
            return raw_proba

# ================== Model Training ==================
class ModelTrainer:
    """Trains and calibrates the ensemble model."""
    def __init__(self):
        self.logger = logging.getLogger('ModelTrainer')
        self.feature_columns = None
        self.base_models = {} # Initialize here to check availability

        # Initialize base models based on imports
        is_multiclass = Config.TARGET_MODE == 'three_class'
        objective = "multiclass" if is_multiclass else "binary"
        
        if LGBMClassifier:
             self.base_models["lgb"] = LGBMClassifier(
                 n_estimators=500, learning_rate=0.01, num_leaves=32, subsample=0.8,
                 colsample_bytree=0.8, objective=objective, n_jobs=-1, random_state=42, verbosity=-1
             )
        if XGBClassifier:
             self.base_models["xgb"] = XGBClassifier(
                 n_estimators=500, learning_rate=0.01, max_depth=6, subsample=0.8,
                 colsample_bytree=0.8, objective="multi:softprob" if is_multiclass else "binary:logistic",
                 eval_metric="mlogloss" if is_multiclass else "logloss",
                 n_jobs=-1, random_state=42, verbosity=0
             )
        if CatBoostClassifier:
             self.base_models["cat"] = CatBoostClassifier(
                 iterations=500, learning_rate=0.01, depth=6, 
                 loss_function="MultiClass" if is_multiclass else "Logloss",
                 verbose=False, random_seed=42
             )

        # Fallback to Logistic Regression if no GB models are available
        if not self.base_models:
            self.logger.warning("No Gradient Boosting libraries found. Using Logistic Regression as base model.")
            self.base_models["lr_base"] = Pipeline([
                 ('scaler', RobustScaler()), # Scale LR inputs
                 ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
             ])


    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels from the data."""
        try:
            # Minimum data required for training (after feature engineering and target creation)
            min_training_len = max(Config.LOOKBACK_WINDOWS) + Config.PREDICTION_HORIZON + 20 # Needs history for TA + target shift
            if df.empty or len(df) < min_training_len:
                self.logger.warning(f"Insufficient data ({len(df)} days) for training after feature creation.")
                return pd.DataFrame(), pd.Series()

            # Ensure 'Close' column exists for target creation
            if 'Close' not in df.columns:
                 self.logger.error("'Close' column not found for target creation.")
                 return pd.DataFrame(), pd.Series()

            # Forward returns for labels
            horizon = Config.PREDICTION_HORIZON
            threshold = Config.PRICE_MOVEMENT_THRESHOLD

            if 'target' not in df.columns:
                 self.logger.error("'target' column not found in DataFrame passed to prepare_data.")
                 return pd.DataFrame(), pd.Series()

            # Drop any remaining NaNs in target column (now handled by target calculation, but safe to keep)
            df = df.dropna(subset=["target"])

            if len(df) < 30: # Minimum samples for reasonable split/training
                 self.logger.warning(f"Insufficient data ({len(df)} samples) after dropping target NaNs for training.")
                 return pd.DataFrame(), pd.Series()

            # Select features - exclude target and any remaining original columns if they weren't dropped
            exclude_cols_final = ["target", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
            feature_cols = [col for col in df.columns if col not in exclude_cols_final]

            # Remove constant features again just in case fillna created some
            feature_cols = [col for col in feature_cols if df[col].nunique() > 1]

            if not feature_cols:
                self.logger.error("No valid features remaining after preparation.")
                return pd.DataFrame(), pd.Series()

            X = df[feature_cols]
            y = df["target"]
            
            # Ensure target is integer for multiclass
            if Config.TARGET_MODE == 'three_class':
                y = y.astype(int)

            # Final check for NaNs/Infinities in features just before training
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isna().any().any():
                 self.logger.warning(f"NaNs found in features before training. Imputing...")
                 # A-2: Forward fill only — backward fill (bfill) propagates future values into past rows (data leakage)
                 # REMOVED: bfill() caused data leakage — backward fill propagates
                 # future values into past rows during model training
                 X = X.ffill()       # forward-fill only — uses only past data
                 X = X.dropna()      # drop rows that cannot be forward-filled

            self.feature_columns = feature_cols
            self.logger.info(f"Prepared data for training: {len(X)} samples, {len(feature_cols)} features. Target distribution:\n{y.value_counts()}")

            # Check class balance
            class_counts = y.value_counts(normalize=True)
            self.logger.info(f"Class distribution: {class_counts.to_dict()}")
            if len(class_counts) < 2:
                 self.logger.warning("Only one class found in target data!")

            return X, y

        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            return pd.DataFrame(), pd.Series()

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Trains the base models, meta-classifier, and calibrator using OOF predictions."""
        try:
            if X.empty or y.empty or len(X) < 50: # Need enough data for CV
                self.logger.warning("Insufficient data for model training.")
                return {}

            # Define TimeSeriesSplit
            n_splits = min(5, max(2, len(X) // 50)) # Use at least 2 splits, max 5, proportional to data size
            cv = TimeSeriesSplit(n_splits=n_splits)
            self.logger.info(f"Training with {n_splits} TimeSeries splits.")

            cv_scores = []
            # Store OOF predictions and corresponding true labels for meta-training and calibration
            is_multiclass = Config.TARGET_MODE == 'three_class'
            num_classes_per_model = 3 if is_multiclass else 1
            meta_feature_cols = len(self.base_models) * num_classes_per_model
            
            oof_meta_features = np.zeros((len(X), meta_feature_cols))
            oof_labels = y.copy() # Will remove rows later if no predictions were made

            # Train base models on CV folds and collect OOF predictions
            oof_indices = [] # Track indices for which OOF predictions are made

            for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
                try:
                    self.logger.info(f"Training Fold {fold+1}/{n_splits}")
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    # Skip fold if validation set is too small or has only one class
                    if len(y_val) < 20 or len(y_val.unique()) < 2:
                         self.logger.warning(f"Skipping Fold {fold+1} due to insufficient validation data or single class.")
                         continue

                    fold_meta_val = np.zeros((len(val_idx), meta_feature_cols))

                    for i, (name, model) in enumerate(self.base_models.items()):
                        try:
                            import copy
                            fold_model = copy.deepcopy(model)

                            # Check if it's a simple GB model or a pipeline before fitting
                            if isinstance(fold_model, (LGBMClassifier, XGBClassifier, CatBoostClassifier)):
                                # Use early stopping if validation set is large enough
                                if len(X_val) > 100: # Arbitrary threshold for early stopping
                                    eval_set = [(X_val, y_val)]
                                    if isinstance(fold_model, LGBMClassifier):
                                        try:
                                            from lightgbm import early_stopping
                                            fold_model.fit(X_train, y_train, eval_set=eval_set, callbacks=[early_stopping(50, verbose=False)])
                                        except ImportError:
                                            fold_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=False) # Fallback for old lightgbm
                                    elif isinstance(fold_model, XGBClassifier):
                                        fold_model.set_params(early_stopping_rounds=50)
                                        fold_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                                    elif isinstance(fold_model, CatBoostClassifier):
                                        fold_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
                                else:
                                    fold_model.fit(X_train, y_train)

                            elif isinstance(fold_model, Pipeline):
                                 fold_model.fit(X_train, y_train)
                            else: # Generic case
                                 fold_model.fit(X_train, y_train)

                            # Store predictions for the meta-classifier training (OOF predictions)
                            if hasattr(fold_model, 'predict_proba'):
                                probs = fold_model.predict_proba(X_val)
                                if is_multiclass:
                                    for c_idx in range(3):
                                        fold_meta_val[:, i * 3 + c_idx] = probs[:, c_idx]
                                else:
                                    fold_meta_val[:, i] = probs[:, 1]
                            else:
                                if is_multiclass:
                                    pred = fold_model.predict(X_val)
                                    for row_idx, p_val in enumerate(pred):
                                        fold_meta_val[row_idx, i * 3 + int(p_val)] = 1.0
                                else:
                                    fold_meta_val[:, i] = fold_model.predict(X_val)

                        except Exception as e:
                            self.logger.warning(f"Base model '{name}' failed in Fold {fold+1}: {type(e).__name__} - {e}")
                            # Fill failed predictions with 0.5 (random guess)
                            fold_meta_val[:, i] = 0.5

                    # Append OOF predictions and indices
                    oof_meta_features[val_idx] = fold_meta_val
                    oof_indices.extend(val_idx)
                    temp_meta_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
                    temp_meta_clf.fit(fold_meta_val, y_val)
                    fold_val_pred_proba = temp_meta_clf.predict_proba(fold_meta_val)

                    # Calculate metrics for the fold
                    try:
                        if is_multiclass:
                            auc = roc_auc_score(y_val, fold_val_pred_proba, multi_class='ovr')
                            mc_pred = np.argmax(fold_val_pred_proba, axis=1)
                            precision = precision_score(y_val, mc_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_val, mc_pred, average='weighted', zero_division=0)
                        else:
                            binary_probs = fold_val_pred_proba[:, 1]
                            auc = roc_auc_score(y_val, binary_probs)
                            binary_pred = (binary_probs > Config.CONFIDENCE_THRESHOLD).astype(int)
                            precision = precision_score(y_val, binary_pred, zero_division=0)
                            recall = recall_score(y_val, binary_pred, zero_division=0)

                        self.logger.info(f"Fold {fold+1} Metrics: AUC={auc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
                        cv_scores.append({
                                                       'fold': fold+1,
                            'auc': auc,
                            'precision': precision,
                            'recall': recall,
                            'num_samples': len(y_val)
                        })
                    except Exception as e:
                        self.logger.warning(f"Metric calculation failed for Fold {fold+1}: {e}")

                except Exception as e:
                    self.logger.error(f"An error occurred during Fold {fold+1} training: {type(e).__name__} - {e}")
            X_indexed = X.copy() # Use index for reliable joining
            y_indexed = y.copy()

            # Create OOF DataFrame directly from the first loop's oof_meta_features
            is_multiclass = Config.TARGET_MODE == 'three_class'
            
            col_names = []
            for name in self.base_models.keys():
                if is_multiclass:
                    for c_idx in range(3):
                        col_names.append(f"{name}_prob_{c_idx}")
                else:
                    col_names.append(name)
                    
            # Replace 0s with NaNs for rows that were never in val_idx
            valid_oof_mask = np.zeros(len(X), dtype=bool)
            valid_oof_mask[oof_indices] = True
            oof_meta_features_clean = np.where(valid_oof_mask[:, None], oof_meta_features, np.nan)
            
            # Combine OOF predictions into a DataFrame
            cv_scores_recalc = []
            oof_meta_df = pd.DataFrame(oof_meta_features_clean, index=X_indexed.index, columns=col_names)

            # Align OOF predictions with true labels, drop rows where any base model failed to predict OOF
            oof_meta_df = oof_meta_df.dropna()
            oof_labels_aligned = y_indexed.loc[oof_meta_df.index]

            if oof_meta_df.empty or len(oof_meta_df) < 50 or len(oof_labels_aligned.unique()) < 2:
                 self.logger.error("Insufficient OOF data generated for meta-training or calibration.")
                 return {}

            self.logger.info(f"Generated {len(oof_meta_df)} OOF samples for meta-training.")

            # --- Train the Meta-Classifier on OOF predictions ---
            meta_clf = Pipeline([
                 ('scaler', RobustScaler()),
                 ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
             ])

            try:
                meta_clf.fit(oof_meta_df, oof_labels_aligned)
                self.logger.info("Meta-classifier trained successfully on OOF predictions.")
            except Exception as e:
                self.logger.error(f"Meta-classifier training failed: {type(e).__name__} - {e}")
                return {}

            # --- Probability Calibration via Manual Platt Scaling ---
            # Bypasses CalibratedClassifierCV (has sklearn version issues)
            try:
                # Get raw meta-classifier probabilities on OOF data
                # For Platt calibration in multiclass, we'd need OVR calibrators.
                # To keep it simple, we only calibrate the "Bullish" prob if binary.
                if is_multiclass:
                    calibrator = meta_clf
                    calibrated_oof_preds = meta_clf.predict_proba(oof_meta_df)
                    self.logger.info("Multiclass meta-classifier trained. Calibration skipped (OVR complexity).")
                else:
                    oof_meta_preds_proba = meta_clf.predict_proba(oof_meta_df)[:, 1]
                    platt_scaler = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
                    platt_scaler.fit(oof_meta_preds_proba.reshape(-1, 1), oof_labels_aligned)
                    calibrator = PlattCalibratedClassifier(meta_clf, platt_scaler)
                    calibrated_oof_preds = platt_scaler.predict_proba(oof_meta_preds_proba.reshape(-1, 1))
                    self.logger.info("Binary probability calibrator trained successfully.")

                try:
                    is_multiclass = Config.TARGET_MODE == 'three_class'
                    if is_multiclass:
                        cal_auc = roc_auc_score(oof_labels_aligned, calibrated_oof_preds, multi_class='ovr')
                        cal_binary_pred = np.argmax(calibrated_oof_preds, axis=1)
                        cal_precision = precision_score(oof_labels_aligned, cal_binary_pred, average='weighted', zero_division=0)
                        cal_recall = recall_score(oof_labels_aligned, cal_binary_pred, average='weighted', zero_division=0)
                    else:
                        cal_auc = roc_auc_score(oof_labels_aligned, calibrated_oof_preds[:, 1])
                        cal_binary_pred = (calibrated_oof_preds[:, 1] > Config.CONFIDENCE_THRESHOLD).astype(int)
                        cal_precision = precision_score(oof_labels_aligned, cal_binary_pred, zero_division=0)
                        cal_recall = recall_score(oof_labels_aligned, cal_binary_pred, zero_division=0)

                    self.logger.info(f"Calibrated OOF Metrics (Meta+Calibrator): AUC={cal_auc:.3f}, Precision={cal_precision:.3f}, Recall={cal_recall:.3f}")

                    cv_scores_recalc.append({
                         'fold': 'Overall_OOF',
                         'auc': cal_auc,
                         'precision': cal_precision,
                         'recall': cal_recall,
                         'num_samples': len(oof_meta_df)
                    })

                except Exception as e:
                    self.logger.warning(f"Calibrated OOF metric calculation failed: {e}")

            except Exception as e:
                self.logger.error(f"Platt calibration failed: {type(e).__name__} - {e}")
                calibrator = meta_clf
                self.logger.warning("Using uncalibrated meta-classifier as fallback.")


            # --- Train Final Base Models on the entire dataset ---
            # Fix: Introduce synthetic 'eval_set' for Early Stopping on final base models
            final_base_models = {}
            
            # Split the last 10% of X and y sequentially as the hold-out eval set
            eval_size = int(len(X) * 0.10)
            X_train_final = X.iloc[:-eval_size]
            y_train_final = y.iloc[:-eval_size]
            X_eval_final = X.iloc[-eval_size:]
            y_eval_final = y.iloc[-eval_size:]

            for name, model in self.base_models.items():
                 try:
                    self.logger.info(f"Training final base model '{name}' on full dataset with validation.")
                    import copy
                    final_model = copy.deepcopy(model) # Clone for final training

                    if hasattr(final_model, 'fit'):
                        if isinstance(final_model, (LGBMClassifier, XGBClassifier, CatBoostClassifier)):
                            eval_set = [(X_eval_final, y_eval_final)]
                            if isinstance(final_model, LGBMClassifier):
                                try:
                                    from lightgbm import early_stopping
                                    final_model.fit(X_train_final, y_train_final, eval_set=eval_set, callbacks=[early_stopping(50, verbose=False)])
                                except ImportError:
                                    final_model.fit(X_train_final, y_train_final, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
                            elif isinstance(final_model, XGBClassifier):
                                final_model.set_params(early_stopping_rounds=50)
                                final_model.fit(X_train_final, y_train_final, eval_set=eval_set, verbose=False)
                            elif isinstance(final_model, CatBoostClassifier):
                                final_model.fit(X_train_final, y_train_final, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
                        else:
                             # For models like LogisticRegression pipeline
                             final_model.fit(X, y)
                        
                        final_base_models[name] = final_model
                        self.logger.info(f"Final base model '{name}' trained.")
                    else:
                         self.logger.warning(f"Model '{name}' is not trainable for final fit.")

                 except Exception as e:
                    self.logger.error(f"Final training failed for base model '{name}': {type(e).__name__} - {e}")
                    # Do not include this model in the final model pack if training failed


            if not final_base_models:
                 self.logger.critical("No base models trained successfully on the full dataset. Cannot create final model pack.")
                 return {}

            # Fix: Track the exact feature sequence (names) for model alignment
            ordered_base_names = list(final_base_models.keys())

            # Final model package
            model_pack = {
                'base_models': final_base_models, # Base models trained on full data
                'ordered_base_names': ordered_base_names, # To preserve column alignment in meta_features
                'meta_clf': meta_clf,             # Meta-classifier trained on OOF
                'calibrator': calibrator,         # Calibrator trained on meta_clf OOF output
                'features': self.feature_columns, # List of feature names
                'cv_scores': cv_scores_recalc     # CV scores from OOF validation
            }

            self.logger.info(f"Model training complete. Ensemble includes {len(final_base_models)} base models.")
            return model_pack

        except Exception as e:
            self.logger.error(f"Overall model training failed: {str(e)}")
            return {}

# ================== Trading Bot ==================
class TradingBot:
    """Orchestrates data fetching, feature engineering, modeling, and signal generation."""
    def __init__(self):
        self.logger = logging.getLogger('TradingBot')
        self.data_fetcher = EnhancedDataFetcher()
        self.feature_engine = FeatureEngine()
        self.model_trainer = ModelTrainer() # Use the trainer instance for base model definitions
        self.models = {} # Stores loaded/trained model packs per index
        self.error_counts = {} # Track consecutive errors per symbol
        self.last_eod_scan_date = None  # Track last EOD scan to avoid duplicates
        self.last_fallback_signal = {}  # Track last fallback signal time per index
        self.last_signal_bar = {}       # Track last emitted signal bar per index (cooldown)
        
        # ===== NEW: Initialize enhanced modules =====
        # State Management
        self.state = TradingState()
        self.logger.info("Trading State initialized.")

        # Risk Manager
        if Config.USE_RISK_MANAGER and RiskManager is not None:
            self.risk_manager = RiskManager(
                account_balance=Config.ACCOUNT_BALANCE
            )
            # A-9/C-2: Sync RiskManager with persistent state from TradingState
            self.risk_manager.daily_trades = self.state.state.get("daily_trades", 0)
            self.risk_manager.daily_pnl = self.state.state.get("daily_pnl", 0.0)
            self.risk_manager.account_balance = Config.ACCOUNT_BALANCE + self.risk_manager.daily_pnl
            
            self.logger.info(f"Risk Manager initialized with balance: ₹{Config.ACCOUNT_BALANCE:,.2f} (Daily PnL: ₹{self.risk_manager.daily_pnl:,.2f}, Trades: {self.risk_manager.daily_trades})")
        else:
            self.risk_manager = None
            if Config.USE_RISK_MANAGER:
                self.logger.warning("Risk Manager not available. Risk management features disabled.")
        
        # Performance Tracker
        if Config.USE_PERFORMANCE_TRACKER and PerformanceTracker is not None:
            self.performance_tracker = PerformanceTracker(
                db_path=Config.PERFORMANCE_DB_PATH
            )
            self.logger.info(f"Performance Tracker initialized: {Config.PERFORMANCE_DB_PATH}")
        else:
            self.performance_tracker = None
            if Config.USE_PERFORMANCE_TRACKER:
                self.logger.warning("Performance Tracker not available. Trade logging disabled.")
        
        # Regime Detector
        if Config.USE_REGIME_DETECTION and RegimeDetector is not None:
            self.regime_detector = RegimeDetector(
                adx_threshold=Config.REGIME_ADX_THRESHOLD,
                lookback=Config.REGIME_LOOKBACK
            )
            self.logger.info("Regime Detector initialized")
        else:
            self.regime_detector = None
            if Config.USE_REGIME_DETECTION:
                self.logger.warning("Regime Detector not available. Using default parameters.")
        
        # TradingView Validator
        if Config.USE_TRADINGVIEW_VALIDATION and TradingViewValidator is not None:
            self.tradingview_validator = TradingViewValidator(
                chart_dir=Config.CHART_ANALYSIS_DIR,
                auto_open=Config.AUTO_OPEN_CHARTS
            )
            self.logger.info(f"TradingView Validator initialized (auto_open={Config.AUTO_OPEN_CHARTS})")
        else:
            self.tradingview_validator = None
            if Config.USE_TRADINGVIEW_VALIDATION:
                self.logger.warning("TradingView Validator not available. Chart validation disabled.")

        # Signal Validator
        if SignalValidator is not None:
             self.signal_validator = SignalValidator(config=getattr(Config, 'SCORING_CONFIG', {}))
             self.logger.info("Signal Validator initialized.")
        else:
             self.signal_validator = None
             self.logger.warning("Signal Validator not available.")

        # State is now initialized earlier (before RiskManager)

    def get_dynamic_thresholds(self, regime: str) -> Dict:
        """Get confidence thresholds based on market regime."""
        if self.regime_detector:
            return self.regime_detector.get_regime_parameters(regime)
        return {
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
            'adx_min': Config.ADX_MIN,
            'num_filters': 3,
            'description': 'Default (No Regime Detector)'
        }

    @staticmethod
    def get_session_phase(now_ist: 'pd.Timestamp') -> str:
        """B-3: Classify the current time into an intraday trading session phase.

        Returns one of: 'pre_open', 'opening', 'mid_session', 'afternoon', 'power_hour'
        """
        minutes = now_ist.hour * 60 + now_ist.minute
        if minutes < 570:   return 'pre_open'    # before 09:30
        if minutes < 630:   return 'opening'     # 09:30–10:30
        if minutes < 750:   return 'mid_session' # 10:30–12:30
        if minutes < 840:   return 'afternoon'   # 12:30–14:00
        return 'power_hour'                       # 14:00–15:10

    # B-3: Phase-specific ADX and validator boost config
    PHASE_CONFIG = {
        'opening':     {'min_adx': 18, 'validator_boost': -5},
        'mid_session': {'min_adx': 12, 'validator_boost':  0},
        'afternoon':   {'min_adx': 12, 'validator_boost':  0},
        'power_hour':  {'min_adx': 20, 'validator_boost': +5},
    }

    def _calculate_target(self, df: pd.DataFrame) -> pd.Series:
        """Calculates improved target labeling (triple-barrier method)."""
        if df.empty or len(df) < Config.LOOKBACK_WINDOWS[-1] + 20:
             return pd.Series()

        try:
            if calculate_improved_targets:
                # Use the new high-quality labeling logic
                return calculate_improved_targets(
                    df, 
                    atr_tp_mult=1.5, 
                    atr_sl_mult=1.0, 
                    horizon=Config.PREDICTION_HORIZON,
                    mode=Config.TARGET_MODE
                )
            else:
                # A-8: Fallback three-class labeling (was binary [0,1] — mismatch with model expecting {0,1,2})
                # 0=BEAR, 1=BULL, 2=NEUTRAL  (matches improved_targets convention)
                horizon = Config.PREDICTION_HORIZON
                threshold = Config.PRICE_MOVEMENT_THRESHOLD
                future_return = df['Close'].shift(-horizon).pct_change(horizon)
                up   = future_return >  threshold
                down = future_return < -threshold
                target = np.select([down, up], [0, 1], default=2)
                target = pd.Series(target, index=df.index, name='target', dtype=float)
                target[future_return.isna()] = np.nan  # NaN for rows with no future data

                # Sanity check
                assert set(target.dropna().unique()).issubset({0.0, 1.0, 2.0}), \
                    "Target labels outside expected {0,1,2}"

                label_dist = target.dropna().value_counts(normalize=True).to_dict()
                self.logger.info(
                    f"[Labels] BEAR:{label_dist.get(0.0, 0):.1%} "
                    f"BULL:{label_dist.get(1.0, 0):.1%} "
                    f"NEUTRAL:{label_dist.get(2.0, 0):.1%}"
                )
                return target

        except Exception as e:
            self.logger.error(f"Target calculation failed: {str(e)}")
            return pd.Series()


    def load_or_train_models(self):
        """Load existing models or train new ones for each index."""
        self.logger.info("Attempting to load or train models...")
        for name, symbol in Config.INDICES.items():
            model_path = os.path.join(Config.MODEL_DIR, f"{name}_model.joblib")

            if Config.FORCE_RETRAIN or not os.path.exists(model_path):
                if Config.FORCE_RETRAIN and os.path.exists(model_path):
                    self.logger.info(f"FORCE_RETRAIN enabled. Deleting existing model for {name} and retraining.")
                    try:
                        os.remove(model_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {model_path}: {e}")
                self._train_new_model(name, symbol, model_path)
            else:
                try:
                    self.logger.info(f"Loading existing model for {name} from {model_path}")
                    model_pack = joblib.load(model_path)

                    if isinstance(model_pack, dict) and \
                       'base_models' in model_pack and isinstance(model_pack['base_models'], dict) and \
                       'meta_clf' in model_pack and \
                       'calibrator' in model_pack and \
                       'features' in model_pack and isinstance(model_pack['features'], list):

                        self.models[name] = model_pack
                        self.logger.info(f"Successfully loaded model for {name} with {len(model_pack['base_models'])} base models and {len(model_pack['features'])} features.")
                        if 'cv_scores' in model_pack:
                             avg_auc = np.mean([score['auc'] for score in model_pack['cv_scores'] if isinstance(score['auc'], (int, float))]) if model_pack['cv_scores'] else np.nan
                             self.logger.info(f"Loaded Model {name} CV Info: {len(model_pack['cv_scores'])} folds, Avg AUC: {avg_auc:.3f}")
                        else:
                             self.logger.warning(f"Loaded model for {name} has no CV scores.")
                    else:
                        self.logger.error(f"Loaded model file {model_path} has an invalid structure. Retraining.")
                        self._train_new_model(name, symbol, model_path)

                except Exception as e:
                    self.logger.error(f"Failed to load model for {name} from {model_path}: {type(e).__name__} - {e}")
                    self._train_new_model(name, symbol, model_path)

        if not self.models:
             self.logger.critical("No models were loaded or trained successfully. Bot cannot generate signals.")

    def _train_new_model(self, name: str, symbol: str, model_path: str):
        """Train a new model for the given symbol."""
        try:
            self.logger.info(f"Starting training process for {name} ({symbol})...")

            # 1. Fetch data
            self.logger.info(f"Fetching training data for {name}...")
            raw_df = self.data_fetcher.fetch_data(symbol, period=Config.DATA_PERIOD_TRAINING)
            if raw_df.empty or len(raw_df) < Config.LOOKBACK_WINDOWS[-1] + Config.PREDICTION_HORIZON + 30: # Sufficient data check
                self.logger.error(f"Insufficient data ({len(raw_df)} days) to train model for {name}.")
                return

            # 2. Calculate Target
            self.logger.info(f"Calculating target variable for {name}...")
            target_series = self._calculate_target(raw_df.copy()) # Work on a copy to avoid modifying raw_df
            if target_series.empty or target_series.dropna().empty:
                 self.logger.error(f"Target calculation failed or resulted in no valid targets for {name}.")
                 return

            # 3. Create Features (now on the raw_df, keeping original columns)
            self.logger.info(f"Creating features for {name}...")
            # Pass raw_df, FeatureEngine should return raw_df + features
            feature_df = self.feature_engine.create_features(raw_df.copy()) # Work on a copy

            if feature_df.empty or len(feature_df) < len(raw_df) * 0.8:
                 self.logger.error(f"Feature creation failed or lost significant data for {name}. Original: {len(raw_df)}, Features: {len(feature_df)}")
                 return
            feature_df.index = pd.to_datetime(feature_df.index)
            target_series.index = pd.to_datetime(target_series.index)

            # Combine features and target - outer join to keep all dates initially
            merged_df = feature_df.join(target_series, how='inner') # Use inner join to align features and target

            # Drop rows where target is NaN (these cannot be used for training)
            merged_df.dropna(subset=['target'], inplace=True)

            if merged_df.empty or len(merged_df) < self.model_trainer.base_models.__len__() * 30: # Need enough samples per model + CV
                 self.logger.error(f"Insufficient data ({len(merged_df)} samples) after merging features and target for {name}.")
                 return

            # 5. Prepare data for ModelTrainer (split X, y, handle final feature selection/scaling if needed by trainer)
            self.logger.info(f"Preparing final data for model trainer for {name}...")
            X, y = self.model_trainer.prepare_data(merged_df.copy()) # Pass merged_df to trainer

            if X.empty or y.empty:
                 self.logger.error(f"Final data preparation failed for {name}.")
                 return

            # 6. Train models
            self.logger.info(f"Initiating model training for {name}...")
            model_pack = self.model_trainer.train_models(X, y)

            if model_pack:
                self.models[name] = model_pack
                joblib.dump(model_pack, model_path)
                self.logger.info(f"Successfully trained and saved model for {name} at {model_path}")

                # Log CV scores if available
                if 'cv_scores' in model_pack and model_pack['cv_scores']:
                    fold_aucs = [score['auc'] for score in model_pack['cv_scores'] 
                                 if isinstance(score.get('auc'), (int, float)) and score.get('fold') != 'Overall_OOF']
                    avg_auc_overall = np.mean(fold_aucs) if fold_aucs else np.nan
                    overall_oof_auc = next((score['auc'] for score in model_pack['cv_scores'] if score.get('fold') == 'Overall_OOF'), np.nan)

                    self.logger.info(f"Training complete for {name}. Avg CV AUC: {avg_auc_overall:.3f}, Overall OOF AUC (Calibrated): {overall_oof_auc:.3f}")
            else:
                self.logger.error(f"Model training failed for {name}. No model saved.")

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during model training process for {name}: {str(e)}")

    def generate_signal(self, name: str, symbol: str) -> Optional[Dict]:
        """
        Generate trading signal with enhanced features:
        - Regime detection and adaptive parameters
        - Multi-factor validation (Trend, Momentum, Volume, Volatility)
        - Risk management integration
        - TradingView chart validation
        """
        try:
            # Fix 4: Dynamic time-based filtering (replaces rigid 14:30 cutoff)
            # After 3:00 PM, require higher conviction (less time for trade to develop)
            # After 3:10 PM, block new entries (only 20 min left, too risky)
            now = datetime.now(Config.TIMEZONE)
            current_hour = now.hour
            current_minute = now.minute
            
            # Calculate minutes until market close (15:30 IST)
            minutes_to_close = (15 * 60 + 30) - (current_hour * 60 + current_minute)
            
            # After 3:10 PM, hard block (only 20 min remaining)
            if minutes_to_close < 20:
                self.logger.info(f"Entry cutoff: Too late for {name} (3:10 PM cutoff, {minutes_to_close} min to close).")
                return {
                    'index': name, 'position_size': 0,
                    'validation_reason': "Too Late (3:10 PM cutoff)",
                    'price': 0, 'regime': 'N/A'
                }
            
            # Store late_entry flag for later use in validation
            late_entry = minutes_to_close < 60  # After 2:30 PM
            
            # Expiry Day Awareness
            # On expiry days, gamma risk and pin risk are extreme.
            # Reduce position size, tighten targets, and raise validator threshold.
            expiry_adj = None
            if get_expiry_adjustment is not None:
                try:
                    expiry_adj = get_expiry_adjustment(name)
                    if expiry_adj.get('should_skip', False):
                        self.logger.info(f"Expiry day skip: Trading disabled for {name} on expiry.")
                        return {
                            'index': name, 'position_size': 0, 
                            'validation_reason': "Expiry Day Skip", 
                            'price': 0, 'regime': 'N/A'
                        }
                    if expiry_adj.get('is_expiry_day', False):
                        self.logger.info(f"⚠️ EXPIRY DAY for {name}. Applying tighter params.")
                    elif expiry_adj.get('is_expiry_week', False):
                        self.logger.info(f"📅 Expiry week for {name}. Applying mild adjustments.")
                except Exception as e:
                    self.logger.debug(f"Expiry check failed: {e}")
                    expiry_adj = None
            
            # B-3: Session Phase Awareness
            now_ist = datetime.now(Config.TIMEZONE)
            session_phase = self.get_session_phase(now_ist)
            phase_cfg = self.PHASE_CONFIG.get(session_phase, {'min_adx': 12, 'validator_boost': 0})
            self.logger.info(f"[Signal] [{name}] session_phase={session_phase} — phase_adx_min={phase_cfg['min_adx']}")

            # B-2: Days-to-Expiry (DTE) Awareness
            dte = -1
            expiry_date = None
            if get_days_to_expiry is not None:
                try:
                    dte = get_days_to_expiry(name, now_ist.date())
                    self.logger.info(f"[Signal] [{name}] DTE={dte}")
                except Exception as e:
                    self.logger.debug(f"DTE calculation failed for {name}: {e}")

            # DTE-based gates (reduce risk near expiry due to gamma/pin risk)
            min_conviction_boost = 0.0
            if dte == 0:      # Expiry day: require higher conviction
                min_conviction_boost = 0.05
                self.logger.info(f"[Signal] [{name}] Expiry day — min_conviction boosted by {min_conviction_boost}")
            elif dte == 1:    # Day before expiry: mild boost
                min_conviction_boost = 0.02

            # Fix 9: Cross-index correlation limit
            # Prevent concentrated same-direction bets across correlated indices
            open_positions = self.state.state.get('open_positions', {})
            if len(open_positions) >= 2:
                open_directions = [p.get('direction') for p in open_positions.values()]
                # Will check after direction is determined (below)
            
            model_pack = self.models.get(name)
            if not model_pack:
                self.logger.warning(f"No trained model available for {name}. Cannot generate signal.")
                return None

            # 1. Fetch latest data
            fetch_period = Config.DATA_PERIOD_SIGNAL
            self.logger.debug(f"Fetching data for signal generation for {name} ({symbol})")
            raw_df = self.data_fetcher.fetch_data(symbol, period=fetch_period)

            if raw_df.empty or len(raw_df) < max(Config.LOOKBACK_WINDOWS) + 10:
                self.logger.warning(f"Insufficient data ({len(raw_df)} bars) for {name}. Skipping signal.")
                self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
                return {
                    'index': name, 'position_size': 0, 
                    'validation_reason': "Insufficient Data", 
                    'price': 0, 'regime': 'N/A'
                }

            # Reset error count on successful fetch
            self.error_counts[symbol] = 0

            # 2. Detect Market Regime
            regime = 'unknown'
            regime_params = {}
            if self.regime_detector:
                regime_result = self.regime_detector.get_regime_with_params(raw_df)
                regime = regime_result['regime']
                regime_params = regime_result['parameters']
                self.logger.info(f"Detected regime for {name}: {regime} - {regime_params['description']}")
            else:
                regime_params = {
                    'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                    'adx_min': Config.ADX_MIN,
                    'num_filters': 3
                }

            # 3. Create Features
            self.logger.debug(f"Creating features for {name}")
            feature_df = self.feature_engine.create_features(raw_df.copy())
            
            # Ensure volatility_bbli exists
            if 'volatility_bbli' not in feature_df.columns:
                 bb_period = 20
                 sma = raw_df['Close'].rolling(window=bb_period, min_periods=1).mean()
                 std = raw_df['Close'].rolling(window=bb_period, min_periods=1).std()
                 bb_upper = sma + (2 * std)
                 bb_lower = sma - (2 * std)
                 feature_df['volatility_bbli'] = (raw_df['Close'] - bb_lower) / (bb_upper - bb_lower)
                 feature_df['volatility_bbli'] = feature_df['volatility_bbli'].fillna(0.5)

            if feature_df.empty:
                self.logger.warning(f"Feature creation failed for {name}. Skipping signal.")
                return {
                    'index': name, 'position_size': 0, 
                    'validation_reason': "Feature Error", 
                    'price': 0, 'regime': 'N/A'
                }

            # Revised Candle Policy: 4-Minute Look (Optimized for faster entry)
            # Standard: Wait for full 5m close.
            # Optimization: If bar is >4m old (80% complete), use it for predictive entry.
            latest_bar_time = feature_df.index[-1]
            now = datetime.now(Config.TIMEZONE)
            if latest_bar_time.tzinfo is None:
                latest_bar_time = Config.TIMEZONE.localize(latest_bar_time)
            
            bar_age_seconds = (now - latest_bar_time).total_seconds()
            
            # If bar is very fresh (<240s), it's risky to use for indicators. Use the previous closed bar.
            # If bar is >240s (4 mins), it's reliable enough for an "early-look" entry.
            if bar_age_seconds < 240 and len(feature_df) > 2:
                self.logger.debug(f"Current 5-min candle only {bar_age_seconds:.0f}s old. Using previous bar for {name}.")
                current_features_row = feature_df.iloc[[-2]].copy()
                latest_date = feature_df.index[-2]
            else:
                if bar_age_seconds >= 240:
                    self.logger.info(f"Using 4-minute 'Early Look' on current bar for {name} ({bar_age_seconds:.0f}s old).")
                current_features_row = feature_df.iloc[[-1]].copy()
                latest_date = feature_df.index[-1]

            raw_df.index = pd.to_datetime(raw_df.index)
            current_price = raw_df['Close'].iloc[-1]  # Always use latest live price for entry

            # Fix 6: India VIX Integration
            # Reduce or halt trading when market fear is elevated.
            try:
                vix_df = self.data_fetcher.fetch_data('^INDIAVIX', period='5d')
                if vix_df is not None and not vix_df.empty:
                    india_vix = vix_df['Close'].iloc[-1]
                    if india_vix > 25:
                        self.logger.warning(f"India VIX = {india_vix:.1f} (>25). Extreme fear. Skipping {name}.")
                        return {
                            'index': name, 'position_size': 0, 
                            'validation_reason': "High VIX (>25)", 
                            'price': current_price, 'regime': regime
                        }
                    elif india_vix > 22:
                        self.logger.info(f"India VIX = {india_vix:.1f} (>22). Elevated fear. Will reduce size for {name}.")
                        # We'll halve position size downstream via a flag
                        vix_size_multiplier = 0.5
                    else:
                        vix_size_multiplier = 1.0
                else:
                    vix_size_multiplier = 1.0  # VIX data unavailable, proceed normally
            except Exception as e:
                self.logger.debug(f"Could not fetch India VIX: {e}. Proceeding normally.")
                vix_size_multiplier = 1.0

            # Get ATR
            atr = 0
            if 'atr' in current_features_row.columns:
                 atr = float(current_features_row['atr'].iloc[0])
            elif 'atr_ratio' in current_features_row.columns:
                 atr = float(current_features_row['atr_ratio'].iloc[0]) * current_price
            else:
                 try:
                     # Calculate actual 14-period ATR using raw 5-minute data
                     high = raw_df['High']
                     low = raw_df['Low']
                     close = raw_df['Close']
                     tr = pd.concat([
                         high - low,
                         (high - close.shift()).abs(),
                         (low - close.shift()).abs()
                     ], axis=1).max(axis=1)
                     atr = float(tr.rolling(14).mean().iloc[-1])
                 except Exception:
                     atr = current_price * 0.002 # Fallback to 0.2% for intraday

            # 4. Neural/ML Prediction
            required_features = model_pack.get("features", [])
            current_features_row = current_features_row.reindex(columns=required_features, fill_value=0.0)
            X_pred = current_features_row[required_features]
            X_pred = X_pred.fillna(0).replace([np.inf, -np.inf], 0)

            base_models = model_pack.get("base_models", {})
            ordered_base_names = model_pack.get("ordered_base_names", list(base_models.keys()))

            # Fix: Ensure meta_features order precisely aligns with OOF training order
            is_multiclass = Config.TARGET_MODE == 'three_class'
            num_base_classes = 3 if is_multiclass else 1
            meta_features = np.zeros((1, len(ordered_base_names) * num_base_classes))
            
            for i, mname in enumerate(ordered_base_names):
                if mname in base_models:
                    model = base_models[mname]
                    try:
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(X_pred)[0]
                            if is_multiclass:
                                for c_idx in range(3):
                                    meta_features[0, i * 3 + c_idx] = prob[c_idx]
                            else:
                                meta_features[0, i] = prob[1] if len(prob) > 1 else prob[0]
                        else:
                            # Fallback if no proba
                            if is_multiclass:
                                meta_features[0, i * 3 + 2] = 1.0 # Default to Neutral
                            else:
                                meta_features[0, i] = 0.5
                    except Exception as e:
                         self.logger.warning(f"Base model {mname} prediction failed: {e}")

            calibrator = model_pack.get("calibrator") or model_pack.get("meta_clf")
            
            # Probability and Direction Logic (3-Class Aware)
            direction = None
            probability = 0.5
            ml_conviction = 0.0
            
            if calibrator:
                try:
                    probs = calibrator.predict_proba(meta_features)[0]
                    if is_multiclass and len(probs) >= 3:
                        # Index Mapping from improved_targets: 0: BEAR, 1: BULL, 2: NEUTRAL
                        bear_prob = probs[0]
                        bull_prob = probs[1]
                        neutral_prob = probs[2]
                        
                        confidence_threshold = regime_params.get('confidence_threshold', Config.CONFIDENCE_THRESHOLD)
                        
                        # Logic: Only trade if BULL or BEAR beats the opposite direction and exceeds threshold.
                        # Do NOT require it to beat NEUTRAL, as sideways regimes often have high NEUT probabilities
                        # even when an actionable directional setup exists. Let the validator decide.
                        if bull_prob > bear_prob and bull_prob >= confidence_threshold:
                            direction = "CE"
                            probability = bull_prob
                            ml_conviction = bull_prob - bear_prob
                        elif bear_prob > bull_prob and bear_prob >= confidence_threshold:
                            direction = "PE"
                            probability = bear_prob
                            ml_conviction = bear_prob - bull_prob
                        else:
                            self.logger.info(f"ML Neutral/Low Conviction for {name}: BULL={bull_prob:.2f}, BEAR={bear_prob:.2f}, NEUT={neutral_prob:.2f}")
                            return {
                                'index': name, 'position_size': 0, 
                                'validation_reason': "ML Neutral/Low Conv", 
                                'price': current_price, 'regime': regime,
                                'confidence': max(probs)
                            }
                    else:
                        # Binary fallback
                        probability = probs[1] if len(probs) > 1 else probs[0]
                        confidence_threshold = regime_params.get('confidence_threshold', Config.CONFIDENCE_THRESHOLD)
                        if probability >= confidence_threshold:
                            direction = "CE"
                        elif (1.0 - probability) >= confidence_threshold:
                             direction = "PE"
                        ml_conviction = abs(probability - 0.5) * 2
                            
                except Exception as e:
                    self.logger.warning(f"Inference failed for {name}: {e}")
                    return {
                        'index': name, 'position_size': 0, 
                        'validation_reason': "ML Inference Error", 
                        'price': current_price, 'regime': regime
                    }
            
            if not direction:
                 return {
                    'index': name, 'position_size': 0, 
                    'validation_reason': "No Directional Edge", 
                    'price': current_price, 'regime': regime
                }

            # Fix 3: ML Confidence Hard Floor (with late-entry adjustment)
            # If the ML model has essentially no conviction (probability near 0.5),
            # the direction decision is unreliable regardless of validator approval.
            ml_conviction_raw = abs(probability - 0.5)  # 0.0 = coin flip, 0.5 = perfect
            ML_FLOOR = 0.02  # Minimum conviction (Lowered from 0.04)
            
            # After 3:00 PM, conviction needs minimal boost
            if 'late_entry' in locals() and late_entry and minutes_to_close < 30:
                required_conviction = ML_FLOOR * 1.05  # Lowered from 1.15
                if ml_conviction_raw < required_conviction:
                    self.logger.info(
                        f"ML confidence too low for late entry on {name}: conviction={ml_conviction_raw:.3f} "
                        f"(prob={probability:.3f}, required={required_conviction:.3f}). Skipping."
                    )
                    return {
                        'index': name, 'position_size': 0,
                        'validation_reason': f"Low Conviction for Late Entry (need {required_conviction:.2f})",
                        'price': current_price, 'regime': regime,
                        'confidence': probability
                    }
            
            # Standard ML floor check
            if ml_conviction_raw < ML_FLOOR:
                self.logger.info(
                    f"ML confidence too low for {name}: conviction={ml_conviction_raw:.3f} "
                    f"(prob={probability:.3f}, floor={ML_FLOOR}). Skipping."
                )
                return {
                    'index': name, 'position_size': 0, 
                    'validation_reason': "ML Conviction < Floor", 
                    'price': current_price, 'regime': regime,
                    'confidence': probability
                }

            # Regime Alignment Check
            # Prevent counter-trend trades in strong trending regimes
            if regime == 'trending_up' and direction == 'PE':
                self.logger.warning(f"Counter-trend PE signal BLOCKED in {regime} for {name}")
                return {
                    'index': name, 'position_size': 0,
                    'validation_reason': f"Counter-trend PE in {regime}",
                    'price': current_price, 'regime': regime
                }
            elif regime == 'trending_down' and direction == 'CE':
                self.logger.warning(f"Counter-trend CE signal BLOCKED in {regime} for {name}")
                return {
                    'index': name, 'position_size': 0,
                    'validation_reason': f"Counter-trend CE in {regime}",
                    'price': current_price, 'regime': regime
                }

            # Fix 9: Cross-index correlation check (after direction is known)
            open_positions = self.state.state.get('open_positions', {})
            if len(open_positions) >= 2:
                open_directions = [p.get('direction') for p in open_positions.values()]
                if all(d == direction for d in open_directions):
                    self.logger.warning(
                        f"Correlation limit: All {len(open_positions)} open positions are {direction}. "
                        f"Skipping correlated {direction} entry for {name}."
                    )
                    return {
                        'index': name, 'position_size': 0, 
                        'validation_reason': "Correlation Limit", 
                        'price': current_price, 'regime': regime,
                        'direction': direction
                    }

            # Fix 6: Momentum Exhaustion Detection
            # NOTE: RSI divergence is also checked in SignalValidator._evaluate_momentum() with -40 penalty
            # We removed the hard block here to avoid double-penalty. The validator's score penalty is sufficient.
            # If RSI exhaustion is severe, validator will score < threshold and trade will be rejected.

            # B-5: Volume Confirmation Gate
            # Require volume >= 70% of 20-bar average before generating a signal.
            volume_ratio = 0.0
            try:
                if 'Volume' in raw_df.columns and len(raw_df) >= 20:
                    avg_vol_20 = raw_df['Volume'].iloc[-20:].mean()
                    current_vol = raw_df['Volume'].iloc[-1]
                    volume_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
                    if volume_ratio < 0.70:
                        self.logger.info(
                            f"[Signal] [{name}] Volume gate failed — "
                            f"vol_ratio={volume_ratio:.2f} < 0.70 (current={current_vol:.0f}, avg20={avg_vol_20:.0f})"
                        )
                        return {
                            'index': name, 'position_size': 0,
                            'validation_reason': f"Low Volume ({volume_ratio:.2f}x avg)",
                            'price': current_price, 'regime': regime
                        }
                    else:
                        self.logger.debug(f"[Signal] [{name}] Volume gate passed — vol_ratio={volume_ratio:.2f}")
            except Exception as e:
                self.logger.debug(f"Volume gate check failed: {e}")

            # B-4: Strengthened Choppy Regime Filter (RSI + MACD)
            if regime in ('choppy', 'sideways'):
                rsi_ok = True
                macd_ok = True
                try:
                    rsi_col  = next((c for c in feature_df.columns if 'rsi' in c.lower()), None)
                    macd_col = next((c for c in feature_df.columns if 'macd_diff' in c.lower() or 'macd_hist' in c.lower()), None)

                    if rsi_col:
                        rsi_val = float(feature_df[rsi_col].iloc[-1])
                        if 40 <= rsi_val <= 60:  # RSI stuck in neutral — high chop probability
                            rsi_ok = False
                            self.logger.info(f"[Signal] [{name}] Choppy filter: RSI neutral ({rsi_val:.1f}) in choppy regime.")

                    if macd_col:
                        macd_hist = feature_df[macd_col].iloc[-3:]  # last 3 bars
                        # Contracting histogram: momentum dying
                        if len(macd_hist) >= 2 and (macd_hist.abs().diff().iloc[-1] < 0):
                            macd_ok = False
                            self.logger.info(f"[Signal] [{name}] Choppy filter: MACD histogram contracting in choppy regime.")
                except Exception as e:
                    self.logger.debug(f"Choppy regime RSI/MACD check failed: {e}")

                if not rsi_ok or not macd_ok:
                    return {
                        'index': name, 'position_size': 0,
                        'validation_reason': f"Choppy regime — RSI/MACD confirm chop",
                        'price': current_price, 'regime': regime
                    }

            # ADX Hard Gate: Require minimum trend strength for directional trades
            # B-3: Use phase-specific ADX minimum
            adx_col = 'trend_adx' if 'trend_adx' in feature_df.columns else None
            if adx_col:
                current_adx = feature_df[adx_col].iloc[-1]
            else:
                # Fallback ADX calculation if column missing
                try:
                    high = feature_df['High']
                    low = feature_df['Low']
                    close = feature_df['Close']
                    plus_dm = high.diff().clip(lower=0)
                    minus_dm = (-low.diff()).clip(lower=0)
                    tr = pd.concat([
                        high - low,
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()
                    ], axis=1).max(axis=1)
                    atr14 = tr.rolling(14).mean()
                    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
                    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
                    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
                    current_adx = dx.rolling(14).mean().iloc[-1]
                    self.logger.debug(f"Fallback ADX calculated for {name}: {current_adx:.1f}")
                except Exception:
                    current_adx = 0  # Cannot compute → fail closed (block trade)
                    self.logger.warning(f"ADX unavailable for {name}. Blocking trade (fail-safe).")
            
            adx_threshold = max(
                regime_params.get('adx_min', 20),
                phase_cfg.get('min_adx', 12)   # B-3: phase-specific minimum ADX
            ) + min_conviction_boost * 10       # B-2: DTE-based conviction boost translates to ADX cushion
            if current_adx < adx_threshold:
                self.logger.info(
                    f"[Signal] [{name}] ADX too low — ADX={current_adx:.1f} < {adx_threshold:.0f}. Skipping."
                )
                return {
                    'index': name, 'position_size': 0,
                    'validation_reason': f"ADX Low ({current_adx:.1f})",
                    'price': current_price, 'regime': regime
                }

            # Determine Direction for logging
            log_direction = "\033[92mCE\033[0m" if direction == "CE" else "\033[91mPE\033[0m"
            self.logger.info(f"ML Direction: {log_direction} (Conf: {probability:.3f})")

            # 6. Signal Validation (Deep Analysis)
            # Fix 5: Scale validator threshold based on ML confidence
            # Low ML confidence → require higher validator score
            if getattr(self, 'signal_validator', None):
                context = {'regime': regime, 'regime_params': regime_params}
                temp_signal = {'direction': direction, 'price': current_price, 'confidence': probability}
                
                # Dynamic threshold scaling: More gradual scaling to prevent over-rejection
                base_threshold = self.signal_validator.config.get('min_score', 70)
                
                if ml_conviction < 0.05:  # Lowered from 0.10
                    self.signal_validator.min_score = base_threshold + 2   # Lowered from +5
                elif ml_conviction < 0.10:
                    self.signal_validator.min_score = base_threshold + 0   # Lowered from +2
                elif ml_conviction < 0.15:
                    self.signal_validator.min_score = base_threshold + 0   # Lowered from +0
                else:
                    self.signal_validator.min_score = base_threshold
                
                # Additional late-entry penalty removed to allow catching late afternoon momentum
                if 'late_entry' in locals() and late_entry:
                    self.logger.debug(f"Late entry detected. Softened rules let this proceed with standard threshold {self.signal_validator.min_score}")
                
                # Boost threshold on expiry days
                if expiry_adj and expiry_adj.get('validator_threshold_boost', 0) > 0:
                    self.signal_validator.min_score += expiry_adj['validator_threshold_boost']
                
                is_valid, score, reason = self.signal_validator.validate_trade_setup(
                    signal=temp_signal,
                    df=feature_df,
                    context=context
                )
                
                # Restore default min_score
                self.signal_validator.min_score = self.signal_validator.config.get('min_score', 70)
                
                # Re-assign colored direction for logging
                log_direction = "\033[92mCE\033[0m" if direction == "CE" else "\033[91mPE\033[0m"

                if not is_valid:
                    self.logger.info(f"Signal REJECTED by Validator: {name} {log_direction} | Score: {score:.1f} | Reason: {reason}")
                    position_size = 0.0
                else:
                    self.logger.info(f"Signal \033[92mAPPROVED\033[0m by Validator: {name} {log_direction} | Score: {score:.1f} | Reason: {reason}")
                    position_size = 1.0 # Will be recalculated by risk manager
            else:
                # Fail-safe: If validator is unavailable, reject trade
                # Without scoring, VWAP blocks, and regime checks, trade quality is unverifiable.
                self.logger.warning(f"Signal validator unavailable for {name}. Blocking trade (fail-safe).")
                return {
                    'index': name, 'position_size': 0, 
                    'validation_reason': "Validator Missing", 
                    'price': current_price, 'regime': regime
                }
            
            # 7. Risk Management & Exits
            risk_amount = 0
            exits = {}
            
            if self.risk_manager:
                # Calculate size details
                signal_for_risk = {'atr': atr, 'price': current_price, 'confidence': probability}
                pos_info = self.risk_manager.calculate_position_size(signal_for_risk)
                
                # If was already rejected, force size to 0
                if 'position_size' in locals() and position_size == 0.0:
                    pos_info['position_size'] = 0.0
                
                position_size = pos_info['position_size']
                
                # Apply VIX-based size reduction if applicable
                if 'vix_size_multiplier' in locals() and vix_size_multiplier < 1.0:
                    position_size = int(position_size * vix_size_multiplier)
                    self.logger.info(f"VIX size reduction applied: position_size = {position_size}")
                
                # Apply expiry day size reduction
                if expiry_adj and expiry_adj.get('position_size_multiplier', 1.0) < 1.0:
                    position_size = int(position_size * expiry_adj['position_size_multiplier'])
                    self.logger.info(f"Expiry size reduction: position_size = {position_size}")
                
                risk_amount = pos_info['risk_amount']
                
                # Calculate actual Stops/Targets using Risk Manager
                # Now passes regime and current_hour for adaptive TP/SL (Fixes 10, 12)
                current_hour = datetime.now(Config.TIMEZONE).hour
                exits = self.risk_manager.calculate_exits(
                     entry_price=current_price,
                     direction=direction,
                     atr=atr,
                     df=feature_df,
                     regime=regime,
                     current_hour=current_hour
                )
                
                # A-6: Slippage applied to ENTRY PRICE only, never to SL/TP levels
                # SL and TP are structural/ATR levels derived from adjusted entry.
                SLIPPAGE_PTS = 2.0
                if direction == 'CE':   # BUY
                    entry_adjusted = current_price + SLIPPAGE_PTS
                else:                   # PE (BUY PUT)
                    entry_adjusted = current_price - SLIPPAGE_PTS

                # Recalculate exits using the slippage-adjusted entry price
                exits = self.risk_manager.calculate_exits(
                    entry_price=entry_adjusted,
                    direction=direction,
                    atr=atr,
                    df=feature_df,
                    regime=regime,
                    current_hour=current_hour
                )
                # Store adjusted entry back for signal output
                current_price = entry_adjusted
                
                # Apply expiry day tightening to TP/SL
                if expiry_adj:
                    tp_tight = expiry_adj.get('tp_tightening', 1.0)
                    sl_tight = expiry_adj.get('sl_tightening', 1.0)
                    if tp_tight < 1.0:
                        # Tighten TP toward entry
                        tp1_dist = abs(exits['tp1'] - current_price)
                        tp2_dist = abs(exits['tp2'] - current_price)
                        if direction == 'CE':
                            exits['tp1'] = current_price + tp1_dist * tp_tight
                            exits['tp2'] = current_price + tp2_dist * tp_tight
                        else:
                            exits['tp1'] = current_price - tp1_dist * tp_tight
                            exits['tp2'] = current_price - tp2_dist * tp_tight
                    if sl_tight < 1.0:
                        sl_dist = abs(exits['stop_loss'] - current_price)
                        if direction == 'CE':
                            exits['stop_loss'] = current_price - sl_dist * sl_tight
                        else:
                            exits['stop_loss'] = current_price + sl_dist * sl_tight
            
            # 8. Filter Check (Volume, etc - fallback if not using Validator)
            valid_signal = True
            if position_size <= 0:
                valid_signal = False

            # 9. Construct Final Signal — enriched with all context metadata
            # D-2: Compute recommended strike
            recommended_strike, _ = self._get_recommended_strike(
                name, current_price, direction, regime, dte
            ) if dte >= 0 else (self._get_atm_strike(name, current_price), None)

            # D-3: Compute expiry label
            try:
                expiry_label = str(get_next_expiry(name, now_ist.date())) if get_next_expiry else 'N/A'
            except Exception:
                expiry_label = 'N/A'

            signal = {
                'index': name,
                'direction': direction,
                'price': current_price,
                'confidence': probability,
                'timestamp': datetime.now(Config.TIMEZONE).isoformat(),
                'time': datetime.now(Config.TIMEZONE).strftime("%H:%M:%S"),
                'regime': regime,
                'atr': atr,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_loss': exits.get('stop_loss'),
                'take_profit_1': exits.get('tp1'),
                'take_profit_2': exits.get('tp2'),
                'validation_score': score if 'score' in locals() else None,
                'validation_reason': reason if 'reason' in locals() else "Low Confidence",
                # D-1/D-2: New metadata for rich signal card
                'strike': recommended_strike,
                'expiry': expiry_label,
                'dte': dte,
                'session_phase': session_phase if 'session_phase' in locals() else 'unknown',
                'volume_ratio': round(volume_ratio, 2) if 'volume_ratio' in locals() else 0.0,
                'adx': round(float(current_adx), 1) if 'current_adx' in locals() else 0.0,
            }

            # Print Actionable Signal Card and save JSON ONLY if approved
            if position_size > 0:
                self.print_signal_card(signal)
                self._save_signal_json(signal)  # D-3: write signals/latest_signal.json + history

            return signal

        except Exception as e:
            self.logger.error(f"Signal generation failed for {name} ({symbol}): {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def print_signal_card(self, signal: Dict):
        """D-1: Prints a rich, institutional-grade signal alert to console and log."""
        name       = signal.get('index', 'N/A')
        direction  = signal.get('direction', 'N/A')
        price      = signal.get('price', 0)
        sl         = signal.get('stop_loss', 0) or 0
        tp1        = signal.get('take_profit_1', 0) or 0
        tp2        = signal.get('take_profit_2', 0) or 0
        conf       = signal.get('confidence', 0) or 0
        score      = signal.get('validation_score', 0) or 0
        regime     = signal.get('regime', 'N/A')
        dte        = signal.get('dte', '?')
        strike     = signal.get('strike', 'ATM')
        expiry     = signal.get('expiry', 'N/A')
        vol_ratio  = signal.get('volume_ratio', 0) or 0
        session    = signal.get('session_phase', 'N/A')
        adx        = signal.get('adx', 0) or 0
        ts         = datetime.now(Config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')

        risk_amt = abs(price - sl) if sl else 0
        reward1  = abs(tp1 - price) if tp1 else 0
        rr_ratio = reward1 / risk_amt if risk_amt > 0 else 0
        signal['rr_ratio'] = round(rr_ratio, 2)  # enrich signal dict

        dir_label  = 'CE (BULLISH)' if direction == 'CE' else 'PE (BEARISH)'

        box = (
            f"\n╔══════════════════════════════════════════════════╗\n"
            f"║  SIGNAL ALERT — {ts:<33}║\n"
            f"╠══════════════════════════════════════════════════╣\n"
            f"║  Index     : {name:<36}║\n"
            f"║  Direction : {dir_label:<36}║\n"
            f"║  Spot Price: {price:<36.2f}║\n"
            f"║  Strike    : {str(strike) + ' ' + direction:<36}║\n"
            f"║  Expiry    : {str(expiry) + '  (' + str(dte) + 'd to expiry)':<36}║\n"
            f"╠══════════════════════════════════════════════════╣\n"
            f"║  ENTRY     : {price:<36.2f}║\n"
            f"║  STOP LOSS : {sl:<.2f}  (risk: ₹{risk_amt:<.2f}){'':<10}║\n"
            f"║  TARGET 1  : {tp1:<.2f}  (partial exit — 50%){'':<9}║\n"
            f"║  TARGET 2  : {tp2:<36.2f}║\n"
            f"║  R:R Ratio : 1:{rr_ratio:<33.1f}║\n"
            f"╠══════════════════════════════════════════════════╣\n"
            f"║  CONFIDENCE: {conf:.1%}{'':<33}║\n"
            f"║  Regime    : {regime:<36}║\n"
            f"║  Validator : {score:<.1f}/100{'':<31}║\n"
            f"║  Volume    : {vol_ratio:<.1f}x avg{'':<30}║\n"
            f"║  Session   : {session:<36}║\n"
            f"║  ADX       : {adx:<36.1f}║\n"
            f"╚══════════════════════════════════════════════════╝"
        )
        print(box)
        self.logger.info(
            f"[Signal] [{name}] {direction} signal — confidence={conf:.1%}, "
            f"regime={regime}, score={score:.1f}, dte={dte}, session={session}, adx={adx:.1f}, vol={vol_ratio:.2f}x"
        )

    def _get_atm_strike(self, index_name: str, spot_price: float) -> int:
        """Calculate ATM strike price based on index."""
        if "BANK" in index_name:
            return int(round(spot_price / 100) * 100)  # Step = 100
        else:
            return int(round(spot_price / 50) * 50)   # Step = 50 (NIFTY / FINNIFTY)

    def _get_recommended_strike(self, index_name: str, spot_price: float,
                                direction: str, regime: str, dte: int) -> Tuple[int, int]:
        """D-2: Return (recommended_strike, strike_step) based on regime and DTE.

        Selection rules:
            trending regime + DTE > 1 : ATM (max delta)
            choppy regime             : 1 strike OTM (reduce premium cost)
            expiry day (DTE == 0)     : ATM only (avoid OTM gamma risk)
        """
        # Determine strike step per index
        STRIKE_STEPS = {'NIFTY': 50, 'BANKNIFTY': 100, 'FINNIFTY': 50}
        step = STRIKE_STEPS.get(index_name, 50)
        atm = int(round(spot_price / step) * step)

        if dte == 0:  # Expiry day: ATM only
            return atm, step
        if regime in ('choppy', 'sideways'):
            # 1 OTM strike to reduce premium
            if direction == 'CE':
                return atm + step, step
            else:
                return atm - step, step
        # Default: ATM for trending or unknown
        return atm, step

    def _save_signal_json(self, signal: Dict):
        """D-3: Write signal to signals/latest_signal.json (overwrite) and append to signal_history.json."""
        try:
            import json as _json
            signals_dir = os.path.join(Config.BASE_DIR, 'signals')
            os.makedirs(signals_dir, exist_ok=True)

            payload = {
                'timestamp':       signal.get('timestamp', datetime.now(Config.TIMEZONE).isoformat()),
                'index':           signal.get('index'),
                'direction':       signal.get('direction'),
                'spot_price':      round(float(signal.get('price', 0)), 2),
                'strike':          signal.get('strike'),
                'expiry':          str(signal.get('expiry', '')),
                'dte':             signal.get('dte', -1),
                'entry':           round(float(signal.get('price', 0)), 2),
                'stop_loss':       round(float(signal.get('stop_loss') or 0), 2),
                'target1':         round(float(signal.get('take_profit_1') or 0), 2),
                'target2':         round(float(signal.get('take_profit_2') or 0), 2),
                'rr_ratio':        signal.get('rr_ratio', 0.0),
                'confidence':      round(float(signal.get('confidence', 0)), 4),
                'regime':          signal.get('regime', 'unknown'),
                'validator_score': signal.get('validation_score', 0),
                'volume_ratio':    round(float(signal.get('volume_ratio', 0)), 2),
                'session_phase':   signal.get('session_phase', 'unknown'),
                'adx':             round(float(signal.get('adx', 0)), 1),
                'status':          'PENDING',
            }

            # Overwrite latest_signal.json
            latest_path = os.path.join(signals_dir, 'latest_signal.json')
            with open(latest_path, 'w') as f:
                _json.dump(payload, f, indent=2, default=str)

            # Append to signal_history.json
            history_path = os.path.join(signals_dir, 'signal_history.json')
            history = []
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        history = _json.load(f)
                except Exception:
                    history = []
            history.append(payload)
            with open(history_path, 'w') as f:
                _json.dump(history, f, indent=2, default=str)

            self.logger.info(f"[Signal] [{signal.get('index')}] JSON saved — {latest_path}")
        except Exception as e:
            self.logger.warning(f"[Signal] Failed to save signal JSON: {e}")

    def print_dashboard(self, results: List[Dict]):
        """Prints a summary dashboard of the trading cycle."""
        if not results: return
        
        # Color definitions
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m" 
        RESET = "\033[0m"
        BOLD = "\033[1m"

        print("\n" + "="*140)
        print(f"║ {BOLD}⚡ MARKET STATUS SUMMARY - {datetime.now(Config.TIMEZONE).strftime('%H:%M:%S')} {RESET}{' '*100}║")
        print("="*140)
        # Columns: Index | Price | Regime | Action | SL | Target | Status | Reason | Conf
        print(f"║ {'INDEX':<12} ║ {'PRICE':<10} ║ {'REGIME':<15} ║ {'ACTION':<6} ║ {'SL':<10} ║ {'TARGET':<10} ║ {'STATUS':<15} ║ {'REASON/SCORE':<25} ║ {'CONF':<6} ║")
        print("╠" + "═"*14 + "╬" + "═"*12 + "╬" + "═"*17 + "╬" + "═"*8 + "╬" + "═"*12 + "╬" + "═"*12 + "╬" + "═"*17 + "╬" + "═"*27 + "╬" + "═"*8 + "╣")
        
        for res in results:
            index = res.get('index', 'N/A')
            price = res.get('price', 0.0)
            regime = res.get('regime', 'N/A')
            action = res.get('signal', '-')
            sl = res.get('sl', '-')
            target = res.get('target', '-')
            status = res.get('status', 'Waiting')
            reason = res.get('reason', '-')
            conf = res.get('conf', 0.0)
            
            # Color coding
            status_color = RESET
            if "APPROVED" in status or "OPEN" in status: status_color = GREEN
            elif "REJECTED" in status: status_color = RED
            elif "FILTERED" in status: status_color = CYAN
            elif "Error" in status: status_color = YELLOW
            
            # Additional formatting
            price_str = f"{price:.2f}" if isinstance(price, (int, float)) and price > 0 else "-"
            sl_str = f"{sl:.2f}" if isinstance(sl, (int, float)) and sl > 0 else "-"
            tgt_str = f"{target:.2f}" if isinstance(target, (int, float)) and target > 0 else "-"
            conf_str = f"{conf*100:.1f}%" if isinstance(conf, float) and conf > 0 else "-"
            
            # Truncate reason if too long
            if len(reason) > 25: reason = reason[:22] + "..."

            print(f"║ {index:<12} ║ {price_str:<10} ║ {regime:<15} ║ {action:<6} ║ {sl_str:<10} ║ {tgt_str:<10} ║ {status_color}{status:<15}{RESET} ║ {reason:<25} ║ {conf_str:<6} ║")
            
        print("="*140 + "\n")

    def execute_trade_lifecycle(self, name: str, symbol: str) -> Dict:
        """
        Manages the full lifecycle of a trade: Entry -> Monitor -> Exit.
        Returns details for the dashboard.
        """
        try:
            current_price = 0.0
            # 1. Monitor Open Position
            if name in self.state.state['open_positions']:
                # Update holding status
                try: 
                    raw_df = self.data_fetcher.fetch_data(symbol, period="1d")
                    if not raw_df.empty:
                        current_price = raw_df['Close'].iloc[-1]
                except:
                    pass
                
                pos = self.state.state['open_positions'][name]
                self._monitor_position(name, symbol, pos)
                
                # Check if still open after monitoring
                if name in self.state.state['open_positions']:
                     return {
                        'index': name, 
                        'price': current_price, 
                        'regime': 'HOLDING', 
                        'signal': pos['direction'],
                        'status': 'OPEN'
                     }
                else:
                     # Add a cooldown mechanism by setting last_signal_bar
                     self.last_signal_bar[name] = pd.Timestamp.now()
                     return {
                        'index': name,
                        'price': current_price,
                        'regime': '-',
                        'signal': '-',
                        'status': 'CLOSED'
                     }

            # 2. Look for New Entry (if no position)
            # Check for cooldown
            if name in self.last_signal_bar:
                time_since_last_signal = pd.Timestamp.now() - self.last_signal_bar[name]
                # A-7: Use Config.COOLDOWN_MINUTES (was hardcoded 30, now reads from Config)
                cooldown_minutes = getattr(Config, 'COOLDOWN_MINUTES', 30)
                if time_since_last_signal < pd.Timedelta(minutes=cooldown_minutes):
                    self.logger.info(f"[{name}] Cooldown active — {time_since_last_signal.seconds//60}m elapsed of {cooldown_minutes}m. Skipping entry.")
                    return {'index': name, 'status': 'Cooldown'}
            # Check daily limits first
            limit_status = self.risk_manager.check_daily_limits() if self.risk_manager else 1.0
            if limit_status == 0.0:
                self.logger.info(f"Daily loss limit reached. Skipping entry for {name}.")
                return {'index': name, 'status': 'Daily Limit Reached'}

            signal = self.generate_signal(name, symbol)
            
            if not signal:
                 return {'index': name, 'status': 'No Signal/Data', 'reason': 'Fetch/Feat Err'}
            
            # Setup for dashboard return
            status = "Waiting"
            if signal.get('position_size', 0) > 0: 
                status = "APPROVED"
            elif signal.get('validation_reason'):
                # Distinguish between high-score rejection and early filtering
                if signal.get('validation_score') is not None:
                    status = "REJECTED"
                else:
                    status = "FILTERED"
            
            if signal.get('position_size', 0) > 0:
                self._execute_entry(name, symbol, signal, limit_status)
            
            return {
                'index': name,
                'price': signal.get('price', 0.0),
                'regime': signal.get('regime', '-'),
                'signal': signal.get('direction', '-'),
                'status': status,
                'sl': signal.get('stop_loss'),
                'target': signal.get('take_profit_1'),
                'reason': signal.get('validation_reason', '-'),
                'conf': signal.get('confidence', 0.0)
            }

        except Exception as e:
            self.logger.error(f"Error in trade lifecycle for {name}: {e}")
            import traceback
            traceback.print_exc()
            return {'index': name, 'status': 'Error'}

    def _monitor_position(self, name: str, symbol: str, position: Dict):
        """Monitor an open position for exit conditions with emergency breach detection."""
        try:
            raw_df = self.data_fetcher.fetch_data(symbol, period="1d")
            
            if raw_df.empty:
                return
            
            current_price = raw_df['Close'].iloc[-1]
            entry_price = position['entry_price']
            direction = position['direction']
            
            # Handle key variations for backward compatibility
            sl = position.get('sl', position.get('stop_loss'))
            tp = position.get('tp', position.get('take_profit_1', position.get('take_profit')))
            size = position.get('position_size', position.get('size', 0))
            
            # Emergency SL Breach Detection
            # If price has blown past SL by > 1.5× intended risk, trigger immediate exit
            # and log as emergency event for post-trade analysis.
            if sl is not None:
                intended_risk = abs(entry_price - sl)
                if direction == 'CE':
                    actual_breach = sl - current_price  # Positive if below SL
                else:  # PE
                    actual_breach = current_price - sl  # Positive if above SL
                
                if actual_breach > 0:
                    breach_multiple = actual_breach / intended_risk if intended_risk > 0 else 0
                    if breach_multiple > Config.EMERGENCY_SL_MULTIPLIER:
                        # Price has moved far past SL — emergency exit
                        if direction == 'CE':
                            pnl = (current_price - entry_price) * size
                        else:
                            pnl = (entry_price - current_price) * size
                        
                        # Deduct transaction costs
                        lot_count = max(1, size // 25)  # Approx lots
                        tx_cost = lot_count * Config.TRANSACTION_COST_PER_LOT
                        net_pnl = pnl - tx_cost
                        
                        self.logger.critical(
                            f"🚨 EMERGENCY SL BREACH for {name}: Price={current_price:.2f}, "
                            f"SL={sl:.2f}, Breach={breach_multiple:.1f}× risk. "
                            f"Gross P&L: {pnl:.2f}, Net P&L (after ₹{tx_cost} costs): {net_pnl:.2f}"
                        )
                        
                        if self.performance_tracker:
                            self.performance_tracker.update_trade_exit(
                                trade_id=position['trade_id'],
                                exit_price=current_price,
                                exit_reason='EMERGENCY_SL_BREACH',
                                pnl=net_pnl
                            )
                        
                        self.state.remove_position(symbol)
                        self.state.update_daily_stats(net_pnl, index_name=name)
                        self.logger.info(f"Emergency exit complete for {name}.")
                        return
            # Fix 10: Automated Trailing Stop
            # Move SL to breakeven (entry price) once 1:1 R:R is achieved.
            if sl is not None:
                risk = abs(entry_price - sl)
                if direction == "CE" and current_price >= entry_price + risk:
                    new_sl = entry_price
                    if sl < new_sl:  # Only move SL up, never down
                        self.logger.info(f"📈 Trailing SL to breakeven for {name}: {sl:.2f} → {new_sl:.2f}")
                        position['sl'] = new_sl
                        sl = new_sl
                        self.state.save_state()  # Persist the updated SL
                elif direction == "PE" and current_price <= entry_price - risk:
                    new_sl = entry_price
                    if sl > new_sl:  # Only move SL down, never up
                        self.logger.info(f"📉 Trailing SL to breakeven for {name}: {sl:.2f} → {new_sl:.2f}")
                        position['sl'] = new_sl
                        sl = new_sl
                        self.state.save_state()  # Persist the updated SL

            # C-1: STT Trap Alert for PE on Expiry Day
            # If holding a PE position on expiry day after 15:00 IST and price is within
            # 0.5% of the long put strike, warn about the STT trap on exercise settlement.
            try:
                if direction == 'PE' and is_expiry_day is not None:
                    pos_index = position.get('index', name)
                    now_ist = datetime.now(Config.TIMEZONE)
                    if is_expiry_day(pos_index) and now_ist.hour >= 15:
                        entry_strike = position.get('strike', entry_price)  # Use stored strike if available
                        dist_from_strike_pct = abs(current_price - entry_strike) / entry_strike if entry_strike else 0
                        if dist_from_strike_pct <= 0.005:  # Within 0.5% of strike
                            self.logger.warning(
                                f"[RiskMgr] [{name}] ⚠️ STT TRAP ALERT — "
                                f"PE on expiry day, price={current_price:.2f} within 0.5% of strike={entry_strike:.0f}. "
                                f"Do NOT let this expire ITM — manual exit strongly advised."
                            )
            except Exception as e:
                self.logger.debug(f"STT trap check failed: {e}")

            # Check Exit Conditions
            exit_reason = None
            pnl = 0.0
            
            # Fix 11: Partial Profit Booking at TP1
            # If the position hasn't already been partially closed, book 50% at TP1
            # and trail SL to entry for the remaining half.
            tp2 = position.get('tp2', position.get('take_profit_2'))
            already_partial = position.get('partial_booked', False)
            
            if not already_partial and tp is not None and size > 1:
                partial_hit = False
                if direction == "CE" and current_price >= tp:
                    partial_hit = True
                elif direction == "PE" and current_price <= tp:
                    partial_hit = True
                
                if partial_hit:
                    # Book 50% of position at TP1
                    partial_size = max(1, size // 2)
                    remaining_size = size - partial_size
                    
                    if direction == "CE":
                        partial_pnl = (current_price - entry_price) * partial_size
                    else:
                        partial_pnl = (entry_price - current_price) * partial_size
                    
                    self.logger.info(
                        f"💰 Partial profit booked for {name}: {partial_size} qty at {current_price:.2f}. "
                        f"P&L: {partial_pnl:.2f}. Remaining: {remaining_size} qty"
                    )
                    
                    # Update position: trail SL to entry, set new TP to TP2, mark partial booked
                    position['position_size'] = remaining_size
                    position['sl'] = entry_price  # Trail SL to breakeven
                    sl = entry_price
                    if tp2:
                        position['tp'] = tp2  # Move target to TP2
                        tp = tp2
                    position['partial_booked'] = True
                    size = remaining_size
                    
                    # Log partial trade to performance tracker
                    if self.performance_tracker:
                        self.performance_tracker.update_trade_exit(
                            trade_id=position['trade_id'] + '_partial',
                            exit_price=current_price,
                            exit_reason='Partial_TP1',
                            pnl=partial_pnl
                        )
                    
                    # Add partial profit to daily P&L
                    self.state.update_daily_stats(partial_pnl, index_name=name)
                    self.state.save_state()  # Store updated position
                    return  # Don't check full exit conditions this cycle
            
            if direction == "CE":
                if current_price <= sl:
                    exit_reason = "Stop Loss"
                elif current_price >= tp:
                    exit_reason = "Take Profit"
            elif direction == "PE":
                if current_price >= sl:
                    exit_reason = "Stop Loss"
                elif current_price <= tp:
                    exit_reason = "Take Profit"
            
            if exit_reason:
                # Calculate P&L
                if direction == "CE":
                    pnl = (current_price - entry_price) * size
                else:
                    pnl = (entry_price - current_price) * size
                
                # Deduct transaction costs for net P&L
                lot_count = max(1, size // 25)
                tx_cost = lot_count * Config.TRANSACTION_COST_PER_LOT
                net_pnl = pnl - tx_cost
                
                self.logger.info(
                    f"Closing {direction} position for {name} ({symbol}) at {current_price}. "
                    f"Reason: {exit_reason}. Gross P&L: {pnl:.2f}, Net P&L: {net_pnl:.2f} (after ₹{tx_cost} costs)"
                )
                
                # Update Risk Manager
                if self.risk_manager:
                    self.risk_manager.close_position(
                        position_id=position.get('trade_id'),
                        exit_price=current_price
                    )
                
                # Log to DB
                if self.performance_tracker:
                    self.performance_tracker.update_trade_exit(
                        trade_id=position['trade_id'],
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        pnl=net_pnl
                    )
                
                # Update State — use `name` (e.g., 'NIFTY') not `symbol` (e.g., '^NSEI')
                # This matches how execute_trade_lifecycle looks up positions by name.
                self.state.remove_position(symbol)
                self.state.update_daily_stats(net_pnl, index_name=name)
                
                # Notify User (via log)
                self.logger.info(f"Trade Closed: {name} {direction} | Net P&L: {net_pnl:.2f}")

        except Exception as e:
            self.logger.error(f"Error monitoring position for {name}: {e}")

    def _execute_entry(self, name: str, symbol: str, signal: Dict, limit_multiplier: float):
        """Execute a new trade entry."""
        try:
            current_price = signal['price']
            direction = signal['direction']
            atr = signal.get('atr', 0)
            
            # Use pre-calculated sizing from signal if available, otherwise fallback
            size = signal.get('position_size', 1)
            sl = signal.get('stop_loss', current_price)
            tp = signal.get('take_profit_1', current_price)
            
            # Apply limit multiplier
            size = int(size * limit_multiplier)
            
            if size <= 0:
                self.logger.info(f"Calculated position size is 0 for {name}. Skipping.")
                return

            # Log Trade
            # Generate a temporary string ID for logging before we get the DB ID
            temp_trade_id = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            final_trade_id = temp_trade_id 

            if self.performance_tracker:
                 db_id = self.performance_tracker.log_trade(
                    trade_id=temp_trade_id,
                    index_name=name, # Changed param name to match tracker
                    direction=direction,
                    entry_price=current_price,
                    position_size=size,
                    stop_loss=sl,
                    take_profit_1=tp,
                    take_profit_2=signal.get('take_profit_2'),
                    confidence=signal['confidence'],
                    regime=signal['regime'],
                    features={
                        'atr': atr,
                        'score': signal.get('validation_score')
                    },
                    validation_score=signal.get('validation_score')
                )
                 if db_id:
                     final_trade_id = db_id # Store the integer ID for database updates
            
            # Save to State
            position_data = {
                "trade_id": final_trade_id,
                "entry_price": current_price,
                "direction": direction,
                "position_size": size, # Standardized key
                "sl": sl,
                "tp": tp,
                "tp2": signal.get('take_profit_2'),  # Needed for partial profit booking
                "entry_time": datetime.now().isoformat(),
                "regime": signal['regime']
            }
            self.state.add_position(name, position_data)  # Use name, not symbol
            
            self.logger.info(f"🚀 OPENED {direction} for {name} @ {current_price} | Size: {size} | SL: {sl} | TP: {tp}")

        except Exception as e:
            self.logger.error(f"Failed to execute entry for {name}: {e}")

    @staticmethod
    def is_market_open() -> bool:
        """Check if market is open (IST). Avoids opening chop and closing volatility."""
        now = datetime.now(Config.TIMEZONE)
        # Check for weekends (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            return False

        # Check for market hours
        market_start_time = datetime.strptime(Config.MARKET_START, "%H:%M").time()
        market_end_time = datetime.strptime(Config.MARKET_END, "%H:%M").time()

        # Shield: Skip the opening 15 minutes (09:15-09:30) for erratic overnight unwind
        # Shield: Skip the last 15 minutes (15:15-15:30) for closing square-off volatility
        check_start_time = (datetime.combine(datetime.today(), market_start_time) + timedelta(minutes=15)).time()
        check_end_time = (datetime.combine(datetime.today(), market_end_time) - timedelta(minutes=15)).time()

        return check_start_time <= now.time() < check_end_time # Use strict less than for end time

    def run(self, poll_interval: int = 300):
        """Main trading loop."""
        self.logger.info("Starting trading bot loop...")
        self.logger.info(f"Monitoring indices: {list(Config.INDICES.keys())}")
        self.logger.info(f"Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")
        self.logger.info(f"Volume Ratio threshold: {Config.MIN_VOLUME_RATIO}")

        consecutive_run_errors = 0
        max_consecutive_run_errors = 10 # Max runtime errors before stopping

        # Ensure models are loaded/trained before starting the loop
        if not self.models:
            self.logger.critical("No models loaded/trained. Exiting run loop.")
            return

        while True:
            try:
                current_time = datetime.now(Config.TIMEZONE)

                # Clean up overnight positions if any exist (e.g. from previous days)
                open_positions = dict(self.state.state.get('open_positions', {}))
                for pos_name, pos_data in open_positions.items():
                    entry_time_str = pos_data.get('entry_time')
                    if entry_time_str:
                        try:
                            entry_dt = datetime.fromisoformat(entry_time_str)
                            if entry_dt.date() < current_time.date():
                                self.logger.warning(f"🧹 Stale overnight position detected for {pos_name}. Closing it to allow new trades today.")
                                if getattr(self, 'performance_tracker', None):
                                    self.performance_tracker.update_trade_exit(
                                        trade_id=pos_data.get('trade_id', pos_name),
                                        exit_price=pos_data.get('entry_price', 0.0),
                                        exit_reason='STALE_OVERNIGHT_CLEANUP',
                                        pnl=0.0
                                    )
                                self.state.remove_position(pos_name)
                        except Exception as e:
                            self.logger.debug(f"Error checking position age for {pos_name}: {e}")

                if Config.USE_EOD_SIGNALS:
                    # EOD mode: wait until after market close and trigger only once per day
                    if self.is_market_open():
                        print(f"\r[{current_time.strftime('%Y-%m-%d %H:%M:%S IST')}] Market open (EOD mode). Next check in 60s...", end="", flush=True)
                        time.sleep(60)
                        consecutive_run_errors = 0
                        continue
                    # Market closed. Run once per new date.
                    if self.last_eod_scan_date == current_time.date():
                        print(f"\r[{current_time.strftime('%Y-%m-%d %H:%M:%S IST')}] EOD scan already done today. Next check in 300s...", end="", flush=True)
                        time.sleep(300)
                        continue
                    self.last_eod_scan_date = current_time.date()
                else:
                    if not self.is_market_open():
                        # Check again periodically even when market is closed
                        print(f"\r[{current_time.strftime('%Y-%m-%d %H:%M:%S IST')}] Market closed. Next check in 60s...", end="", flush=True)
                        time.sleep(60)
                        consecutive_run_errors = 0 # Reset error count when waiting for market
                        continue

                # Fix 2: Auto-square-off at 15:10 IST
                # Close all open positions to prevent overnight gap risk.
                # Index options decay overnight (theta) and gap risk is unhedgeable.
                if current_time.hour == 15 and current_time.minute >= 10:
                    open_positions = dict(self.state.state.get('open_positions', {}))
                    if open_positions:
                        self.logger.info(f"🔔 Auto-square-off triggered at 15:10 IST. Closing {len(open_positions)} open positions.")
                        for pos_name, pos_data in open_positions.items():
                            try:
                                symbol = Config.INDICES.get(pos_name, pos_name)
                                raw_df = self.data_fetcher.fetch_data(symbol, period="1d")
                                if not raw_df.empty:
                                    exit_price = raw_df['Close'].iloc[-1]
                                else:
                                    exit_price = pos_data['entry_price']  # Fallback
                                
                                entry_price = pos_data['entry_price']
                                direction = pos_data['direction']
                                size = pos_data.get('position_size', pos_data.get('size', 0))
                                
                                if direction == 'CE':
                                    pnl = (exit_price - entry_price) * size
                                else:
                                    pnl = (entry_price - exit_price) * size
                                
                                self.logger.info(
                                    f"⏹️ EOD Square-off: {pos_name} {direction} | "
                                    f"Entry: {entry_price:.2f} → Exit: {exit_price:.2f} | P&L: {pnl:.2f}"
                                )
                                
                                if self.performance_tracker:
                                    self.performance_tracker.update_trade_exit(
                                        trade_id=pos_data.get('trade_id', pos_name),
                                        exit_price=exit_price,
                                        exit_reason='EOD_SQUAREOFF',
                                        pnl=pnl
                                    )
                                
                                self.state.remove_position(pos_name)
                                self.state.update_daily_stats(pnl, index_name=pos_name)
                                
                            except Exception as e:
                                self.logger.error(f"Error during EOD square-off for {pos_name}: {e}")
                        
                        self.logger.info("✅ EOD square-off complete. All positions closed.")

                # C-3: Auto daily summary at market close (after 15:30 IST)
                if current_time.hour >= 15 and current_time.minute >= 30:
                    daily_pnl = getattr(self.risk_manager, 'daily_pnl', 0.0) if self.risk_manager else 0.0
                    daily_trades = getattr(self.risk_manager, 'daily_trades', 0) if self.risk_manager else 0
                    self.logger.info(
                        f"[DailySummary] Date={current_time.date()} — "
                        f"Trades={daily_trades}, Net_PnL=₹{daily_pnl:.2f}, "
                        f"Account=₹{getattr(self.risk_manager, 'account_balance', 0):.2f}"
                    )
                
                # Proceed with scanning
                self.logger.info("Market open. Executing trade lifecycle...")
                
                cycle_results = []
                for name, symbol in Config.INDICES.items():
                    try:
                        res = self.execute_trade_lifecycle(name, symbol)
                        if res: cycle_results.append(res)
                    except Exception as e:
                        self.logger.error(f"Error in trade lifecycle for {name}: {e}")
                        cycle_results.append({'index': name, 'status': 'Error'})

                consecutive_run_errors = 0  # Reset error counter on successful run cycle
                
                # Print Dashboard
                self.print_dashboard(cycle_results)

                # Sleep
                # Split-interval sleep: fast-poll open positions every MONITOR_INTERVAL,
                # full lifecycle scan every POLL_INTERVAL. Worst-case SL delay = MONITOR_INTERVAL.
                poll_interval = Config.POLL_INTERVAL
                monitor_interval = Config.MONITOR_INTERVAL
                self.logger.info(f"Lifecycle completed. Next full scan in {poll_interval}s. Monitoring every {monitor_interval}s.")
                
                elapsed = 0
                while elapsed < poll_interval:
                    remaining = poll_interval - elapsed
                    sleep_time = min(monitor_interval, remaining)
                    
                    print(f"\r⏳ Next scan in {remaining}s (monitoring active)...", end="", flush=True)
                    time.sleep(sleep_time)
                    elapsed += sleep_time
                    
                    # Fast-poll: check open positions only (no signal generation)
                    open_positions = dict(self.state.state.get('open_positions', {}))
                    if open_positions:
                        for pos_name, pos_data in open_positions.items():
                            try:
                                pos_symbol = Config.INDICES.get(pos_name, pos_name)
                                if pos_symbol:
                                    self._monitor_position(pos_name, pos_symbol, pos_data)
                            except Exception as e:
                                self.logger.error(f"Fast-poll monitor error for {pos_name}: {e}")
                print("\r" + " "*30 + "\r", end="", flush=True) # Clear line

            except KeyboardInterrupt:
                print("\n\n🛑 Bot stopped by user")
                self.logger.info("Bot stopped by user")
                break # Exit the loop

            except Exception as e:
                consecutive_run_errors += 1
                self.logger.error(f"Top-level runtime error ({consecutive_run_errors}/{max_consecutive_run_errors}): {str(e)}")
                print(f"\n❌ Top-level error occurred: {str(e)}")

                if consecutive_run_errors >= max_consecutive_run_errors:
                    self.logger.critical("Too many consecutive runtime errors. Shutting down.")
                    print("🚨 Too many errors. Shutting down bot.")
                    break # Exit the loop

                # Exponential backoff for errors in the main loop
                error_sleep_time = min(60 * (2 ** (consecutive_run_errors - 1)), 600) # Max 10 min sleep
                self.logger.info(f"Retrying main loop in {error_sleep_time}s due to error.")
                print(f"⏳ Retrying in {error_sleep_time}s...")
                time.sleep(error_sleep_time) # Sleep before next attempt

    def health_check(self) -> Dict:
        """Perform health checks on the bot components."""
        health = {}

        # Check if models are loaded
        health['models_loaded'] = len(self.models) > 0

        health['loaded_indices'] = list(self.models.keys())

        # Check data connectivity by attempting a small fetch
        health['data_connectivity_test'] = {}
        data_test_period = "5d" # Short period for quick check
        all_data_ok = True
        for name, symbol in Config.INDICES.items():
            try:
                df = self.data_fetcher._fetch_yf_download(symbol, data_test_period)
                if df is None or df.empty or len(df) < 1:
                    health['data_connectivity_test'][name] = 'Failed'
                    all_data_ok = False
                else:
                                       health['data_connectivity_test'][name] = 'OK'
            except Exception:
                health['data_connectivity_test'][name] = 'Error'
                all_data_ok = False
        health['all_data_sources_ok'] = all_data_ok

        health['market_open_now'] = self.is_market_open()
        health['timezone'] = str(Config.TIMEZONE)
        health['market_hours'] = f"{Config.MARKET_START} - {Config.MARKET_END}"
        health['config'] = {
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
            'min_volume_ratio': Config.MIN_VOLUME_RATIO,
            'prediction_horizon': Config.PREDICTION_HORIZON,
            'price_movement_threshold': Config.PRICE_MOVEMENT_THRESHOLD,
            'cache_freshness_seconds': Config.CACHE_FRESHNESS_SECONDS,
            'max_fetch_retries': Config.MAX_FETCH_RETRIES,
            'lookback_windows': Config.LOOKBACK_WINDOWS,
        }
        health['model_info_summary'] = self.get_model_info()

        return health

    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about loaded models, including training details."""
        info = {}
        for name, model_pack in self.models.items():
            if isinstance(model_pack, dict):
                info[name] = {
                    'num_features': len(model_pack.get('features', [])),
                    'base_models': list(model_pack.get('base_models', {}).keys()),
                    'has_calibrator': 'calibrator' in model_pack and model_pack['calibrator'] is not model_pack.get('meta_clf'), # Check if it's a real calibrator
                    'cv_scores_count': len(model_pack.get('cv_scores', [])),
                    'avg_cv_auc': (lambda aucs: np.mean(aucs) if aucs else np.nan)([score['auc'] for score in model_pack.get('cv_scores', []) if isinstance(score.get('auc'), (int, float)) and score.get('fold') != 'Overall_OOF']) if model_pack.get('cv_scores') else np.nan,
                    'overall_oof_auc_calibrated': next((score['auc'] for score in model_pack.get('cv_scores', []) if score.get('fold') == 'Overall_OOF'), np.nan),
                    'last_trained': datetime.fromtimestamp(os.path.getmtime(os.path.join(Config.MODEL_DIR, f"{name}_model.joblib"))).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(os.path.join(Config.MODEL_DIR, f"{name}_model.joblib")) else 'N/A'
                }
            else:
                info[name] = {
                    'type': type(model_pack).__name__,
                    'num_features': getattr(model_pack, 'n_features_in_', 'N/A'),
                    'base_models': [],
                    'has_calibrator': False,
                    'cv_scores_count': 0,
                    'avg_cv_auc': np.nan,
                    'overall_oof_auc_calibrated': np.nan,
                    'last_trained': datetime.fromtimestamp(os.path.getmtime(os.path.join(Config.MODEL_DIR, f"{name}_model.joblib"))).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(os.path.join(Config.MODEL_DIR, f"{name}_model.joblib")) else 'N/A'

                }
        return info

def setup_logging():
    """Setup logging configuration."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler = logging.FileHandler(
        os.path.join(Config.LOG_DIR, 'trading_bot.log'),
        mode='a'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)

def print_banner():
    """Print startup banner."""
    banner = """
    ███████╗██╗   ██╗██████╗ ███████╗██████╗     ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗
    ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
    ███████╗██║   ██║██████╔╝█████╗  ██████╔╝       ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝
    ╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗
    ███████║╚██████╔╝██║     ███████╗██║  ██║       ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║
    ╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝

    🚀 Enhanced Trading Bot v3.1.1 - Optimized for Reliability
    ----------------------------------------------------------
    📊 Monitoring Indices: {}
    🎯 Confidence Threshold: {:.1%}
    📈 Volume Ratio Threshold: {:.1f}
    ⏰ Market Hours (IST): {} - {}
    💾 Cache Directory: {}
    📂 Model Directory: {}
    📄 Log Directory: {}
    ----------------------------------------------------------
    """.format(
        ", ".join(Config.INDICES.keys()),
        Config.CONFIDENCE_THRESHOLD,
        Config.MIN_VOLUME_RATIO,
        Config.MARKET_START, Config.MARKET_END,
        Config.CACHE_DIR,
        Config.MODEL_DIR,
        Config.LOG_DIR
    )

    print(banner)
def main():
    """Main function to set up and run the bot."""
    setup_logging()
    logger = logging.getLogger('Main')
    print_banner()

    try:
        logger.info("Application started.")
        logger.info("Initializing Trading Bot instance...")
        bot = TradingBot()
        bot.load_or_train_models()
        health_status = bot.health_check()
        logger.info("--- Initial Health Check ---")
        for key, value in health_status.items():
             if isinstance(value, dict):
                 logger.info(f"{key}:")
                 for sub_key, sub_value in value.items():
                     logger.info(f"  {sub_key}: {sub_value}")
             elif isinstance(value, list):
                  logger.info(f"{key}: {', '.join(map(str, value))}")
             else:
                 logger.info(f"{key}: {value}")
        logger.info("--------------------------")

        if not health_status.get('models_loaded', False):
            logger.critical("Bot requires loaded models to run. Exiting.")
            print("\n\n🚨 Bot failed to load or train models successfully. Check logs for details. Exiting.")
            return
        logger.info("Starting the main trading loop.")
        bot.run(poll_interval=300)

    except Exception as e:
        logger.critical(f"Critical unexpected error in main execution: {str(e)}", exc_info=True)
        print(f"\n\n🚨 A critical error occurred: {str(e)}. Check logs for details. Exiting.")

    finally:
        logger.info("Application finished.")
        print("\nBot process ended.")
if __name__ == "__main__":
    main()