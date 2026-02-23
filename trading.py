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
            "daily_trades": 0
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

    def update_daily_stats(self, pnl: float):
        """Update daily P&L and trade count."""
        # Check if it's a new day to reset
        last_update = self.state.get("last_update")
        if last_update:
            last_date = datetime.fromisoformat(last_update).date()
            if datetime.now().date() > last_date:
                self.state["daily_pnl"] = 0.0
                self.state["daily_trades"] = 0
        
        self.state["daily_pnl"] += pnl
        self.state["daily_trades"] += 1
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
    CONFIDENCE_THRESHOLD = 0.52
    FORCE_RETRAIN = False  # Set to True to force model retraining
    MIN_VOLUME_RATIO = 0.5   # Minimum volume ratio vs 20-day MA for signal consideration
    LOOKBACK_WINDOWS = [5, 10, 20, 50, 100] # Intraday-friendly windows (bars)
    PREDICTION_HORIZON = 12    # Predict movement over the next N bars (5m -> ~1 hour)
    PRICE_MOVEMENT_THRESHOLD = 0.0015 # 0.15% move threshold for intraday target


    # Behavior toggles
    RETRAIN_ON_FEATURE_MISMATCH = False  # If True, retrain when saved model expects features not present

    # Signal/logic parameters
    FORCE_RETRAIN = True          # Retrain models on startup (set to False after successful run)

    USE_EOD_SIGNALS = False         # If True, only act on closed daily candles (after market close)
    AGREEMENT_STD_MAX = 0.35        # Max std dev of base model probabilities to consider consensus (more relaxed)
    ADX_MIN = 12.0                  # Minimal ADX to accept a trend-following trade (more relaxed)
    CONFIRMATION_FILTERS = False    # Temporarily disable extra confirmations to increase signals
    MIN_ATR_RATIO = 0.002           # Minimal ATR/Price ratio (more permissive)
    USE_FALLBACK_TREND_SIGNALS = True  # If ML is neutral, use trend/RSI fallback to emit a signal

    # Confirmation/cooldown toggles
    CONFIRMATION_FILTERS = True     # Use RSI/SMA/MACD/VWAP confirmations
    ENABLE_VWAP_CONFIRMATION = True
    ENABLE_MACD_CONFIRMATION = True
    USE_BAR_CLOSE_ONLY = True       # Only act on closed bars (for 15m intraday)
    COOLDOWN_BARS = 1               # Avoid multiple signals on same/new bar


    # Data fetching parameters
    YFINANCE_TIMEOUT = 10
    INVESTING_TIMEOUT = 15
    # Interval for OHLCV (e.g., '1d', '15m', '5m'). Switching to 5m intraday
    INTERVAL = "5m"
    DATA_PERIOD_TRAINING = "60d"   # For 5m, Yahoo supports ~60 days max
    DATA_PERIOD_SIGNAL = "30d"     # Signals use recent window to stay fast
    CACHE_FRESHNESS_SECONDS = 300 # 5 minutes
    MAX_FETCH_RETRIES = 5         # Retries for data fetching
    POLL_INTERVAL = 300           # Seconds to wait between cycles
    
    # ===== NEW: Enhanced Features Configuration =====
    # Risk Management
    USE_RISK_MANAGER = True
    ACCOUNT_BALANCE = 100000.0  # Initial capital
    RISK_PER_TRADE = 0.015      # 1.5% risk per trade
    MAX_DAILY_LOSS = 0.03       # 3% max daily loss
    
    # Performance Tracking
    USE_PERFORMANCE_TRACKER = True
    PERFORMANCE_DB_PATH = os.path.join(BASE_DIR, "trading_performance.db")
    
    # Regime Detection
    USE_REGIME_DETECTION = True
    REGIME_ADX_THRESHOLD = 25.0
    REGIME_LOOKBACK = 20 # Lowered from 100 for intraday responsiveness
    
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
                # Ensure standard column names and drop potential multi-index
                df.columns = [col.replace(' ', '_') for col in df.columns]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                # Explicitly drop 'Adj Close' if present, as we use 'Close'
                if 'Adj_Close' in df.columns:
                     df = df.drop(columns=['Adj_Close'])
                return df[['Open', 'High', 'Low', 'Close', 'Volume']] # Standardize output columns
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
        except Exception as e:
            self.logger.debug(f"Direct YF API failed for {symbol}: {type(e).__name__} - {e}")
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
                        min_data_points = max(Config.LOOKBACK_WINDOWS) + Config.PREDICTION_HORIZON + 10 # Add buffer
                        if len(df) >= min_data_points:
                             self.logger.info(f"Successfully fetched {len(df)} days for {symbol} using {fetch_method.__name__}")
                             self._save_cache(symbol, df)
                             return df
                        else:
                            self.logger.warning(f"Fetched only {len(df)} days from {fetch_method.__name__}, insufficient for processing.")
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
            feature_dict = {}  # Dictionary to collect all features

            # Keep original OHLCV columns in the feature dictionary first
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                feature_dict[col] = df[col]  # Add OHLCV columns to feature_dict

            # --- Basic TA features from 'ta' library ---
            try:
                df = df.sort_index()
                ta_df = add_all_ta_features(
                    df, open="Open", high="High",
                    low="Low", close="Close",
                    volume="Volume", fillna=True
                )
                # Add TA features to dictionary
                for col in ta_df.columns:
                    if col not in required_cols:
                        feature_dict[col] = ta_df[col]
            except Exception as e:
                logging.warning(f"TA library feature creation failed: {e}")

            # --- Custom features ---

            # Volatility features
            try:
                # Calculate range and TR
                feature_dict['range'] = df['High'] - df['Low']
                feature_dict['tr'] = np.maximum(
                    df['High'] - df['Low'],
                    np.maximum(
                        abs(df['High'] - df['Close'].shift(1)),
                        abs(df['Low'] - df['Close'].shift(1))
                    )
                )

                # Range ratio
                feature_dict['range_ratio'] = np.where(
                    df['Close'].shift(1) != 0,
                    feature_dict['range'] / df['Close'].shift(1),
                    0
                )

                # ATR and ratios
                atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14, fillna=True)
                feature_dict["atr"] = atr.average_true_range()
                feature_dict["atr_ratio"] = np.where(
                    df["Close"] != 0,
                    feature_dict["atr"] / df["Close"],
                    0
                )

                # Bollinger Bands
                bb = BollingerBands(df["Close"], window=20, window_dev=2, fillna=True)
                bb_mavg = bb.bollinger_mavg()
                feature_dict["bb_width"] = np.where(
                    bb_mavg != 0,
                    (bb.bollinger_hband() - bb.bollinger_lband()) / bb_mavg,
                    0
                )
                feature_dict["bb_position"] = np.where(
                    (bb.bollinger_hband() - bb.bollinger_lband()) != 0,
                    (df["Close"] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband()),
                    0.5
                )
            except Exception as e:
                logging.warning(f"Custom volatility features failed: {e}")

            # Trend strength
            try:
                adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=14, fillna=True)
                feature_dict["adx"] = adx.adx()
                feature_dict["trend_strength_adx"] = np.where(feature_dict["adx"] > 25, 1, 0)
            except Exception as e:
                logging.warning(f"Custom ADX feature failed: {e}")

            # Price action patterns
            try:
                feature_dict["body"] = df["Close"] - df["Open"]
                feature_dict["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
                feature_dict["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
                feature_dict["full_range"] = df["High"] - df["Low"]

                # Calculate ratios with safe division
                feature_dict["body_ratio"] = np.where(
                    feature_dict["full_range"] > 0,
                    abs(feature_dict["body"]) / feature_dict["full_range"],
                    0
                )
                feature_dict["upper_shadow_ratio"] = np.where(
                    feature_dict["full_range"] > 0,
                    feature_dict["upper_shadow"] / feature_dict["full_range"],
                    0
                )
                feature_dict["lower_shadow_ratio"] = np.where(
                    feature_dict["full_range"] > 0,
                    feature_dict["lower_shadow"] / feature_dict["full_range"],
                    0
                )

                # Candlestick patterns - ensure all are converted to int
                feature_dict["doji"] = (feature_dict["body_ratio"] <= 0.1).astype(int)

                # Calculate hammer pattern
                feature_dict["hammer"] = np.where(
                    (feature_dict["lower_shadow_ratio"] > 2 * feature_dict["body_ratio"]) &
                    (feature_dict["upper_shadow_ratio"] <= 0.2),  # Simplified condition
                    1,
                    0
                )

                # Calculate shooting star pattern
                feature_dict["shooting_star"] = np.where(
                    (feature_dict["upper_shadow_ratio"] > 2 * feature_dict["body_ratio"]) &
                    (feature_dict["lower_shadow_ratio"] <= 0.2),  # Simplified condition
                    1,
                    0
                )
            except Exception as e:
                logging.warning(f"Price action features failed: {e}")

            # Volume analysis
            try:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
                feature_dict["volume_ma_20d"] = df["Volume"].rolling(window=20, min_periods=1).mean()

                # Modified volume ratio calculation to handle zero values
                volume_ratio = np.where(
                    (feature_dict["volume_ma_20d"] > 0) & (df["Volume"] > 0),
                    df["Volume"] / feature_dict["volume_ma_20d"],
                    1.0
                )
                feature_dict["volume_ratio_20d"] = volume_ratio

                # Fix: Explicitly add the volume_spike feature to feature_dict
                # Previous code created volume_spikes array but didn't properly add it to feature_dict
                feature_dict["volume_spike"] = np.where(volume_ratio > Config.MIN_VOLUME_RATIO, 1, 0)

            except Exception as e:
                logging.warning(f"Volume analysis features failed: {e}")
                # Initialize features with default values on error
                feature_dict["volume_ma_20d"] = df["Volume"].rolling(window=20, min_periods=1).mean().fillna(0)
                feature_dict["volume_ratio_20d"] = pd.Series(1.0, index=df.index)
                # Fix: Ensure volume_spike gets a default value even on error
                feature_dict["volume_spike"] = pd.Series(0, index=df.index)

            # Multiple timeframe features
            try:
                for window in Config.LOOKBACK_WINDOWS:
                    # Returns
                    feature_dict[f"return_{window}d"] = df["Close"].pct_change(periods=window)

                    # Volatility
                    feature_dict[f"volatility_{window}d"] = df["Close"].pct_change().rolling(
                        window=window,
                        min_periods=1
                    ).std()

                    # Price to MA ratio
                    ma = df["Close"].rolling(window=window, min_periods=1).mean()
                    feature_dict[f"price_to_ma_{window}d"] = np.where(ma > 0, df["Close"] / ma - 1, 0)

                    # Volume ratios
                    feature_dict[f"volume_ma_{window}d"] = df["Volume"].rolling(
                        window=window,
                        min_periods=1
                    ).mean()
                    feature_dict[f"volume_ratio_{window}d"] = np.where(
                        feature_dict[f"volume_ma_{window}d"] > 0,
                        df["Volume"] / feature_dict[f"volume_ma_{window}d"],
                        1.0
                    )
            except Exception as e:
                logging.warning(f"Multiple timeframe features failed: {e}")


            # Create final DataFrame all at once with OHLCV included
            feature_df = pd.DataFrame(feature_dict, index=df.index)

            # Final cleanup
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            # Forward-fill only to avoid future leakage; drop remaining NaNs or set conservative defaults
            feature_df = feature_df.fillna(method='ffill')
            feature_df = feature_df.fillna(0)

            preserved_cols = required_cols + ['volume_spike']
            non_preserved_cols = [col for col in feature_df.columns if col not in preserved_cols]
            constant_cols = [col for col in non_preserved_cols if feature_df[col].nunique() <= 1]
            feature_df = feature_df.drop(columns=constant_cols)

            logging.info(f"Feature creation successful. Final DataFrame shape: {feature_df.shape}")
            return feature_df
        except Exception as e:
            logging.error(f"Overall Feature creation failed: {str(e)}")
            return pd.DataFrame()

# ================== Meta Probability Estimator ==================
class MetaProbEstimator(BaseEstimator, ClassifierMixin):
    """
    Wrapper estimator for meta-classifier that can be pickled.
    This class wraps the meta-classifier to provide probability predictions
    for calibration purposes.
    """
    _estimator_type = "classifier"
    
    def __init__(self, meta_clf=None):
        self.meta_clf = meta_clf
        if hasattr(meta_clf, 'classes_'):
            self.classes_ = meta_clf.classes_
        else:
            self.classes_ = np.array([0, 1])  # Default binary

    def predict_proba(self, X):
        """Predict class probabilities using the meta-classifier."""
        return self.meta_clf.predict_proba(X)

    def fit(self, X, y):
        """Fit method required by sklearn interface."""
        self.classes_ = np.unique(y)
        return self

# ================== Model Training ==================
class ModelTrainer:
    """Trains and calibrates the ensemble model."""
    def __init__(self):
        self.logger = logging.getLogger('ModelTrainer')
        self.feature_columns = None
        self.base_models = {} # Initialize here to check availability

        # Initialize base models based on imports
        if LGBMClassifier:
             self.base_models["lgb"] = LGBMClassifier(
                 n_estimators=500, learning_rate=0.01, num_leaves=32, subsample=0.8,
                 colsample_bytree=0.8, objective="binary", n_jobs=-1, random_state=42, verbosity=-1
             )
        if XGBClassifier:
             self.base_models["xgb"] = XGBClassifier(
                 n_estimators=500, learning_rate=0.01, max_depth=6, subsample=0.8,
                 colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
                 n_jobs=-1, random_state=42, verbosity=0
             )
        if CatBoostClassifier:
             self.base_models["cat"] = CatBoostClassifier(
                 iterations=500, learning_rate=0.01, depth=6, loss_function="Logloss",
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

            # Drop any remaining NaNs in target column (already done in the revised workflow)
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

            # Final check for NaNs/Infinities in features just before training
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isna().any().any():
                 self.logger.warning(f"NaNs found in features before training. Imputing...")
                 X = X.fillna(X.median()) # Use median imputation as a fallback

            self.feature_columns = feature_cols
            self.logger.info(f"Prepared data for training: {len(X)} samples, {len(feature_cols)} features. Target distribution:\n{y.value_counts()}")

            # Check class balance again
            class_counts = y.value_counts(normalize=True)
            if len(class_counts) < 2 or min(class_counts) < 0.05: # Warn if minority class is less than 5%
                 self.logger.warning(f"Significant class imbalance: {class_counts.to_dict()}")


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
            oof_meta_features = np.zeros((len(X), len(self.base_models)))
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

                    fold_meta_val = np.zeros((len(val_idx), len(self.base_models)))

                    for i, (name, model) in enumerate(self.base_models.items()):
                        try:
                            import copy
                            fold_model = copy.deepcopy(model)

                            # Check if it's a simple GB model or a pipeline before fitting
                            if isinstance(fold_model, (LGBMClassifier, XGBClassifier, CatBoostClassifier)):
                                # Use early stopping if validation set is large enough
                                if len(X_val) > 100: # Arbitrary threshold for early stopping
                                     eval_set = [(X_val, y_val)]
                                     fit_params = {"eval_set": eval_set, "early_stopping_rounds": 50, "verbose": False}
                                     # Check if model supports early stopping args
                                     if hasattr(fold_model, 'fit'):
                                         import inspect
                                         fit_signature = inspect.signature(fold_model.fit)
                                         if 'eval_set' in fit_signature.parameters and 'early_stopping_rounds' in fit_signature.parameters:
                                             fold_model.fit(X_train, y_train, **fit_params)
                                         else:
                                            fold_model.fit(X_train, y_train) # Fit without early stopping
                                else:
                                    fold_model.fit(X_train, y_train)

                            elif isinstance(fold_model, Pipeline):
                                 fold_model.fit(X_train, y_train)
                            else: # Generic case
                                 fold_model.fit(X_train, y_train)

                            # Store predictions for the meta-classifier training (OOF predictions)
                            if hasattr(fold_model, 'predict_proba'):
                                fold_meta_val[:, i] = fold_model.predict_proba(X_val)[:, 1]
                            else:
                                # Fallback for models without predict_proba (though our list has them)
                                fold_meta_val[:, i] = fold_model.predict(X_val) # This is less ideal for calibration

                        except Exception as e:
                            self.logger.warning(f"Base model '{name}' failed in Fold {fold+1}: {type(e).__name__} - {e}")
                            # Fill failed predictions with 0.5 (random guess)
                            fold_meta_val[:, i] = 0.5

                    # Append OOF predictions and indices
                    oof_meta_features[val_idx] = fold_meta_val
                    oof_indices.extend(val_idx)
                    temp_meta_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
                    temp_meta_clf.fit(fold_meta_val, y_val)
                    fold_val_pred_proba = temp_meta_clf.predict_proba(fold_meta_val)[:, 1]

                    # Calculate metrics for the fold
                    try:
                        auc = roc_auc_score(y_val, fold_val_pred_proba)
                        # Use a reasonable threshold (e.g., 0.5 or Config threshold) for precision/recall for reporting
                        binary_pred = (fold_val_pred_proba > Config.CONFIDENCE_THRESHOLD).astype(int)
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

            # Dictionary to hold OOF predictions by index for each base model
            oof_preds_dict = {name: pd.Series(index=X_indexed.index, dtype=float) for name in self.base_models.keys()}

            # List to store metrics for each fold
            cv_scores_recalc = []

            # Iterate through folds to train base models and get OOF predictions
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_indexed)):
                 try:
                    self.logger.info(f"Base Model OOF Training Fold {fold+1}/{n_splits}")
                    X_train, X_val = X_indexed.iloc[train_idx], X_indexed.iloc[val_idx]
                    y_train, y_val = y_indexed.iloc[train_idx], y_indexed.iloc[val_idx]

                    if len(y_val) < 20 or len(y_val.unique()) < 2:
                         self.logger.warning(f"Skipping Base Model OOF Fold {fold+1} due to insufficient validation data or single class.")
                         continue

                    fold_val_indices = X_val.index # Get original indices

                    for name, model in self.base_models.items():
                        try:
                             # Clone model for independent fold training
                            import copy
                            fold_model = copy.deepcopy(model)

                            # Use early stopping where applicable
                            if isinstance(fold_model, (LGBMClassifier, XGBClassifier, CatBoostClassifier)) and len(X_val) > 100:
                                eval_set = [(X_val, y_val)]
                                fit_params = {"eval_set": eval_set, "early_stopping_rounds": 50, "verbose": False}
                                import inspect
                                fit_signature = inspect.signature(fold_model.fit)
                                if 'eval_set' in fit_signature.parameters and 'early_stopping_rounds' in fit_signature.parameters:
                                    fold_model.fit(X_train, y_train, **fit_params)
                                else:
                                   fold_model.fit(X_train, y_train) # Fit without early stopping
                            elif hasattr(fold_model, 'fit'):
                                 fold_model.fit(X_train, y_train)
                            else:
                                self.logger.warning(f"Model '{name}' in Fold {fold+1} is not trainable.")
                                continue # Skip prediction if not trainable

                            # Store OOF predictions for this fold's validation set
                            if hasattr(fold_model, 'predict_proba'):
                                fold_preds = fold_model.predict_proba(X_val)[:, 1]
                                oof_preds_dict[name].loc[fold_val_indices] = fold_preds
                            else:
                                # Fallback - less ideal
                                fold_preds = fold_model.predict(X_val)
                                oof_preds_dict[name].loc[fold_val_indices] = fold_preds
                            self.logger.debug(f"Obtained OOF preds for model '{name}' in Fold {fold+1}")

                        except Exception as e:
                            self.logger.warning(f"Base model '{name}' prediction failed in Fold {fold+1}: {type(e).__name__} - {e}")
                            # Leave NaN in oof_preds_dict for this model/index


                 except Exception as e:
                     self.logger.error(f"An error occurred during Base Model OOF Fold {fold+1}: {type(e).__name__} - {e}")
                     continue

            # Combine OOF predictions into a DataFrame
            oof_meta_df = pd.DataFrame(oof_preds_dict)

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

            # --- Probability Calibration on Meta-Classifier OOF predictions ---
            try:
                # Get predictions from the trained meta-classifier on the same OOF data
                oof_meta_preds_proba = meta_clf.predict_proba(oof_meta_df)[:, 1]

                # Use CalibratedClassifierCV with method='isotonic' on the OOF meta predictions
                # 'prefit' means it uses the already trained `meta_clf`
                calibrator = CalibratedClassifierCV(meta_clf, cv='prefit', method='isotonic')

                # from sklearn.isotonic import CalibratedClassifierCV as IsotonicCalibrator # Use the class name for clarity
                from sklearn.calibration import CalibratedClassifierCV as IsotonicCalibrator
                
                # Instantiate the module-level MetaProbEstimator wrapping the trained meta_clf
                meta_prob_estimator = MetaProbEstimator(meta_clf)

                calibrator = IsotonicCalibrator(meta_prob_estimator, cv='prefit', method='isotonic')

                # Fit the calibrator. Since cv='prefit', it uses meta_prob_estimator to predict on X=oof_meta_df
                # and fits the calibrator mapping these probabilities to y=oof_labels_aligned.
                calibrator.fit(oof_meta_df, oof_labels_aligned)

                self.logger.info("Probability calibrator trained successfully on meta-classifier OOF predictions.")

                 # Evaluate meta + calibrated model on OOF predictions
                calibrated_oof_preds = calibrator.predict_proba(oof_meta_df)[:, 1]

                try:
                    cal_auc = roc_auc_score(oof_labels_aligned, calibrated_oof_preds)
                    cal_binary_pred = (calibrated_oof_preds > Config.CONFIDENCE_THRESHOLD).astype(int)
                    cal_precision = precision_score(oof_labels_aligned, cal_binary_pred, zero_division=0)
                    cal_recall = recall_score(oof_labels_aligned, cal_binary_pred, zero_division=0)

                    self.logger.info(f"Calibrated OOF Metrics (Meta+Calibrator): AUC={cal_auc:.3f}, Precision={cal_precision:.3f}, Recall={cal_recall:.3f}")

                    # Add these overall OOF scores to CV scores list or store separately
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
                self.logger.error(f"Probability calibration failed: {type(e).__name__} - {e}")
                 # Use the uncalibrated meta-classifier as a fallback
                calibrator = meta_clf
                self.logger.warning("Using uncalibrated meta-classifier for final predictions.")


            # --- Train Final Base Models on the entire dataset ---
            # These are used for prediction on new data, not for training the meta-classifier
            final_base_models = {}
            for name, model in self.base_models.items():
                 try:
                    self.logger.info(f"Training final base model '{name}' on full dataset.")
                    import copy
                    final_model = copy.deepcopy(model) # Clone for final training

                    if hasattr(final_model, 'fit'):
                         # For GB models, training on the full dataset might not need early stopping
                         # as there's no separate validation set here.
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


            # Final model package
            model_pack = {
                'base_models': final_base_models, # Base models trained on full data
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
        # Risk Manager
        if Config.USE_RISK_MANAGER and RiskManager is not None:
            self.risk_manager = RiskManager(
                account_balance=Config.ACCOUNT_BALANCE
            )
            self.logger.info(f"Risk Manager initialized with balance: ₹{Config.ACCOUNT_BALANCE:,.2f}")
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

        # State Management
        self.state = TradingState()
        self.logger.info("Trading State initialized.")

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

    def _calculate_target(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the target variable (future price movement)."""
        if df.empty or len(df) < Config.PREDICTION_HORIZON + 1:
             self.logger.warning("Insufficient data for target calculation.")
             return pd.Series()

        try:
            # Ensure 'Close' column is present and numeric
            if 'Close' not in df.columns:
                 self.logger.error("Close column missing for target calculation.")
                 return pd.Series()

            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close']) # Drop rows where Close is NaN

            if len(df) < Config.PREDICTION_HORIZON + 1:
                 self.logger.warning("Insufficient data after dropping NaN Close values for target calculation.")
                 return pd.Series()

            horizon = Config.PREDICTION_HORIZON
            threshold = Config.PRICE_MOVEMENT_THRESHOLD

            # Calculate forward returns
            fwd_returns = df["Close"].shift(-horizon) / df["Close"] - 1

            target = np.where(fwd_returns > threshold, 1.0,
                            np.where(fwd_returns < -threshold, 0.0, np.nan))

            return pd.Series(target, index=df.index, name='target')

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

            if feature_df.empty or len(feature_df) < len(raw_df):
                 self.logger.error(f"Feature creation failed or lost data for {name}. Original: {len(raw_df)}, Features: {len(feature_df)}")
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
                    avg_auc_overall = np.mean([score['auc'] for score in model_pack['cv_scores'] if isinstance(score['auc'], (int, float)) and score.get('fold') != 'Overall_OOF'])
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
                return None

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
                return None

            latest_date = feature_df.index[-1]
            current_features_row = feature_df.iloc[[-1]].copy()
            raw_df.index = pd.to_datetime(raw_df.index)
            current_price = raw_df.loc[latest_date, 'Close']

            # Get ATR
            atr = 0
            if 'atr' in current_features_row.columns:
                 atr = float(current_features_row['atr'].iloc[0])
            elif 'atr_ratio' in current_features_row.columns:
                 atr = float(current_features_row['atr_ratio'].iloc[0]) * current_price
            else:
                 atr = current_price * 0.01 # Fallback

            # 4. Neural/ML Prediction
            required_features = model_pack.get("features", [])
            current_features_row = current_features_row.reindex(columns=required_features, fill_value=0.0)
            X_pred = current_features_row[required_features]
            X_pred = X_pred.fillna(0).replace([np.inf, -np.inf], 0)

            base_models = model_pack.get("base_models", {})
            meta_features = np.zeros((1, len(base_models)))
            
            for i, (mname, model) in enumerate(base_models.items()):
                try:
                    meta_features[0, i] = 0.5
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_pred)
                        if prob.shape[1] == 2:
                            meta_features[0, i] = prob[0, 1]
                        elif prob.shape[1] == 1:
                            meta_features[0, i] = prob[0, 0]
                except Exception as e:
                     self.logger.warning(f"Base model {mname} prediction failed: {e}")
                     pass

            calibrator = model_pack.get("calibrator") or model_pack.get("meta_clf")
            probability = 0.5
            if calibrator:
                try:
                    probability = calibrator.predict_proba(meta_features)[0, 1]
                except:
                    probability = 0.5

            # 5. Core Direction Logic
            confidence_threshold = regime_params.get('confidence_threshold', Config.CONFIDENCE_THRESHOLD)
            direction = None
            if probability >= confidence_threshold:
                direction = "CE"
            elif probability <= (1.0 - confidence_threshold):
                direction = "PE"
            
            if not direction:
                 return None

            # Determine Direction for logging
            log_direction = "\033[92mCE\033[0m" if direction == "CE" else "\033[91mPE\033[0m"
            self.logger.info(f"ML Direction: {log_direction} (Conf: {probability:.3f})")

            # 6. Signal Validation (Deep Analysis)
            if getattr(self, 'signal_validator', None):
                context = {'regime': regime, 'regime_params': regime_params}
                temp_signal = {'direction': direction, 'price': current_price, 'confidence': probability}
                
                is_valid, score, reason = self.signal_validator.validate_trade_setup(
                    signal=temp_signal,
                    df=feature_df,
                    context=context
                )
                
                # Re-assign colored direction for logging
                log_direction = "\033[92mCE\033[0m" if direction == "CE" else "\033[91mPE\033[0m"

                if not is_valid:
                    self.logger.info(f"Signal REJECTED by Validator: {name} {log_direction} | Score: {score:.1f} | Reason: {reason}")
                    # return None <-- CHANGED: Return signal but inactive
                    position_size = 0.0
                else:
                    self.logger.info(f"Signal \033[92mAPPROVED\033[0m by Validator: {name} {log_direction} | Score: {score:.1f} | Reason: {reason}")
                    position_size = 1.0 # Will be recalculated by risk manager
            
            # 7. Risk Management & Exits
            # Even if rejected, we calculate potential exits for display purposes if possible
            # But normally we only calc exits if we are entering. 
            # Let's do a lightweight calculation or use risk manager with size=0
            
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
                risk_amount = pos_info['risk_amount']
                stop_distance = pos_info.get('stop_distance', atr * 2) # Fallback
                
                # Calculate Stops/Targets for Display
                if direction == "CE":
                    exits['stop_loss'] = current_price - stop_distance
                    exits['tp1'] = current_price + (stop_distance * 1.5)
                    exits['tp2'] = current_price + (stop_distance * 3.0)
                else: 
                    exits['stop_loss'] = current_price + stop_distance
                    exits['tp1'] = current_price - (stop_distance * 1.5)
                    exits['tp2'] = current_price - (stop_distance * 3.0)
            
            # 8. Filter Check (Volume, etc - fallback if not using Validator)
            valid_signal = True
            if position_size <= 0:
                valid_signal = False

            # if probability < confidence_threshold: # Double check
            #      valid_signal = False
            #      position_size = 0

            # 9. Construct Final Signal
            signal = {
                'index': name,
                'direction': direction,
                'price': current_price,
                'confidence': probability,
                'time': datetime.now(Config.TIMEZONE).strftime("%H:%M:%S"),
                'regime': regime,
                'atr': atr,
                'position_size': position_size,
                'risk_amount': risk_amount,
                'stop_loss': exits.get('stop_loss'),
                'take_profit_1': exits.get('tp1'),
                'take_profit_2': exits.get('tp2'),
                'validation_score': score if 'score' in locals() else None,
                'validation_reason': reason if 'reason' in locals() else "Low Confidence"
            }
            
            # Print Actionable Signal Card ONLY if approved
            if position_size > 0:
                self.print_signal_card(signal)

            return signal
                


            # 8. TradingView Validation
            if self.tradingview_validator:
                self.tradingview_validator.validate_signal(
                    signal={
                        'index': name,
                        'direction': direction,
                        'price': current_price,
                        'confidence': probability,
                        'time': datetime.now(Config.TIMEZONE).strftime("%H:%M:%S"),
                        'atr_ratio': atr / current_price if current_price > 0 else 0,
                        'regime': regime
                    },
                    symbol=symbol,
                    interval=Config.INTERVAL
                )

            # 9. Construct Final Signal
            # Calculate Strike and Expiry
            strike = self._get_atm_strike(name, current_price)
            expiry = self._get_next_expiry(name)
            contract_name = f"{name} {expiry.strftime('%d %b').upper()} {strike} {direction}"
            
            signal = {
                'index': name,
                'direction': direction,
                'price': current_price,
                'confidence': probability,
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
                'strike': strike,
                'expiry': expiry.strftime('%d %b %Y'),
                'contract': contract_name
            }
            
            # Print Actionable Signal Card ONLY if approved
            if position_size > 0:
                self.print_signal_card(signal)

            return signal

            return signal

        except Exception as e:
            self.logger.error(f"Signal generation failed for {name} ({symbol}): {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def print_signal_card(self, signal: Dict):
        """Prints a beautifully formatted, colored signal card to the console."""
        
        name = signal['index']
        direction = signal['direction']
        price = signal['price']
        sl = signal['stop_loss']
        tp1 = signal['take_profit_1']
        tp2 = signal['take_profit_2']
        conf = signal['confidence'] * 100
        score = signal.get('validation_score', 0)
        contract = signal.get('contract', 'N/A')
        
        # Color definitions
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        border_color = GREEN if direction == "CE" else RED
        action_color = GREEN if direction == "CE" else RED
        action_text = "BUY CALL (CE)" if direction == "CE" else "BUY PUT (PE)"
        
        print(f"\n{border_color}" + "="*60 + f"{RESET}")
        print(f"{border_color}║ {BOLD}⚡ TRADE SIGNAL ALERT: {name:<30}{RESET} {border_color}║{RESET}")
        print(f"{border_color}" + "="*60 + f"{RESET}")
        print(f"{border_color}║ {RESET}Action:      {action_color}{BOLD}{action_text:<38}{RESET} {border_color}║{RESET}")
        print(f"{border_color}║ {RESET}Entry Price: {CYAN}{price:<.2f}{RESET}{' '*30} {border_color}║{RESET}")
        print(f"{border_color}║ {RESET}Stop Loss:   {YELLOW}{sl:<.2f}{RESET} (Structural/ATR){' '*16} {border_color}║{RESET}")
        print(f"{border_color}║ {RESET}Target 1:    {GREEN}{tp1:<.2f}{RESET}{' '*30} {border_color}║{RESET}")
        print(f"{border_color}║ {RESET}Target 1:    {GREEN}{tp1:<.2f}{RESET}{' '*30} {border_color}║{RESET}")
        print(f"{border_color}║ {RESET}Target 2:    {GREEN}{tp2:<.2f}{RESET} (Runner){' '*21} {border_color}║{RESET}")
        print(f"{border_color}║ {RESET}Contract:    {BOLD}{contract:<38}{RESET} {border_color}║{RESET}")
        print(f"{border_color}║ {RESET}Confidence:  {conf:<.1f}% | Score: {score:.1f}/100{' '*16} {border_color}║{RESET}")
        print(f"{border_color}" + "="*60 + f"{RESET}\n")

    def _get_atm_strike(self, index_name: str, spot_price: float) -> int:
        """Calculate ATM strike price based on index."""
        if "BANK" in index_name:
            # Round to nearest 100
            return int(round(spot_price / 100) * 100)
        else:
            # NIFTY and FINNIFTY round to nearest 50
            return int(round(spot_price / 50) * 50)

    def _get_next_expiry(self, index_name: str) -> datetime:
        """
        Get the next expiry date based on new SEBI rules (Nov 2024).
        - NIFTY: Weekly (Thursday)
        - BANKNIFTY: Monthly Only (Last Wednesday of Month)
        - FINNIFTY: Monthly Only (Last Tuesday of Month)
        """
        today = datetime.now(Config.TIMEZONE).date()
        
        # 1. NIFTY - Weekly Expiry (Thursday)
        if index_name == "NIFTY":
            target_weekday = 3 # Thursday
            days_ahead = target_weekday - today.weekday()
            if days_ahead <= 0: # Target day already happened this week
                if days_ahead == 0 and datetime.now(Config.TIMEZONE).hour >= 15:
                     days_ahead += 7
                elif days_ahead < 0:
                     days_ahead += 7
            return today + timedelta(days=days_ahead)

        # 2. Monthly Expiry Logic (BankNifty/FinNifty)
        # Find the last occurrence of specific weekday in the current month
        year = today.year
        month = today.month
        
        if "BANK" in index_name:
            target_weekday = 2 # Wednesday
        elif "FIN" in index_name:
            target_weekday = 1 # Tuesday
        
        def get_last_weekday_of_month(y, m, weekday):
            # Start from the last day of the month and move back
            if m == 12:
                next_month = datetime(y + 1, 1, 1)
            else:
                next_month = datetime(y, m + 1, 1)
            
            last_day = next_month - timedelta(days=1)
            
            # Move back until we find the target weekday
            while last_day.weekday() != weekday:
                last_day -= timedelta(days=1)
            return last_day.date()
        
        current_month_expiry = get_last_weekday_of_month(year, month, target_weekday)
        
        # If today is past the monthly expiry (or too late on expiry day), move to next month
        if today > current_month_expiry or (today == current_month_expiry and datetime.now(Config.TIMEZONE).hour >= 15):
             if month == 12:
                 current_month_expiry = get_last_weekday_of_month(year + 1, 1, target_weekday)
             else:
                 current_month_expiry = get_last_weekday_of_month(year, month + 1, target_weekday)
                 
        return current_month_expiry
        print(f"{border_color}" + "="*60 + f"{RESET}\n")

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
                cooldown_minutes = 30 # Cooldown of 30 minutes after a trade closes
                if time_since_last_signal < pd.Timedelta(minutes=cooldown_minutes):
                    self.logger.info(f"Cooldown active for {name}. Skipping entry.")
                    return {'index': name, 'status': 'Cooldown'}
            # Check daily limits first
            limit_status = self.risk_manager.check_daily_limits() if self.risk_manager else 1.0
            if limit_status == 0.0:
                self.logger.info(f"Daily loss limit reached. Skipping entry for {name}.")
                return {'index': name, 'status': 'Daily Limit Reached'}

            signal = self.generate_signal(name, symbol)
            
            if not signal:
                 return {'index': name, 'status': 'No Signal/Data'}
            
            # Setup for dashboard return
            status = "Waiting"
            if signal.get('position_size', 0) > 0: status = "APPROVED"
            elif 'validation_score' in signal: status = "REJECTED"
            
            if signal and signal.get('position_size', 0) > 0:
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
        """Monitor an open position for exit conditions."""
        try:
            # Get latest price
            # Using 5m data for monitoring to get more granular price action
            # or just fetch 1d if EOD. Let's stick to 1d for consistency with signal gen for now,
            # but in a real system we'd stream live ticks.
            # Using data_fetcher defaults which might be 1d. 
            # Ideally this should check live price.
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
            
            # Check Exit Conditions
            exit_reason = None
            pnl = 0.0
            
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
            
            # Check for Time-based exit or Regime change (optional - future enhancement)
            
            if exit_reason:
                # Calculate P&L
                if direction == "CE":
                    pnl = (current_price - entry_price) * size
                else:
                    pnl = (entry_price - current_price) * size
                
                self.logger.info(f"Closing {direction} position for {name} ({symbol}) at {current_price}. Reason: {exit_reason}. P&L: {pnl:.2f}")
                
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
                        pnl=pnl
                    )
                
                # Update State
                self.state.remove_position(symbol)
                self.state.update_daily_stats(pnl)
                
                # Notify User (via log)
                self.logger.info(f"Trade Closed: {name} {direction} | P&L: {pnl:.2f}")

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
            trade_id = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            if self.performance_tracker:
                 self.performance_tracker.log_trade(
                    trade_id=trade_id,
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
            
            # Save to State
            position_data = {
                "trade_id": trade_id,
                "entry_price": current_price,
                "direction": direction,
                "position_size": size, # Standardized key
                "sl": sl,
                "tp": tp,
                "entry_time": datetime.now().isoformat(),
                "regime": signal['regime']
            }
            self.state.add_position(symbol, position_data)
            
            self.logger.info(f"🚀 OPENED {direction} for {name} @ {current_price} | Size: {size} | SL: {sl} | TP: {tp}")

        except Exception as e:
            self.logger.error(f"Failed to execute entry for {name}: {e}")

    @staticmethod
    def is_market_open() -> bool:
        """Check if market is open (IST)."""
        now = datetime.now(Config.TIMEZONE)
        # Check for weekends (Saturday=5, Sunday=6)
        if now.weekday() >= 5:
            return False

        # Check for market hours
        market_start_time = datetime.strptime(Config.MARKET_START, "%H:%M").time()
        market_end_time = datetime.strptime(Config.MARKET_END, "%H:%M").time()

        # Add a small buffer to avoid issues right at the open/close
        # e.g., check from 09:16 to 15:29
        check_start_time = (datetime.combine(datetime.today(), market_start_time) + timedelta(minutes=1)).time()
        check_end_time = (datetime.combine(datetime.today(), market_end_time) - timedelta(minutes=1)).time()

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
                poll_interval = Config.POLL_INTERVAL
                self.logger.info(f"Lifecycle completed. Sleeping for {poll_interval}s.")
                
                # Provide visual feedback during sleep
                print(f"⏳ Sleeping for {poll_interval}s...", end="", flush=True)
                time.sleep(poll_interval)
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
                    'avg_cv_auc': np.mean([score['auc'] for score in model_pack.get('cv_scores', []) if isinstance(score.get('auc'), (int, float)) and score.get('fold') != 'Overall_OOF']) if model_pack.get('cv_scores') else np.nan,
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