import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from ta.trend import EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import time
import pytz
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTradingBot:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.indices = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'FINNIFTY': 'NIFTY_FIN_SERVICE.NS'
        }
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        self.confidence_threshold = 0.60
        self.position_size = 0.1
        self.stop_loss_pct = 0.02
        self.target_pct = 0.05
        self.data_cache = {}
        self.cache_expiry = 15  # Cache expiry in minutes

    def is_market_open(self):
        """Check if Indian market is open."""
        current_time = datetime.now(self.ist_timezone)
        if current_time.weekday() >= 5:  # Check for weekends
            return False
        market_start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= current_time <= market_end

    def fetch_data(self, symbol, period='5y', interval='1d'):
        """Fetch market data using yfinance with caching."""
        current_time = datetime.now()
        if symbol in self.data_cache and (current_time - self.data_cache[symbol]['time']).total_seconds() / 60 < self.cache_expiry:
            logger.debug(f"Using cached data for {symbol}")
            return self.data_cache[symbol]['data']

        try:
            logger.info(f"Fetching fresh data for {symbol}")
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if not data.empty:
                self.data_cache[symbol] = {'data': data, 'time': current_time}
                return data
            logger.warning(f"No data found for {symbol}.")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
        return None

    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators with proper error handling"""
        if df is None or df.empty or len(df) < 52:
            print("Insufficient data for technical analysis")
            return None
            
        try:
            df = df.copy()
            
            # Fill missing values using forward fill
            df.ffill(inplace=True)  # Forward fill to handle gaps
            
            # Convert to Series for technical indicators
            close_series = df['Close'].squeeze()  # Ensure it's a 1D Series
            high_series = df['High'].squeeze()    # Ensure it's a 1D Series
            low_series = df['Low'].squeeze()      # Ensure it's a 1D Series
            volume_series = df['Volume'].squeeze()  # Ensure it's a 1D Series
            
            # Calculate EMAs
            df['EMA20'] = EMAIndicator(close=close_series).ema_indicator()
            df['EMA50'] = EMAIndicator(close=close_series, window=50).ema_indicator()
            
            # Calculate MACD
            macd_indicator = MACD(close=close_series)
            df['MACD'] = macd_indicator.macd()
            df['MACD_Signal'] = macd_indicator.macd_signal()
            
            # Calculate RSI
            df['RSI'] = RSIIndicator(close=close_series).rsi()
            
            # Calculate Volume Ratio
            mean_volume = volume_series.rolling(window=20).mean()
            df['Volume_Ratio'] = volume_series / mean_volume
            df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], np.nan)
            
            # Ichimoku Cloud
            ichimoku = IchimokuIndicator(high=high_series, low=low_series)
            df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
            df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            
            # Volume Indicators
            acc_dist = AccDistIndexIndicator(high=high_series, low=low_series, close=close_series, volume=volume_series)
            df['ACC_DIST'] = acc_dist.acc_dist_index()
            
            obv = OnBalanceVolumeIndicator(close=close_series, volume=volume_series)
            df['OBV'] = obv.on_balance_volume()
            
            # ATR
            atr = AverageTrueRange(high=high_series, low=low_series, close=close_series)
            df['ATR'] = atr.average_true_range()
            
            # Stochastic
            stoch = StochasticOscillator(high=high_series, low=low_series, close=close_series)
            df['STOCH_K'] = stoch.stoch()
            df['STOCH_D'] = stoch.stoch_signal()

            # Add candlestick patterns
            df = self.add_candlestick_patterns(df)

            # Add price action analysis
            df = self.add_price_action(df)

            # Add liquidity analysis
            df = self.add_liquidity_analysis(df)

            # Replace NaN and inf values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            return df
        except Exception as e:
            logger.error(f"Error in add_technical_indicators: {e}")
            return None

    def add_candlestick_patterns(self, df):
        """Add candlestick pattern indicators to the DataFrame."""
        if df is None or len(df) == 0:
            logger.warning("Insufficient data for candlestick pattern analysis")
            return df

        # Initialize columns for patterns
        df['Doji'] = 0
        df['Hammer'] = 0
        df['Engulfing'] = 0
        df['Shooting_Star'] = 0
        df['Morning_Star'] = 0
        df['Evening_Star'] = 0
        df['Bullish_Harami'] = 0
        df['Bearish_Harami'] = 0
        df['Piercing_Line'] = 0
        df['Dark_Cloud_Cover'] = 0

        # Calculate Doji
        df['Doji'] = np.where(np.abs(df['Open'] - df['Close']) <= (df['High'] - df['Low']) * 0.1, 1, 0)

        # Calculate Hammer (Bullish)
        df['Hammer'] = np.where((df['Close'] > df['Open']) & 
                                 ((df['Low'] - df['Close']) <= 0.25 * (df['High'] - df['Low'])) & 
                                 ((df['Open'] - df['Low']) >= 2 * (df['Close'] - df['Open'])), 1, 0)

        # Calculate Engulfing (Bullish)
        df['Engulfing'] = np.where((df['Close'] > df['Open']) & 
                                    (df['Open'].shift(1) > df['Close'].shift(1)) & 
                                    (df['Open'] < df['Close'].shift(1)) & 
                                    (df['Close'] > df['Open'].shift(1)), 1, 0)

        # Calculate Shooting Star (Bearish)
        df['Shooting_Star'] = np.where((df['Close'] < df['Open']) & 
                                        ((df['High'] - df['Close']) >= 2 * (df['Open'] - df['Close'])) & 
                                        ((df['Open'] - df['Low']) <= 0.25 * (df['High'] - df['Low'])), 1, 0)

        # Calculate Morning Star (Bullish)
        df['Morning_Star'] = np.where((df['Close'].shift(2) < df['Open'].shift(2)) & 
                                       (df['Close'].shift(1) < df['Open'].shift(1)) & 
                                       (df['Close'] > df['Open']) & 
                                       (df['Open'] < df['Close'].shift(1)), 1, 0)

        # Calculate Evening Star (Bearish)
        df['Evening_Star'] = np.where((df['Close'].shift(2) > df['Open'].shift(2)) & 
                                       (df['Close'].shift(1) > df['Open'].shift(1)) & 
                                       (df['Close'] < df['Open']) & 
                                       (df['Open'] > df['Close'].shift(1)), 1, 0)

        # Calculate Bullish Harami
        df['Bullish_Harami'] = np.where((df['Close'].shift(1) < df['Open'].shift(1)) & 
                                         (df['Close'] > df['Open']) & 
                                         (df['Open'] < df['Close'].shift(1)) & 
                                         (df['Close'] > df['Open'].shift(1)), 1, 0)

        # Calculate Bearish Harami
        df['Bearish_Harami'] = np.where((df['Close'].shift(1) > df['Open'].shift(1)) & 
                                         (df['Close'] < df['Open']) & 
                                         (df['Open'] > df['Close'].shift(1)) & 
                                         (df['Close'] < df['Open'].shift(1)), 1, 0)

        # Calculate Piercing Line (Bullish)
        df['Piercing_Line'] = np.where((df['Close'].shift(1) < df['Open'].shift(1)) & 
                                        (df['Close'] > df['Open']) & 
                                        (df['Open'] < df['Close'].shift(1)) & 
                                        (df['Close'] >= (df['Open'].shift(1) + df['Close'].shift(1)) / 2), 1, 0)

        # Calculate Dark Cloud Cover (Bearish)
        df['Dark_Cloud_Cover'] = np.where((df['Close'].shift(1) > df['Open'].shift(1)) & 
                                           (df['Close'] < df['Open']) & 
                                           (df['Open'] > df['Close'].shift(1)) & 
                                           (df['Close'] <= (df['Open'].shift(1) + df['Close'].shift(1)) / 2), 1, 0)

        return df

    def add_price_action(self, df):
        """Add price action analysis to the DataFrame."""
        if df is None or len(df) == 0:
            logger.warning("Insufficient data for price action analysis")
            return df

        # Calculate support and resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()

        # Calculate price action signals
        df['Price_Action_Buy'] = np.where((df['Close'] > df['Resistance']), 1, 0)
        df['Price_Action_Sell'] = np.where((df['Close'] < df['Support']), 1, 0)

        return df

    def add_liquidity_analysis(self, df):
        """Add liquidity analysis to the DataFrame."""
        if df is None or len(df) == 0:
            logger.warning("Insufficient data for liquidity analysis")
            return df

        # Calculate average volume over the last 20 periods
        df['Avg_Volume'] = df['Volume'].rolling(window=20).mean()

        # Identify liquidity signals
        df['High_Liquidity'] = np.where(df['Volume'] > df['Avg_Volume'] * 1.5, 1, 0)
        df['Low_Liquidity'] = np.where(df['Volume'] < df['Avg_Volume'] * 0.5, 1, 0)

        return df

    def prepare_features(self, df, symbol):
        """Prepare features for model training."""
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series()

        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        feature_columns = ['EMA20', 'EMA50', 'EMA200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                           'ACC_DIST', 'OBV', 'ATR', 'Doji', 'Hammer', 'Engulfing', 'Shooting_Star', 
                           'Morning_Star', 'Evening_Star', 'Bullish_Harami', 'Bearish_Harami', 
                           'Piercing_Line', 'Dark_Cloud_Cover', 'Price_Action_Buy', 'Price_Action_Sell', 
                           'High_Liquidity', 'Low_Liquidity']

        df_features = df[feature_columns + ['Target']].dropna()
        if df_features.empty:
            return pd.DataFrame(), pd.Series()

        X = df_features[feature_columns][:-1]
        y = df_features['Target'][:-1]

        if symbol not in self.scalers:
            self.scalers[symbol] = StandardScaler()
            X_scaled = self.scalers[symbol].fit_transform(X)
        else:
            X_scaled = self.scalers[symbol].transform(X)

        return pd.DataFrame(X_scaled, columns=feature_columns), y

    def train_model(self, symbol):
        """Train model for a specific index."""
        try:
            data = self.fetch_data(symbol)
            if data is None or len(data) < 50:
                logger.warning(f"Insufficient data for {symbol}, skipping model training")
                return False

            df = self.add_technical_indicators(data)
            if df is None:
                return False

            X, y = self.prepare_features(df, symbol)
            if X.empty or y.empty:
                logger.warning(f"No features generated for {symbol}")
                return False

            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', None]
            }
            logger.info(f"Training model for {symbol} with {len(X)} samples")
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X, y)
            self.models[symbol] = grid_search.best_estimator_
            logger.info(f"Best parameters for {symbol}: {grid_search.best_params_}")
            return True
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False

    def predict_index(self, symbol):
        """Generate prediction for a specific index."""
        try:
            if symbol not in self.models:
                if not self.train_model(symbol):
                    return None, 0

            data = self.fetch_data(symbol)
            if data is None:
                return None, 0

            df = self.add_technical_indicators(data)
            if df is None:
                return None, 0

            X, _ = self.prepare_features(df, symbol)
            if X.empty:
                return None, 0

            latest_data = X.iloc[-1:].copy()
            prediction = self.models[symbol].predict(latest_data)[0]
            probability = self.models[symbol].predict_proba(latest_data)[0]

            option_type = 'CE' if prediction == 1 else 'PE'
            confidence = probability[1] if prediction == 1 else probability[0]

            latest_close_price = data['Close'].iloc[-1]
            strike_price = round(latest_close_price / 100) * 100

            return f"{symbol} {strike_price} {option_type}", confidence
        except Exception as e:
            logger.error(f"Error in predict_index for {symbol}: {e}")
            return None, 0

    def analyze_market(self):
        """Analyze market in parallel and generate trade signals."""
        best_signal = None
        highest_confidence = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.indices)) as executor:
            future_to_symbol = {
                executor.submit(self.predict_index, symbol): (index_name, symbol)
                for index_name, symbol in self.indices.items()
            }

            for future in concurrent.futures.as_completed(future_to_symbol):
                index_name, symbol = future_to_symbol[future]
                try:
                    signal, confidence = future.result()
                    if signal and confidence >= self.confidence_threshold and confidence > highest_confidence:
                        highest_confidence = confidence
                        best_signal = f"{signal} (Confidence: {confidence:.2f})"
                except Exception as e:
                    logger.error(f"Error processing {index_name} prediction: {e}")

        return best_signal

    def run_live(self):
        """Run the trading bot live."""
        logger.info("Initializing Enhanced Trading Bot...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.indices)) as executor:
            training_futures = [executor.submit(self.train_model, symbol) for symbol in self.indices.values()]
            for future in concurrent.futures.as_completed(training_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error during model training: {e}")

        logger.info("Starting enhanced market monitoring...")

        while True:
            try:
                if not self.is_market_open():
                    logger.info("Market closed. Checking again in 60 seconds.")
                    time.sleep(60)
                    continue

                start_time = time.time()
                best_signal = self.analyze_market()
                analysis_time = time.time() - start_time

                if best_signal:
                    logger.info(f"SIGNAL: {best_signal} (Analysis took {analysis_time:.2f}s)")
                else:
                    logger.info(f"No trading signal generated (Analysis took {analysis_time:.2f}s)")

                sleep_time = 300 if datetime.now(self.ist_timezone).hour < 10 or datetime.now(self.ist_timezone).hour >= 14 else 180
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Stopping monitoring by user request...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = EnhancedTradingBot()
    bot.run_live()