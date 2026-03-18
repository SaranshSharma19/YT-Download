import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from datetime import datetime, timedelta
from ta.trend import EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, MACD, IchimokuIndicator, VortexIndicator, PSARIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, TSIIndicator
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator, MFIIndicator, EaseOfMovementIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
import time
import pytz
import concurrent.futures
import logging
from sklearn.decomposition import PCA

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
        self.models = {}  # Store models for each index
        self.scalers = {}  # Store scalers for each index
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
        self.data_cache = {}  # Cache for market data
        self.cache_expiry = 15  # Cache expiry in minutes
        self.last_cache_time = {}  # Track when data was last cached

    def is_market_open(self):
        """Check if Indian market is open with efficient time handling"""
        current_time = datetime.now(self.ist_timezone)
        
        # Quick check for weekends
        if current_time.weekday() >= 5:
            return False
            
        # Define market hours
        market_start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= current_time <= market_end

    def fetch_data(self, symbol, period='5y', interval='1d'):
        """Fetch market data using yfinance with caching for optimization"""
        current_time = datetime.now()
        
        # Check if we have cached data that's still valid
        if symbol in self.data_cache and symbol in self.last_cache_time:
            cache_age = (current_time - self.last_cache_time[symbol]).total_seconds() / 60
            if cache_age < self.cache_expiry:
                logger.debug(f"Using cached data for {symbol}, age: {cache_age:.1f} minutes")
                return self.data_cache[symbol]
        
        try:
            print(f"Fetching fresh data for {symbol}")
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if not data.empty:
                data.index = pd.to_datetime(data.index)
                # Cache the result
                self.data_cache[symbol] = data
                self.last_cache_time[symbol] = current_time
                return data
            
            print(f"No data found for {symbol}.")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
        
        return None

    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators with proper error handling"""
        if df is None or len(df) == 0 or len(df) < 52:
            print("Insufficient data for technical analysis")
            return None
            
        try:
            df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
            
            # Log the shape of the DataFrame for debugging
            logger.debug(f"DataFrame shape: {df.shape}")
            
            # Access series directly and ensure they are 1D
            close_series = df['Close'].squeeze()  # Ensure it's 1D
            high_series = df['High'].squeeze()
            low_series = df['Low'].squeeze()
            volume_series = df['Volume'].squeeze()
            open_series = df['Open'].squeeze()
            
            # Check for NaN values
            if (np.isnan(close_series).any() or np.isnan(high_series).any() or 
                np.isnan(low_series).any() or np.isnan(volume_series).any() or 
                np.isnan(open_series).any()):
                logger.error("One or more series contain NaN values.")
                return None
            
            # Trend Indicators
            df['EMA20'] = EMAIndicator(close=close_series, window=20).ema_indicator()
            df['EMA50'] = EMAIndicator(close=close_series, window=50).ema_indicator()
            df['EMA200'] = EMAIndicator(close=close_series, window=200).ema_indicator()
            
            # MACD
            macd_indicator = MACD(close=close_series)
            df['MACD'] = macd_indicator.macd()
            df['MACD_Signal'] = macd_indicator.macd_signal()
            df['MACD_Hist'] = macd_indicator.macd_diff()
            
            # Ichimoku Cloud
            ichimoku = IchimokuIndicator(high=high_series, low=low_series)
            df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
            df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            df['Ichimoku_A'] = ichimoku.ichimoku_a()
            df['Ichimoku_B'] = ichimoku.ichimoku_b()
            
            # Vortex Indicator
            vortex = VortexIndicator(high=high_series, low=low_series, close=close_series)
            df['Vortex_Pos'] = vortex.vortex_indicator_pos()
            df['Vortex_Neg'] = vortex.vortex_indicator_neg()
            df['Vortex_Diff'] = df['Vortex_Pos'] - df['Vortex_Neg']
            
            # Parabolic SAR
            psar = PSARIndicator(high=high_series, low=low_series, close=close_series)
            df['PSAR'] = psar.psar()
            df['PSAR_Up'] = psar.psar_up()
            df['PSAR_Down'] = psar.psar_down()
            
            # Momentum Indicators
            df['RSI'] = RSIIndicator(close=close_series).rsi()
            stoch = StochasticOscillator(high=high_series, low=low_series, close=close_series)
            df['STOCH_K'] = stoch.stoch()
            df['STOCH_D'] = stoch.stoch_signal()
            df['ROC'] = ROCIndicator(close=close_series).roc()
            df['TSI'] = TSIIndicator(close=close_series).tsi()
            
            # Volume Indicators
            df['ACC_DIST'] = AccDistIndexIndicator(high=high_series, low=low_series, close=close_series, volume=volume_series).acc_dist_index()
            df['OBV'] = OnBalanceVolumeIndicator(close=close_series, volume=volume_series).on_balance_volume()
            df['MFI'] = MFIIndicator(high=high_series, low=low_series, close=close_series, volume=volume_series).money_flow_index()
            df['EOM'] = EaseOfMovementIndicator(high=high_series, low=low_series, volume=volume_series).ease_of_movement()
            
            # Volume Ratio - safe division
            volume_mean = df['Volume'].rolling(window=20).mean()
            # Avoid division by zero with numpy where
            volume_ratio = np.where(volume_mean > 0, df['Volume'].values / volume_mean.values, np.nan)
            df['Volume_Ratio'] = volume_ratio
            # Replace inf values
            # df['Volume_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df.loc[:, 'Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], 0) # np.nan
            
            # Volatility Indicators
            df['ATR'] = AverageTrueRange(high=high_series, low=low_series, close=close_series).average_true_range()
            # Avoid division by zero
            df['ATR_Pct'] = np.where(close_series > 0, (df['ATR'] / close_series) * 100, 0)
            
            # Bollinger Bands
            bollinger = BollingerBands(close=close_series)
            df['Bollinger_Upper'] = bollinger.bollinger_hband()
            df['Bollinger_Lower'] = bollinger.bollinger_lband()
            df['Bollinger_Middle'] = bollinger.bollinger_mavg()
            
            # Check if Bollinger Bands were created successfully
            if 'Bollinger_Upper' not in df or 'Bollinger_Lower' not in df:
                logger.error("Bollinger Bands not created successfully.")
                return None
            
            # Calculate width and percentage safely using numpy operations
            upper = df['Bollinger_Upper'].values
            lower = df['Bollinger_Lower'].values
            middle = df['Bollinger_Middle'].values
            close = close_series.values
            
            # Calculate width - avoiding division by zero
            width = np.where(middle > 0, (upper - lower) / middle, 0)
            df['Bollinger_Width'] = width
            
            # Calculate percentage - avoiding division by zero
            upper_lower_diff = upper - lower
            bollinger_pct = np.where(upper_lower_diff > 0, (close - lower) / upper_lower_diff, 0.5)
            df['Bollinger_Pct'] = bollinger_pct
            
            # Candle pattern features
            df['Body_Size'] = np.abs(open_series - close_series)
            df['Upper_Shadow'] = high_series - np.maximum(open_series, close_series)
            df['Lower_Shadow'] = np.minimum(open_series, close_series) - low_series
            df['Body_Size_Pct'] = np.where(close_series > 0, (df['Body_Size'] / close_series) * 100, 0)
            df['Total_Range'] = high_series - low_series
            
            # Daily returns
            df['Daily_Return'] = DailyReturnIndicator(close=close_series).daily_return()
            
            # Pivot Points
            high_prev = high_series.shift(1).values
            low_prev = low_series.shift(1).values
            close_prev = close_series.shift(1).values
            df['Pivot'] = (high_prev + low_prev + close_prev) / 3
            df['R1'] = (2 * df['Pivot']) - low_prev
            df['S1'] = (2 * df['Pivot']) - high_prev
            
            # VWAP
            typical_price = (high_series + low_series + close_series) / 3
            cum_tp_vol = np.cumsum(typical_price * volume_series)
            cum_vol = np.cumsum(volume_series)
            df['VWAP'] = np.where(cum_vol > 0, cum_tp_vol / cum_vol, 0)
            
            # Gap Analysis
            df['Gap'] = open_series - close_series.shift(1)
            close_prev_nonzero = close_series.shift(1) > 0
            df['Gap_Pct'] = np.where(close_prev_nonzero, (df['Gap'] / close_series.shift(1)) * 100, 0)
            
            # Trend Strength
            df['Price_to_EMA20'] = np.where(df['EMA20'] > 0, (close_series / df['EMA20']) - 1, 0)
            df['Price_to_EMA50'] = np.where(df['EMA50'] > 0, (close_series / df['EMA50']) - 1, 0)
            df['Price_to_EMA200'] = np.where(df['EMA200'] > 0, (close_series / df['EMA200']) - 1, 0)
            
            # Moving Average Crosses
            ema20 = df['EMA20'].values
            ema50 = df['EMA50'].values
            ema200 = df['EMA200'].values
            
            # Create boolean masks for valid values
            valid_ema20_ema50 = ~np.isnan(ema20) & ~np.isnan(ema50)
            valid_ema50_ema200 = ~np.isnan(ema50) & ~np.isnan(ema200)
            
            # Initialize with zeros
            ema20_cross_ema50 = np.zeros(len(df))
            ema50_cross_ema200 = np.zeros(len(df))
            
            # Set values only where valid
            ema20_cross_ema50[valid_ema20_ema50] = (ema20[valid_ema20_ema50] > ema50[valid_ema20_ema50]).astype(int)
            ema50_cross_ema200[valid_ema50_ema200] = (ema50[valid_ema50_ema200] > ema200[valid_ema50_ema200]).astype(int)
            
            df['EMA20_Cross_EMA50'] = ema20_cross_ema50
            df['EMA50_Cross_EMA200'] = ema50_cross_ema200
            
            # Historical Volatility
            close_prices = close_series.values
            close_shifted = np.roll(close_prices, 1)
            close_shifted[0] = close_shifted[1]  # Avoid first element issue
            
            # Calculate log returns where both values are positive
            valid_prices = (close_prices > 0) & (close_shifted > 0)
            log_returns = np.zeros(len(close_prices))
            log_returns[valid_prices] = np.log(close_prices[valid_prices] / close_shifted[valid_prices])
            
            # Calculate rolling standard deviation
            window_size = 20
            volatility = np.zeros(len(log_returns))
            
            for i in range(window_size, len(log_returns)):
                window = log_returns[i-window_size:i]
                volatility[i] = np.std(window) * np.sqrt(252) * 100
            
            df['Hist_Volatility'] = volatility
            
            # Replace NaN and inf values
            df = df.replace([np.inf, -np.inf], np.nan)
            df.fillna(0, inplace=True)
            
            # Add candlestick patterns
            df = self.add_candlestick_patterns(df)
            
            return df
                    
        except Exception as e:
            logger.error(f"Error in add_technical_indicators: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def add_candlestick_patterns(self, df):
        """Add candlestick pattern indicators to the DataFrame"""
        if df is None or len(df) == 0:
            print("Insufficient data for candlestick pattern analysis")
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

    def prepare_features(self, df, symbol):
        """Prepare features for model training with scaled data caching"""
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series()
            
        # Create target variable
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Define feature columns
        # feature_columns = [
        #     'EMA20', 'EMA50','EMA200', 'RSI', 
        #     'MACD', 'MACD_Signal', 
        #     'ATR', 'STOCH_K', 'STOCH_D', 
        #     'Ichimoku_Conversion', 'Ichimoku_Base', 
        #     'Volume_Ratio', 'ACC_DIST', 'OBV', 
        #     'ROC', 'Bollinger_Upper', 'Bollinger_Lower'
        # ]
        feature_columns = [
        'EMA20', 'EMA50', 'EMA200', 'RSI', 
        'MACD', 'MACD_Signal', 'MACD_Hist',  # Added MACD Histogram
        'ATR', 'STOCH_K', 'STOCH_D', 
        'Ichimoku_Conversion', 'Ichimoku_Base', 
        'Ichimoku_A', 'Ichimoku_B',  # Added Ichimoku A and B
        'Volume_Ratio', 'ACC_DIST', 'OBV', 
        'ROC', 'Bollinger_Upper', 'Bollinger_Lower', 
        'Bollinger_Width', 'Bollinger_Pct',  # Added Bollinger Width and Percentage
        'Vortex_Pos', 'Vortex_Neg', 'Vortex_Diff',  # Added Vortex indicators
        'PSAR', 'PSAR_Up', 'PSAR_Down',  # Added Parabolic SAR indicators
        'Daily_Return', 'Gap', 'Gap_Pct',  # Added Gap analysis indicators
        'Price_to_EMA20', 'Price_to_EMA50', 'Price_to_EMA200',  # Added Price to EMA indicators
        'EMA20_Cross_EMA50', 'EMA50_Cross_EMA200',  # Added EMA cross indicators
        'Hist_Volatility',  # Added Historical Volatility
        'Doji', 'Hammer', 'Engulfing', 'Shooting_Star', 'Morning_Star', 'Evening_Star', 'Bullish_Harami', 'Bearish_Harami', 'Piercing_Line', 'Dark_Cloud_Cover'  # Added candlestick pattern indicators
        ]
        
        # Drop rows with NaN values
        df_features = df[feature_columns + ['Target']].dropna()
        
        if len(df_features) == 0:
            return pd.DataFrame(), pd.Series()
            
        X = df_features[feature_columns][:-1]  # All rows except last one
        y = df_features['Target'][:-1]  # All rows except last one
        
        # Create or retrieve scaler
        if symbol not in self.scalers:
            self.scalers[symbol] = StandardScaler()
            X_scaled = self.scalers[symbol].fit_transform(X)
        else:
            X_scaled = self.scalers[symbol].transform(X)
        
        return pd.DataFrame(X_scaled, columns=feature_columns), y

    def train_model(self, symbol):
        """Train model for a specific index"""
        try:
            data = self.fetch_data(symbol)
            if data is None or len(data) < 50:
                print(f"Insufficient data for {symbol}, skipping model training")
                return False
            
            df = self.add_technical_indicators(data)
            if df is None:
                return False
            
            X, y = self.prepare_features(df, symbol)
            if X.empty or y.empty:
                print(f"No features generated for {symbol}")
                return False
            
            # Hyperparameter optimization using GridSearchCV
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', None]
            }
            print(f"Training model for {symbol} with {len(X)} samples")
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
            grid_search.fit(X, y)
            self.models[symbol] = grid_search.best_estimator_
            print(f"Best parameters for {symbol}: {grid_search.best_params_}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False

    def predict_index(self, symbol):
        """Generate prediction for a specific index"""
        try:
            # Check if model exists
            if symbol not in self.models:
                if not self.train_model(symbol):
                    return None, 0
            
            # Fetch latest data
            data = self.fetch_data(symbol)
            if data is None:
                return None, 0
            
            df = self.add_technical_indicators(data)
            if df is None:
                return None, 0
            
            # Prepare features
            X, _ = self.prepare_features(df, symbol)
            if X.empty:
                return None, 0
            
            # Get most recent data point for prediction
            latest_data = X.iloc[-1:].copy()
            
            # Make prediction
            prediction = self.models[symbol].predict(latest_data)[0]
            probability = self.models[symbol].predict_proba(latest_data)[0]
            
            option_type = 'CE' if prediction == 1 else 'PE'
            confidence = probability[1] if prediction == 1 else probability[0]
            
            latest_close_price = data['Close'].iloc[-1]
            strike_price = round(latest_close_price / 100) * 100  # Round to nearest 100
            
            return f"{symbol} {strike_price} {option_type}", confidence
            
        except Exception as e:
            logger.error(f"Error in predict_index for {symbol}: {e}")
            return None, 0

    def paper_trade(self, symbol):
        """Simulate a paper trade for a specific index"""
        print('Inside Paper')
        try:
            print('Inside Paper 1')
            data = self.fetch_data(symbol)
            print('Inside Paper 2')
            if data is None or data.empty:  # Use .empty to check for empty DataFrame
                print('Inside Paper 3')
                logger.warning(f"Insufficient data for {symbol}, skipping paper trading")
                return
            
            print('Inside Paper 4')
            
            df = self.add_technical_indicators(data)
            print('Inside Paper 5')

            if df is None or df.empty:
                print('Inside Paper 6')
                logger.warning(f"No indicators generated for {symbol}, skipping paper trading")
                return
            
            # Get the latest close price
            print('Inside Paper 7')
            latest_close_price = df['Close'].iloc[-1]
            print('Inside Paper 8')
            strike_price = round(latest_close_price / 100) * 100  # Round to nearest 100
            print('Inside Paper 9')
            # Simulate entering a trade
            entry_price = latest_close_price
            print('Inside Paper 10')
            target_price = entry_price * (1 + self.target_pct)  # Target price
            print('Inside Paper 11')
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)  # Stop loss price
            print('Inside Paper 12')
            
            logger.info(f"Entering paper trade for {symbol}: Entry Price: {entry_price}, Target: {target_price}, Stop Loss: {stop_loss_price}")
            
            trade_start_time = datetime.now()
            while True:
                # Fetch the latest data to check current price
                data = self.fetch_data(symbol)
                if data is None or data.empty:  # Use .empty to check for empty DataFrame
                    logger.warning(f"Failed to fetch data for {symbol} during trade monitoring")
                    break
                
                current_price = data['Close'].iloc[-1]
                logger.info(f"Current Price: {current_price}")

                # Check for target hit
                if current_price >= target_price:
                    profit = current_price - entry_price
                    logger.info(f"Target hit! Exiting trade for {symbol}. Profit: {profit:.2f}")
                    break

                # Check for stop loss hit
                if current_price <= stop_loss_price:
                    loss = entry_price - current_price
                    logger.info(f"Stop loss hit! Exiting trade for {symbol}. Loss: {loss:.2f}")
                    break

                # Check if trade duration exceeds 45 minutes
                if (datetime.now() - trade_start_time).total_seconds() > 2700:  # 45 minutes
                    logger.info(f"Trade duration exceeded 45 minutes. Exiting trade for {symbol}.")
                    break

                time.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"Error in paper trading for {symbol}: {e}")

    def analyze_market(self):
        """Analyze market in parallel and generate trade signals"""
        best_signal = None
        highest_confidence = 0.5
        
        # Using ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.indices)) as executor:
            # Submit all symbol predictions to the thread pool
            future_to_symbol = {
                executor.submit(self.predict_index, symbol): (index_name, symbol) 
                for index_name, symbol in self.indices.items()
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                index_name, symbol = future_to_symbol[future]
                try:
                    signal, confidence = future.result()
                    print("Paper Trade Confidence", confidence, self.confidence_threshold)
                    if signal and confidence >= self.confidence_threshold and confidence > highest_confidence:
                        highest_confidence = confidence
                        best_signal = f"{signal} (Confidence: {confidence:.2f})"
                        # Execute paper trade for the best signal
                        self.paper_trade(symbol)
                except Exception as e:
                    logger.error(f"Error processing {index_name} prediction: {e}")

        return best_signal

    def run_live(self):
        """Run the trading bot live with improved error handling and performance"""
        print("Initializing Enhanced Trading Bot...")
        
        # Pre-train models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.indices)) as executor:
            training_futures = []
            for index_name, symbol in self.indices.items():
                training_futures.append(executor.submit(self.train_model, symbol))
            
            # Wait for all training to complete
            for future in concurrent.futures.as_completed(training_futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error during model training: {e}")
        
        print("Starting enhanced market monitoring...")
        
        while True:
            try:
                if not self.is_market_open():
                    next_check = 60
                    print(f"Market closed. Checking again in {next_check} seconds.")
                    time.sleep(next_check)
                    continue
                
                # Analyze market
                start_time = time.time()
                best_signal = self.analyze_market()
                analysis_time = time.time() - start_time
                
                if best_signal:
                    print(f"SIGNAL: {best_signal} (Analysis took {analysis_time:.2f}s)")
                else:
                    print(f"No trading signal generated (Analysis took {analysis_time:.2f}s)")
                
                # Adaptive sleep time - less frequent checks outside of peak hours
                current_time = datetime.now(self.ist_timezone)
                if current_time.hour < 10 or current_time.hour >= 14:
                    sleep_time = 300  # 5 minutes outside peak hours
                else:
                    sleep_time = 180  # 3 minutes during peak hours
                
                logger.debug(f"Waiting {sleep_time} seconds before next analysis...")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print("Stopping monitoring by user request...")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retry

    def backtest(self, symbol, start_date='2023-01-01', end_date=None):
        """Backtest the model on historical data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Backtesting {symbol} from {start_date} to {end_date}")
        
        try:
            # Get historical data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data is None or data.empty:
                logger.error("No backtest data available")
                return
                
            # Prepare data
            df = self.add_technical_indicators(data)
            if df is None:
                return
                
            # Create target
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            # Split into train/test (70% train, 30% test)
            train_size = int(len(df) * 0.7)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # Train model on training data
            X_train, y_train = self.prepare_features(train_df, symbol)
            if X_train.empty or y_train.empty:
                logger.error("Failed to prepare training features")
                return
                
            model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
            
            # Test on test data
            X_test, y_test = self.prepare_features(test_df, symbol)
            if X_test.empty or y_test.empty:
                logger.error("Failed to prepare test features")
                return
                
            # Make predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Calculate accuracy
            accuracy = (predictions == y_test).mean()
            print(f"Backtest accuracy: {accuracy:.2f}")
            
            # Simulate trading with the model
            test_df_reset = test_df.reset_index()
            test_df_reset['Prediction'] = np.nan
            test_df_reset['Confidence'] = np.nan
            
            for i in range(len(X_test)):
                test_df_reset.loc[i, 'Prediction'] = predictions[i]
                test_df_reset.loc[i, 'Confidence'] = probabilities[i][1] if predictions[i] == 1 else probabilities[i][0]
            
            # Filter trades based on confidence threshold
            trades = test_df_reset[test_df_reset['Confidence'] >= self.confidence_threshold]
            correct_trades = trades[trades['Prediction'] == trades['Target']]
            
            print(f"Total trades: {len(trades)}")
            print(f"Correct trades: {len(correct_trades)}")
            print(f"Win rate: {len(correct_trades) / max(1, len(trades)):.2f}")
            
            return accuracy, len(trades), len(correct_trades)
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return None, 0, 0

if __name__ == "__main__":
    bot = EnhancedTradingBot()
    
    # Uncomment to run backtest before going live
    # for index_name, symbol in bot.indices.items():
    #     bot.backtest(symbol)
    
    bot.run_live()