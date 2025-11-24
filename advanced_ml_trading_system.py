#!/usr/bin/env python3
"""
Advanced ML/AI Trading System with RAG
Production-grade system for recording data, analysis, and generating live trading advice
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
import hashlib

# ML/AI Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn xgboost")

# Vector Database for RAG
try:
    import faiss
    import sentence_transformers
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu sentence-transformers")

# LLM for RAG
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ========== CONFIGURATION ==========

DATA_DIR = Path("data/ml_trading_system")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "trading_data.db"
VECTOR_DB_PATH = DATA_DIR / "vector_index.faiss"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pkl"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Feature engineering parameters
FEATURE_WINDOWS = [5, 10, 20, 50, 100]
PREDICTION_HORIZON = 5  # Predict next 5 periods

# RAG Configuration
RAG_TOP_K = 5  # Number of similar historical cases to retrieve
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast embedding model

# ========== DATA STRUCTURES ==========

@dataclass
class MarketSnapshot:
    """Structured market data snapshot."""
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    atr: Optional[float] = None
    vwap: Optional[float] = None
    price_change_pct: Optional[float] = None
    volume_change_pct: Optional[float] = None

@dataclass
class TradingSignal:
    """Trading signal with confidence and reasoning."""
    timestamp: str
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    ml_prediction: Optional[str] = None
    ml_confidence: Optional[float] = None
    rag_context: Optional[List[str]] = None

@dataclass
class TradeOutcome:
    """Record of actual trade outcome for learning."""
    signal_id: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    outcome: str  # WIN, LOSS, BREAKEVEN
    duration_minutes: int

# ========== DATABASE LAYER ==========

class TradingDatabase:
    """SQLite database for storing market data and signals."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                ma_20 REAL,
                ma_50 REAL,
                bb_upper REAL,
                bb_lower REAL,
                atr REAL,
                vwap REAL,
                price_change_pct REAL,
                volume_change_pct REAL,
                UNIQUE(timestamp, symbol)
            )
        """)
        
        # Trading signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                reasoning TEXT,
                ml_prediction TEXT,
                ml_confidence REAL,
                rag_context TEXT
            )
        """)
        
        # Trade outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                outcome TEXT,
                duration_minutes INTEGER,
                FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id)
            )
        """)
        
        # Analysis logs table (for RAG context)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                symbol TEXT,
                analysis_text TEXT,
                market_context TEXT,
                outcome_summary TEXT,
                embedding_id INTEGER
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_symbol ON market_snapshots(timestamp, symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_timestamp ON trading_signals(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcome_signal ON trade_outcomes(signal_id)")
        
        conn.commit()
        conn.close()
    
    def save_snapshot(self, snapshot: MarketSnapshot):
        """Save market snapshot to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO market_snapshots 
            (timestamp, symbol, open, high, low, close, volume, rsi, macd, macd_signal, macd_hist,
             ma_20, ma_50, bb_upper, bb_lower, atr, vwap, price_change_pct, volume_change_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.timestamp, snapshot.symbol, snapshot.open, snapshot.high, snapshot.low,
            snapshot.close, snapshot.volume, snapshot.rsi, snapshot.macd, snapshot.macd_signal,
            snapshot.macd_hist, snapshot.ma_20, snapshot.ma_50, snapshot.bb_upper,
            snapshot.bb_lower, snapshot.atr, snapshot.vwap, snapshot.price_change_pct,
            snapshot.volume_change_pct
        ))
        
        conn.commit()
        conn.close()
    
    def get_historical_data(self, symbol: str, start_date: str = None, end_date: str = None, limit: int = None) -> pd.DataFrame:
        """Retrieve historical market data."""
        conn = sqlite3.connect(self.db_path)
        
        # Use parameterized queries to prevent SQL injection
        query = "SELECT * FROM market_snapshots WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        query += " ORDER BY timestamp"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    def save_signal(self, signal: TradingSignal):
        """Save trading signal."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        rag_context_str = json.dumps(signal.rag_context) if signal.rag_context else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO trading_signals
            (signal_id, timestamp, symbol, signal, confidence, entry_price, stop_loss, take_profit,
             reasoning, ml_prediction, ml_confidence, rag_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.timestamp + "_" + signal.symbol, signal.timestamp, signal.symbol,
            signal.signal, signal.confidence, signal.entry_price, signal.stop_loss,
            signal.take_profit, signal.reasoning, signal.ml_prediction, signal.ml_confidence,
            rag_context_str
        ))
        
        conn.commit()
        conn.close()
    
    def save_outcome(self, outcome: TradeOutcome):
        """Save trade outcome."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trade_outcomes
            (signal_id, entry_time, exit_time, entry_price, exit_price, pnl, pnl_pct, outcome, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome.signal_id, outcome.entry_time, outcome.exit_time,
            outcome.entry_price, outcome.exit_price, outcome.pnl, outcome.pnl_pct,
            outcome.outcome, outcome.duration_minutes
        ))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self, symbol: str = None, min_samples: int = 1000) -> pd.DataFrame:
        """Get data for ML training."""
        conn = sqlite3.connect(self.db_path)
        
        # Use parameterized queries to prevent SQL injection
        query = """
            SELECT ms.*, ts.signal as target_signal, ts.confidence as signal_confidence,
                   to.outcome, to.pnl_pct
            FROM market_snapshots ms
            LEFT JOIN trading_signals ts ON ms.timestamp = ts.timestamp AND ms.symbol = ts.symbol
            LEFT JOIN trade_outcomes to ON ts.signal_id = to.signal_id
        """
        
        params = []
        if symbol:
            query += " WHERE ms.symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY ms.timestamp"
        
        df = pd.read_sql_query(query, conn, params=params if params else None)
        conn.close()
        
        if len(df) < min_samples:
            return pd.DataFrame()
        
        return df

# ========== FEATURE ENGINEERING ==========

class FeatureEngineer:
    """Advanced feature engineering for ML models."""
    
    def __init__(self, windows: List[int] = FEATURE_WINDOWS):
        self.windows = windows
        self.scaler = RobustScaler()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set."""
        if df.empty or len(df) < max(self.windows):
            return pd.DataFrame()
        
        features_df = df.copy()
        
        # Normalize column names to lowercase (yfinance returns uppercase)
        column_mapping = {}
        for col in features_df.columns:
            if col.isupper() or col[0].isupper():
                column_mapping[col] = col.lower()
        if column_mapping:
            features_df = features_df.rename(columns=column_mapping)
        
        # Ensure required columns exist (handle both cases)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in features_df.columns:
                # Try uppercase version
                upper_col = col.capitalize()
                if upper_col in features_df.columns:
                    features_df[col] = features_df[upper_col]
                else:
                    # Column not found, return empty
                    return pd.DataFrame()
        
        # Price-based features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # Rolling statistics
        for window in self.windows:
            if len(df) >= window:
                features_df[f'ma_{window}'] = features_df['close'].rolling(window).mean()
                features_df[f'std_{window}'] = features_df['close'].rolling(window).std()
                features_df[f'min_{window}'] = features_df['low'].rolling(window).min()
                features_df[f'max_{window}'] = features_df['high'].rolling(window).max()
                features_df[f'volume_ma_{window}'] = features_df['volume'].rolling(window).mean()
        
        # Technical indicators (if not already present)
        if 'rsi' not in features_df.columns or features_df['rsi'].isna().all():
            features_df['rsi'] = self._calculate_rsi(features_df['close'])
        
        # MACD features
        if 'macd' not in features_df.columns or features_df['macd'].isna().all():
            macd_data = self._calculate_macd(features_df['close'])
            features_df['macd'] = macd_data['macd']
            features_df['macd_signal'] = macd_data['signal']
            features_df['macd_hist'] = macd_data['hist']
        
        # Bollinger Bands
        if 'ma_20' in features_df.columns:
            bb_std = features_df['close'].rolling(20).std()
            features_df['bb_upper'] = features_df['ma_20'] + (bb_std * 2)
            features_df['bb_lower'] = features_df['ma_20'] - (bb_std * 2)
            features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']
            features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # ATR (Average True Range)
        features_df['tr'] = np.maximum(
            features_df['high'] - features_df['low'],
            np.maximum(
                abs(features_df['high'] - features_df['close'].shift(1)),
                abs(features_df['low'] - features_df['close'].shift(1))
            )
        )
        features_df['atr'] = features_df['tr'].rolling(14).mean()
        
        # Price position features
        features_df['price_position'] = (features_df['close'] - features_df['low']) / (features_df['high'] - features_df['low'])
        features_df['body_size'] = abs(features_df['close'] - features_df['open'])
        features_df['upper_shadow'] = features_df['high'] - features_df[['open', 'close']].max(axis=1)
        features_df['lower_shadow'] = features_df[['open', 'close']].min(axis=1) - features_df['low']
        
        # Volume features
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
        features_df['price_volume'] = features_df['close'] * features_df['volume']
        
        # Momentum features
        for window in [5, 10, 20]:
            if len(df) >= window:
                features_df[f'momentum_{window}'] = features_df['close'] / features_df['close'].shift(window) - 1
                features_df[f'roc_{window}'] = (features_df['close'] - features_df['close'].shift(window)) / features_df['close'].shift(window) * 100
        
        # Drop NaN rows
        features_df = features_df.dropna()
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        hist = macd - signal_line
        return {'macd': macd, 'signal': signal_line, 'hist': hist}
    
    def prepare_ml_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for ML model.
        
        Args:
            df: DataFrame with features
            fit_scaler: If True, fit the scaler (use during training).
                       If False, only transform (use during prediction).
        """
        # Select feature columns (exclude target and metadata)
        exclude_cols = ['timestamp', 'symbol', 'signal', 'target_signal', 'outcome', 'pnl_pct']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        X = df[feature_cols].values
        
        # Only fit scaler during training, transform during prediction
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            # Check if scaler has been fitted
            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                raise ValueError("Scaler not fitted. Call prepare_ml_features with fit_scaler=True during training first.")
            X = self.scaler.transform(X)
        
        return X, feature_cols

# ========== ML MODEL TRAINING ==========

class TradingMLModel:
    """ML model for trading signal prediction."""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_names = []
        self.scaler = RobustScaler()
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the model."""
        self.feature_names = feature_names
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # Cross-validation
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model.fit(X_train, y_train)
            score = self.model.score(X_val, y_val)
            scores.append(score)
        
        # Final training on all data
        self.model.fit(X, y)
        
        print(f"Model trained. CV scores: {scores}")
        print(f"Average CV score: {np.mean(scores):.4f}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict trading signals."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidences = np.max(probabilities, axis=1)
        
        return predictions, confidences
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance."""
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = dict(zip(self.feature_names, importances))
            return dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        return {}
    
    def save(self, filepath: Path):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: Path):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.scaler = model_data.get('scaler', RobustScaler())

# ========== RAG SYSTEM ==========

class TradingRAG:
    """Retrieval Augmented Generation for contextual trading advice."""
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model_name = embedding_model
        self.encoder = None
        self.index = None
        self.contexts = []
        self.db = TradingDatabase()
        
        if FAISS_AVAILABLE:
            self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
    
    def build_index(self):
        """Build vector index from historical data."""
        if not FAISS_AVAILABLE or self.encoder is None:
            print("FAISS or encoder not available. RAG will use database search.")
            return
        
        # Get all analysis logs and trade outcomes
        conn = sqlite3.connect(self.db.db_path)
        
        # Get analysis contexts
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, date, symbol, analysis_text, market_context, outcome_summary
            FROM analysis_logs
            WHERE analysis_text IS NOT NULL
            ORDER BY timestamp
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        self.contexts = []
        texts = []
        
        for row in rows:
            timestamp, date, symbol, analysis_text, market_context, outcome_summary = row
            context_text = f"""
Date: {date}
Symbol: {symbol}
Market Context: {market_context or 'N/A'}
Analysis: {analysis_text}
Outcome: {outcome_summary or 'N/A'}
"""
            texts.append(context_text)
            self.contexts.append({
                'timestamp': timestamp,
                'date': date,
                'symbol': symbol,
                'text': context_text
            })
        
        if not texts:
            print("No historical contexts found for RAG index.")
            return
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} contexts...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(self.index, str(VECTOR_DB_PATH))
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(self.contexts, f)
        
        print(f"RAG index built with {len(self.contexts)} contexts.")
    
    def retrieve_context(self, query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
        """Retrieve relevant historical contexts."""
        if self.index is None or self.encoder is None:
            # Fallback to database search
            return self._database_search(query, top_k)
        
        # Encode query
        query_embedding = self.encoder.encode([query])[0].astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve contexts
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.contexts):
                results.append({
                    **self.contexts[idx],
                    'similarity': float(1 / (1 + dist))  # Convert distance to similarity
                })
        
        return results
    
    def _database_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback database search."""
        # Simple keyword-based search
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        # Extract keywords from query
        keywords = query.lower().split()[:5]  # Use first 5 words
        
        query_sql = """
            SELECT timestamp, date, symbol, analysis_text, market_context, outcome_summary
            FROM analysis_logs
            WHERE analysis_text LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        results = []
        for keyword in keywords:
            cursor.execute(query_sql, (f'%{keyword}%', top_k))
            rows = cursor.fetchall()
            for row in rows:
                results.append({
                    'timestamp': row[0],
                    'date': row[1],
                    'symbol': row[2],
                    'text': f"Analysis: {row[3]}\nOutcome: {row[4] or 'N/A'}",
                    'similarity': 0.5  # Default similarity for keyword match
                })
        
        conn.close()
        return results[:top_k]
    
    def generate_advice(self, current_context: Dict, ml_prediction: str, ml_confidence: float) -> str:
        """Generate trading advice using RAG."""
        # Build query from current context
        query = f"""
        Current market conditions:
        Symbol: {current_context.get('symbol', 'N/A')}
        Price: {current_context.get('price', 'N/A')}
        RSI: {current_context.get('rsi', 'N/A')}
        Trend: {current_context.get('trend', 'N/A')}
        ML Prediction: {ml_prediction} (confidence: {ml_confidence:.2%})
        """
        
        # Retrieve similar historical contexts
        similar_cases = self.retrieve_context(query, top_k=RAG_TOP_K)
        
        # Build context for LLM
        historical_context = "\n\n".join([
            f"Historical Case {i+1}:\n{c['text']}\nSimilarity: {c['similarity']:.2%}"
            for i, c in enumerate(similar_cases)
        ])
        
        # Generate advice using LLM
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            return self._generate_with_openai(current_context, ml_prediction, ml_confidence, historical_context)
        else:
            return self._generate_rule_based_advice(current_context, ml_prediction, ml_confidence, similar_cases)
    
    def _generate_with_openai(self, current_context: Dict, ml_prediction: str, ml_confidence: float, historical_context: str) -> str:
        """Generate advice using OpenAI."""
        try:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            prompt = f"""You are an expert trading advisor. Based on current market conditions, ML predictions, and similar historical cases, provide actionable trading advice.

CURRENT SITUATION:
{json.dumps(current_context, indent=2)}

ML PREDICTION: {ml_prediction} (Confidence: {ml_confidence:.2%})

SIMILAR HISTORICAL CASES:
{historical_context}

Provide:
1. Trading recommendation (BUY/SELL/HOLD) with reasoning
2. Entry price suggestion
3. Stop loss level
4. Take profit target
5. Risk assessment
6. Key factors to watch

Be specific with price levels and percentages. Keep it concise but actionable."""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert futures trading advisor with deep knowledge of technical analysis and risk management."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating advice: {e}. Using rule-based fallback."
    
    def _generate_rule_based_advice(self, current_context: Dict, ml_prediction: str, ml_confidence: float, similar_cases: List[Dict]) -> str:
        """Generate rule-based advice."""
        advice = f"Trading Recommendation: {ml_prediction}\n\n"
        advice += f"ML Confidence: {ml_confidence:.2%}\n\n"
        
        if similar_cases:
            advice += "Similar Historical Cases Found:\n"
            for i, case in enumerate(similar_cases[:3], 1):
                advice += f"{i}. {case.get('date', 'Unknown')} - {case.get('symbol', 'N/A')}\n"
            advice += "\n"
        
        # Basic rule-based logic
        rsi = current_context.get('rsi', 50)
        trend = current_context.get('trend', 'NEUTRAL')
        
        if ml_prediction == "BUY" and ml_confidence > 0.7:
            advice += "Strong BUY signal. Consider entering long position.\n"
        elif ml_prediction == "SELL" and ml_confidence > 0.7:
            advice += "Strong SELL signal. Consider entering short position.\n"
        else:
            advice += "HOLD recommended. Wait for clearer signals.\n"
        
        return advice

# ========== MAIN SYSTEM ==========

class AdvancedTradingSystem:
    """Complete advanced trading system with ML and RAG."""
    
    def __init__(self):
        self.db = TradingDatabase()
        self.feature_engineer = FeatureEngineer()
        self.ml_model = TradingMLModel(model_type="xgboost")
        self.rag = TradingRAG()
        self.is_trained = False
    
    def record_data(self, snapshot: MarketSnapshot):
        """Record market data."""
        self.db.save_snapshot(snapshot)
    
    def train_models(self, symbol: str = None, min_samples: int = 2000):
        """Train ML models on historical data."""
        print("Collecting training data...")
        df = self.db.get_training_data(symbol=symbol, min_samples=min_samples)
        
        if df.empty:
            print("Insufficient data for training. Need at least 2000 samples.")
            return False
        
        print(f"Training on {len(df)} samples...")
        
        # Create features
        features_df = self.feature_engineer.create_features(df)
        
        if features_df.empty:
            print("Feature engineering failed.")
            return False
        
        # Prepare target (next period's price movement)
        features_df['future_return'] = features_df['close'].shift(-PREDICTION_HORIZON) / features_df['close'] - 1
        features_df['target'] = 'HOLD'
        features_df.loc[features_df['future_return'] > 0.002, 'target'] = 'BUY'  # >0.2% move
        features_df.loc[features_df['future_return'] < -0.002, 'target'] = 'SELL'  # <-0.2% move
        
        # Remove rows with NaN targets
        features_df = features_df.dropna(subset=['target'])
        
        # Prepare features (fit scaler during training)
        X, feature_names = self.feature_engineer.prepare_ml_features(features_df, fit_scaler=True)
        y = features_df['target'].values
        
        # Train model
        print("Training ML model...")
        self.ml_model.train(X, y, feature_names)
        
        # Save model
        model_path = MODEL_DIR / f"trading_model_{datetime.now().strftime('%Y%m%d')}.pkl"
        self.ml_model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Build RAG index
        print("Building RAG index...")
        self.rag.build_index()
        
        self.is_trained = True
        return True
    
    def generate_signal(self, symbol: str, current_data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal with ML and RAG."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Create features for current data
        features_df = self.feature_engineer.create_features(current_data)
        
        if features_df.empty or len(features_df) < 1:
            # Get current price (handle both case variations)
            close_col = 'close' if 'close' in current_data.columns else 'Close'
            entry_price = float(current_data[close_col].iloc[-1]) if not current_data.empty else 0.0
            
            return TradingSignal(
                timestamp=datetime.now(timezone.utc).isoformat(),
                symbol=symbol,
                signal="HOLD",
                confidence=0.5,
                entry_price=entry_price,
                reasoning="Insufficient data for prediction"
            )
        
        # Get latest features (transform only, don't refit scaler)
        X, _ = self.feature_engineer.prepare_ml_features(features_df.tail(1), fit_scaler=False)
        
        # ML prediction
        prediction, confidence = self.ml_model.predict(X)
        ml_signal = prediction[0]
        ml_confidence = float(confidence[0])
        
        # Build current context
        latest = features_df.iloc[-1]
        current_context = {
            'symbol': symbol,
            'price': float(latest['close']),
            'rsi': float(latest.get('rsi', 50)),
            'trend': 'UPTREND' if latest.get('ma_20', 0) > latest.get('ma_50', 0) else 'DOWNTREND',
            'volume': int(latest.get('volume', 0)),
        }
        
        # RAG-based advice
        rag_advice = self.rag.generate_advice(current_context, ml_signal, ml_confidence)
        
        # Extract signal from advice or use ML prediction
        final_signal = ml_signal
        if ml_confidence > 0.7:
            final_signal = ml_signal
        else:
            final_signal = "HOLD"
        
        # Calculate stop loss and take profit
        current_price = float(latest['close'])
        atr = float(latest.get('atr', current_price * 0.01))
        
        if final_signal == "BUY":
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        elif final_signal == "SELL":
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        else:
            stop_loss = None
            take_profit = None
        
        signal = TradingSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            signal=final_signal,
            confidence=ml_confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=rag_advice,
            ml_prediction=ml_signal,
            ml_confidence=ml_confidence,
            rag_context=[c.get('text', '')[:200] for c in self.rag.retrieve_context(f"Symbol: {symbol}", top_k=3)]
        )
        
        # Save signal
        self.db.save_signal(signal)
        
        return signal
    
    def record_outcome(self, signal_id: str, exit_price: float, exit_time: str):
        """Record trade outcome for learning."""
        # Get signal
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, entry_price FROM trading_signals WHERE signal_id = ?", (signal_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return
        
        entry_time = row[0]
        entry_price = row[1]
        
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        
        duration = (datetime.fromisoformat(exit_time) - datetime.fromisoformat(entry_time)).total_seconds() / 60
        
        outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
        
        trade_outcome = TradeOutcome(
            signal_id=signal_id,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            outcome=outcome,
            duration_minutes=int(duration)
        )
        
        self.db.save_outcome(trade_outcome)
        
        # Retrain periodically based on new outcomes
        # (In production, you'd want to retrain on a schedule)

# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED ML/AI TRADING SYSTEM")
    print("=" * 60)
    
    system = AdvancedTradingSystem()
    
    print("\nSystem initialized.")
    print("Next steps:")
    print("1. Record market data using system.record_data()")
    print("2. Train models using system.train_models()")
    print("3. Generate signals using system.generate_signal()")
    print("4. Record outcomes using system.record_outcome()")

