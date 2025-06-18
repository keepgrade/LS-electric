"""
개선된 LS전기 전기요금 예측 모델
- 모듈화된 코드 구조
- 고급 피처 엔지니어링 (분 단위 포함)
- 다중 스케일링 전략
- 시계열 교차검증
- 스태킹 앙상블
- LSTM 아키텍처 개선
- 멀티태스크 학습
- 모델 해석 및 실험 추적
"""

import os
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

warnings.filterwarnings('ignore')

@dataclass
class Config:
    """설정 클래스"""
    BASE_DIR: str = "../dashboard/data"
    MODELS_DIR: str = "pickles"
    TIME_STEPS: int = 96 * 7  # 7일 시퀀스
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    N_TRIALS: int = 50
    CV_FOLDS: int = 5
    
    # 타겟 변수들
    MULTI_TARGETS: List[str] = None
    
    def __post_init__(self):
        if self.MULTI_TARGETS is None:
            self.MULTI_TARGETS = [
                "전력사용량(kWh)",
                "지상무효전력량(kVarh)", 
                "진상무효전력량(kVarh)",
                "탄소배출량(tCO2)",
                "지상역률(%)",
                "진상역률(%)",
                "전기요금(원)"
            ]

class DataProcessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.le = LabelEncoder()
        self.scalers = {}
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=config.RANDOM_STATE)
        
        # 요금 테이블
        self.RATE_TABLE = {
            "before": {
                "여름": {"경부하": 93.1, "중간부하": 146.3, "최대부하": 216.6},
                "봄가을": {"경부하": 93.1, "중간부하": 115.2, "최대부하": 138.9},
                "겨울": {"경부하": 100.4, "중간부하": 146.5, "최대부하": 193.4},
            },
            "after": {
                "여름": {"경부하": 110.0, "중간부하": 163.2, "최대부하": 233.5},
                "봄가을": {"경부하": 110.0, "중간부하": 132.1, "최대부하": 155.8},
                "겨울": {"경부하": 117.3, "중간부하": 163.4, "최대부하": 210.3},
            },
        }
        self.CUTOFF = datetime(2024, 10, 24)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        train_df = pd.read_csv(os.path.join(self.config.BASE_DIR, "train.csv"))
        test_df = pd.read_csv(os.path.join(self.config.BASE_DIR, "test.csv"))
        return train_df, test_df
    
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 시간 피처 생성 (분 단위 포함)"""
        df = df.copy()
        df["측정일시"] = pd.to_datetime(df["측정일시"])
        
        # 기본 시간 피처
        df["년"] = df["측정일시"].dt.year
        df["월"] = df["측정일시"].dt.month
        df["일"] = df["측정일시"].dt.day
        df["시간"] = df["측정일시"].dt.hour
        df["분"] = df["측정일시"].dt.minute  # 분 추가
        df["요일"] = df["측정일시"].dt.weekday
        df["주말여부"] = (df["요일"] >= 5).astype(int)
        
        # 순환 인코딩
        df["sin_시간"] = np.sin(2 * np.pi * df["시간"] / 24)
        df["cos_시간"] = np.cos(2 * np.pi * df["시간"] / 24)
        df["sin_분"] = np.sin(2 * np.pi * df["분"] / 60)  # 분 순환 인코딩
        df["cos_분"] = np.cos(2 * np.pi * df["분"] / 60)
        df["sin_월"] = np.sin(2 * np.pi * df["월"] / 12)
        df["cos_월"] = np.cos(2 * np.pi * df["월"] / 12)
        df["sin_일"] = np.sin(2 * np.pi * df["일"] / 31)
        df["cos_일"] = np.cos(2 * np.pi * df["일"] / 31)
        df["sin_요일"] = np.sin(2 * np.pi * df["요일"] / 7)
        df["cos_요일"] = np.cos(2 * np.pi * df["요일"] / 7)
        
        # 계절 피처
        df["계절_encoded"] = df["월"].apply(lambda x: (x-1)//3)
        
        # 시간대 분류 (더 세분화)
        df["시간대_세분화"] = df["시간"].apply(self._get_detailed_time_period)
        
        return df
    
    def _get_detailed_time_period(self, hour: int) -> str:
        """세분화된 시간대 분류"""
        if 6 <= hour < 9:
            return "출근시간"
        elif 9 <= hour < 12:
            return "오전업무"
        elif 12 <= hour < 14:
            return "점심시간"
        elif 14 <= hour < 18:
            return "오후업무"
        elif 18 <= hour < 21:
            return "퇴근시간"
        elif 21 <= hour < 24:
            return "저녁시간"
        else:
            return "야간시간"
    
    def get_season(self, month: int) -> str:
        """계절 분류"""
        if month in [6, 7, 8]:
            return "여름"
        elif month in [3, 4, 5, 9, 10]:
            return "봄가을"
        return "겨울"
    
    def get_time_zone(self, hour: int, season: str) -> str:
        """시간대별 전력 부하 분류"""
        if season in ["여름", "봄가을"]:
            if 22 <= hour or hour < 8:
                return "경부하"
            if (8 <= hour < 11) or (12 <= hour < 13) or (18 <= hour < 22):
                return "중간부하"
            return "최대부하"
        else:
            if 22 <= hour or hour < 8:
                return "경부하"
            if (8 <= hour < 9) or (12 <= hour < 16) or (19 <= hour < 22):
                return "중간부하"
            return "최대부하"
    
    def create_tariff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """요금 관련 피처 생성"""
        df = df.copy()
        df["계절"] = df["월"].apply(self.get_season)
        df["적용시점"] = df["측정일시"].apply(lambda x: "before" if x < self.CUTOFF else "after")
        df["시간대"] = df.apply(lambda r: self.get_time_zone(r["시간"], r["계절"]), axis=1)
        df["요금단가"] = df.apply(lambda r: self.RATE_TABLE[r["적용시점"]][r["계절"]][r["시간대"]], axis=1)
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """지연 피처 생성"""
        df = df.copy()
        df = df.sort_values('측정일시').reset_index(drop=True)
        
        # 다양한 지연 기간
        lags = [1, 2, 6, 12, 24, 48, 96, 168, 336]  # 15분부터 2주까지
        
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """롤링 통계 피처 생성"""
        df = df.copy()
        df = df.sort_values('측정일시').reset_index(drop=True)
        
        # 다양한 윈도우 크기
        windows = [6, 12, 24, 48, 96, 168]  # 1.5시간부터 1주까지
        
        for window in windows:
            df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window).mean()
            df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window).std()
            df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(window).max()
            df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(window).min()
            df[f"{target_col}_rolling_median_{window}"] = df[target_col].rolling(window).median()
            
            # 변화율 피처
            df[f"{target_col}_pct_change_{window}"] = df[target_col].pct_change(window)
        
        return df
    
    def target_encoding(self, df_train: pd.DataFrame, df_test: pd.DataFrame, 
                       col: str, target: str, smoothing: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """타겟 인코딩"""
        global_mean = df_train[target].mean()
        agg = df_train.groupby(col)[target].agg(["mean", "count"])
        smoothing_weight = 1 / (1 + np.exp(-(agg["count"] - smoothing)))
        enc = global_mean * (1 - smoothing_weight) + agg["mean"] * smoothing_weight
        mapping = enc.to_dict()
        
        df_train = df_train.copy()
        df_test = df_test.copy()
        df_train[f"{col}_te"] = df_train[col].map(mapping)
        df_test[f"{col}_te"] = df_test[col].map(mapping)
        
        return df_train, df_test
    
    def remove_outliers(self, df: pd.DataFrame, target_col: str, method: str = "combined") -> pd.DataFrame:
        """개선된 이상치 제거"""
        df = df.copy()
        
        if method == "iqr":
            q1 = df[target_col].quantile(0.25)
            q3 = df[target_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (df[target_col] >= lower) & (df[target_col] <= upper)
            
        elif method == "isolation_forest":
            features_for_outlier = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
            if features_for_outlier:
                outlier_pred = self.isolation_forest.fit_predict(df[features_for_outlier].fillna(0))
                mask = outlier_pred == 1
            else:
                mask = pd.Series([True] * len(df))
                
        elif method == "combined":
            # IQR + Isolation Forest 조합
            q1 = df[target_col].quantile(0.25)
            q3 = df[target_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_mask = (df[target_col] >= lower) & (df[target_col] <= upper)
            
            features_for_outlier = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
            if features_for_outlier:
                outlier_pred = self.isolation_forest.fit_predict(df[features_for_outlier].fillna(0))
                iso_mask = outlier_pred == 1
                mask = iqr_mask & iso_mask
            else:
                mask = iqr_mask
        
        return df[mask]
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """피처 그룹 정의 (분 포함)"""
        return {
            'numerical_features': [
                '요금단가', '작업유형_te', '시간_te', '요일_te', '시간대_te',
                'sin_시간', 'cos_시간', 'sin_분', 'cos_분', 'sin_월', 'cos_월', 
                'sin_일', 'cos_일', 'sin_요일', 'cos_요일'
            ],
            'categorical_features': [
                '작업유형_encoded', '년', '월', '일', '시간', '분', '요일', '주말여부',
                '계절_encoded', '시간대_세분화'
            ],
            'lag_features': [],  # 동적으로 추가됨
            'rolling_features': []  # 동적으로 추가됨
        }
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """피처 타입별 다른 스케일링 적용"""
        # ✅ NaN / Inf 제거
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        feature_groups = self.get_feature_groups()
        actual_features = X_train.columns.tolist()

        # 동적으로 lag, rolling 피처 찾기
        for col in actual_features:
            if '_lag_' in col:
                feature_groups['lag_features'].append(col)
            elif '_rolling_' in col or '_pct_change_' in col:
                feature_groups['rolling_features'].append(col)

        # 실제 존재하는 피처만 필터링
        for group in feature_groups:
            feature_groups[group] = [f for f in feature_groups[group] if f in actual_features]

        # ❗ object 타입 제거 (스케일러 에러 방지)
        for group in ['categorical_features', 'numerical_features']:
            feature_groups[group] = [col for col in feature_groups[group] if X_train[col].dtype != 'object']

        scaled_train_parts = []
        scaled_test_parts = []
        feature_order = []

        # 수치형 - RobustScaler
        if feature_groups['numerical_features']:
            self.scalers['numerical'] = RobustScaler()
            num_train = self.scalers['numerical'].fit_transform(X_train[feature_groups['numerical_features']])
            num_test = self.scalers['numerical'].transform(X_test[feature_groups['numerical_features']])
            scaled_train_parts.append(num_train)
            scaled_test_parts.append(num_test)
            feature_order.extend(feature_groups['numerical_features'])

        # 범주형 - StandardScaler
        if feature_groups['categorical_features']:
            self.scalers['categorical'] = StandardScaler()
            cat_train = self.scalers['categorical'].fit_transform(X_train[feature_groups['categorical_features']])
            cat_test = self.scalers['categorical'].transform(X_test[feature_groups['categorical_features']])
            scaled_train_parts.append(cat_train)
            scaled_test_parts.append(cat_test)
            feature_order.extend(feature_groups['categorical_features'])

        # Lag 피처 - MinMaxScaler
        if feature_groups['lag_features']:
            self.scalers['lag'] = MinMaxScaler()
            lag_train = self.scalers['lag'].fit_transform(X_train[feature_groups['lag_features']])
            lag_test = self.scalers['lag'].transform(X_test[feature_groups['lag_features']])
            scaled_train_parts.append(lag_train)
            scaled_test_parts.append(lag_test)
            feature_order.extend(feature_groups['lag_features'])

        # Rolling 피처 - RobustScaler
        if feature_groups['rolling_features']:
            self.scalers['rolling'] = RobustScaler()
            roll_train = self.scalers['rolling'].fit_transform(X_train[feature_groups['rolling_features']])
            roll_test = self.scalers['rolling'].transform(X_test[feature_groups['rolling_features']])
            scaled_train_parts.append(roll_train)
            scaled_test_parts.append(roll_test)
            feature_order.extend(feature_groups['rolling_features'])

        # 병합
        X_train_scaled = np.hstack(scaled_train_parts) if scaled_train_parts else X_train.values
        X_test_scaled = np.hstack(scaled_test_parts) if scaled_test_parts else X_test.values

        self.feature_order = feature_order
        return X_train_scaled, X_test_scaled

class ModelEnsemble:
    """앙상블 모델 클래스"""

    def __init__(self, config: Config):
        self.config = config
        self.base_models = {}
        self.meta_model = LinearRegression()
        self.lstm_model = None
        self.sarimax_model = None
        self.scalers = {}

    def create_base_models(self) -> Dict[str, Any]:
        """베이스 모델들 생성"""
        return {
            "xgb": XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.config.RANDOM_STATE, n_jobs=-1
            ),
            "lgb": LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.config.RANDOM_STATE, n_jobs=-1, verbose=-1
            ),
            "rf": RandomForestRegressor(
                n_estimators=400, max_depth=12, min_samples_split=5,
                random_state=self.config.RANDOM_STATE, n_jobs=-1
            )
        }

    def create_advanced_lstm(self, input_shape: Tuple[int, int], best_params: Dict) -> Model:
        """개선된 LSTM 아키텍처"""
        model = Sequential([
            LSTM(best_params["units1"], return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(best_params["dropout"]),
            LSTM(best_params["units2"], return_sequences=True),
            BatchNormalization(),
            Dropout(best_params["dropout"]),
            LSTM(best_params["units3"]),
            BatchNormalization(),
            Dropout(best_params["dropout"]),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=best_params["lr"]),
                      loss=self.custom_loss, metrics=['mae'])
        return model

    def custom_loss(self, y_true, y_pred):
        mse = tf.keras.losses.mse(y_true, y_pred)
        mae = tf.keras.losses.mae(y_true, y_pred)
        return 0.7 * mse + 0.3 * mae

    def optimize_lstm_hyperparameters(self, X_seq_train, y_seq_train, X_seq_val, y_seq_val) -> Dict:
        def objective(trial):
            units1 = trial.suggest_int("units1", 64, 256)
            units2 = trial.suggest_int("units2", 32, 128)
            units3 = trial.suggest_int("units3", 16, 64)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

            model = Sequential([
                LSTM(units1, return_sequences=True, input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
                Dropout(dropout),
                LSTM(units2, return_sequences=True),
                Dropout(dropout),
                LSTM(units3),
                Dropout(dropout),
                Dense(32, activation="relu"),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
            es = EarlyStopping(patience=3, restore_best_weights=True, verbose=0)

            model.fit(X_seq_train, y_seq_train,
                      validation_data=(X_seq_val, y_seq_val),
                      epochs=30, batch_size=64,
                      callbacks=[es], verbose=0)

            val_pred = model.predict(X_seq_val, verbose=0)
            return mean_absolute_error(y_seq_val, val_pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.N_TRIALS)
        return study.best_params

    def create_stacking_ensemble(self, X_train, y_train, X_val, y_val):
        """스태킹 앙상블 생성"""
        from sklearn.base import clone

        self.base_models = self.create_base_models()
        meta_features_train = np.zeros((len(X_train), len(self.base_models)))
        meta_features_val = np.zeros((len(X_val), len(self.base_models)))
        tscv = TimeSeriesSplit(n_splits=self.config.CV_FOLDS)

        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"🔧 Training base model: {name}")
            oof_preds = np.zeros(len(X_train))

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_tr, y_val_fold = y_train[train_idx], y_train[val_idx]

                cloned_model = clone(model)
                cloned_model.fit(X_tr, y_tr)
                oof_preds[val_idx] = cloned_model.predict(X_val_fold)

            meta_features_train[:, i] = oof_preds

            # 검증 데이터 예측용 전체 재학습
            model.fit(X_train, y_train)
            meta_features_val[:, i] = model.predict(X_val)

        # 메타 모델 훈련
        print("📈 Training meta model (LinearRegression)")
        self.meta_model.fit(meta_features_train, y_train)

        return meta_features_val

    def predict_stacking(self, X_test):
        """스태킹 앙상블 예측"""
        meta_features_test = np.zeros((len(X_test), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models.items()):
            meta_features_test[:, i] = model.predict(X_test)

        return self.meta_model.predict(meta_features_test)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

class MultiTaskModel:
    """멀티태스크 학습 모델 클래스"""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scalers = {}

    def create_multitask_model(self, input_shape: int, n_targets: int) -> Model:
        """공통 특성 추출기 + 각 타겟별 헤드를 갖는 멀티태스크 신경망 생성"""
        
        input_layer = Input(shape=(input_shape,), name='input_layer')

        # 🧱 Shared layers
        x = Dense(256, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # 🎯 Task-specific heads
        outputs = []
        for i in range(n_targets):
            head = Dense(32, activation='relu')(x)
            head = Dense(16, activation='relu')(head)
            head = Dense(1, name=f'target_{i}')(head)
            outputs.append(head)

        # 🧠 Model compile
        self.model = Model(inputs=input_layer, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return self.model


class ElectricityPredictor:
    """LS 전기요금 예측 전체 파이프라인"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.data_processor = DataProcessor(self.config)
        self.model_ensemble = ModelEnsemble(self.config)
        self.multitask_model = MultiTaskModel(self.config)
        self.results = {}
        self.feature_importance = {}

        os.makedirs(self.config.MODELS_DIR, exist_ok=True)

    def time_based_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_sorted = df.sort_values('측정일시').reset_index(drop=True)
        split_point = int(len(df_sorted) * (1 - self.config.TEST_SIZE))
        return df_sorted.iloc[:split_point].copy(), df_sorted.iloc[split_point:].copy()

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = self.data_processor.load_data()
        train_df = self.data_processor.create_datetime_features(train_df)
        test_df = self.data_processor.create_datetime_features(test_df)
        train_df = self.data_processor.create_tariff_features(train_df)
        test_df = self.data_processor.create_tariff_features(test_df)

        target_col = "전기요금(원)"
        train_df = self.data_processor.create_lag_features(train_df, target_col)
        train_df = self.data_processor.create_rolling_features(train_df, target_col)

        for col in train_df.columns:
            if col not in test_df.columns and ('_lag_' in col or '_rolling_' in col or '_pct_change_' in col):
                test_df[col] = np.nan

        train_df = self.data_processor.remove_outliers(train_df, target_col)

        for col in ['작업유형', '시간', '요일', '시간대']:
            train_df, test_df = self.data_processor.target_encoding(train_df, test_df, col, target=target_col)

        for col in ['작업유형', '시간대_세분화']:
            le = LabelEncoder()
            train_df[col + "_encoded"] = le.fit_transform(train_df[col])
            test_df[col + "_encoded"] = le.transform(test_df[col])

        return train_df, test_df

    def run(self):
        train_df, test_df = self.prepare_data()
        train_split, val_split = self.time_based_split(train_df)
        target_col = "전기요금(원)"

        X_train = train_split.drop(columns=self.config.MULTI_TARGETS + ['측정일시'])
        y_train = train_split[target_col]
        X_val = val_split.drop(columns=self.config.MULTI_TARGETS + ['측정일시'])
        y_val = val_split[target_col]
        X_test = test_df.drop(columns=['측정일시'])

        X_train_scaled, X_val_scaled = self.data_processor.scale_features(X_train, X_val)
        _, X_test_scaled = self.data_processor.scale_features(X_train, X_test)

        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).ravel()

        self.model_ensemble.create_stacking_ensemble(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)

        val_preds_scaled = self.model_ensemble.predict_stacking(X_val_scaled)
        val_preds = target_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).ravel()

        test_preds_scaled = self.model_ensemble.predict_stacking(X_test_scaled)
        test_preds = target_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).ravel()
        test_preds = np.clip(test_preds, 0, None)

        print("📊 MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}".format(
            mean_absolute_error(y_val, val_preds),
            np.sqrt(mean_squared_error(y_val, val_preds)),
            r2_score(y_val, val_preds)
        ))

        submission_df = pd.DataFrame({
            'id': test_df['id'],
            '예측 전기요금(원)': test_preds
        })

        save_path = os.path.join(self.config.MODELS_DIR, "test_predictions.csv")
        submission_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    predictor = ElectricityPredictor()
    predictor.run()
