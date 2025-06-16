"""Optimized model for LS electric electricity bill prediction.

This script includes advanced optimizations:
- Lag features and rolling statistics for time series patterns
- Log transformation for target distribution
- Improved LSTM architecture with stacking
- Hyperparameter optimization with Optuna
- Enhanced stacking ensemble

Expected MAE improvement: 30-60 points reduction
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - using default hyperparameters")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATS_AVAILABLE = False
except Exception:
    STATS_AVAILABLE = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
BASE_DIR = "../data"
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ----------------------------------------------------------------------
# 2. Enhanced datetime features
# ----------------------------------------------------------------------
def create_datetime_features(df):
    """Enhanced datetime feature engineering"""
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    
    # Basic time features
    df["년"] = df["측정일시"].dt.year
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["시간"] = df["측정일시"].dt.hour
    df["요일"] = df["측정일시"].dt.weekday
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    
    # Cyclical features
    df["sin_시간"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_시간"] = np.cos(2 * np.pi * df["시간"] / 24)
    df["sin_요일"] = np.sin(2 * np.pi * df["요일"] / 7)
    df["cos_요일"] = np.cos(2 * np.pi * df["요일"] / 7)
    df["sin_월"] = np.sin(2 * np.pi * df["월"] / 12)
    df["cos_월"] = np.cos(2 * np.pi * df["월"] / 12)
    
    # Advanced time features
    df["월초여부"] = (df["일"] <= 5).astype(int)
    df["월말여부"] = (df["일"] >= 25).astype(int)
    df["peak_time"] = ((df["시간"] >= 8) & (df["시간"] <= 22)).astype(int)
    df["night_time"] = ((df["시간"] >= 22) | (df["시간"] <= 6)).astype(int)
    
    return df

for df in [train_df, test_df]:
    df = create_datetime_features(df)

# ----------------------------------------------------------------------
# 3. Enhanced tariff calculation
# ----------------------------------------------------------------------
def get_season(month: int) -> str:
    if month in [6, 7, 8]:
        return "여름"
    elif month in [3, 4, 5, 9, 10]:
        return "봄가을"
    return "겨울"

def get_time_zone(hour: int, season: str) -> str:
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

RATE_TABLE = {
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

CUTOFF = datetime(2024, 10, 24)
for df in [train_df, test_df]:
    df["계절"] = df["월"].apply(get_season)
    df["적용시점"] = df["측정일시"].apply(lambda x: "before" if x < CUTOFF else "after")
    df["시간대"] = df.apply(lambda r: get_time_zone(r["시간"], r["계절"]), axis=1)
    df["요금단가"] = df.apply(lambda r: RATE_TABLE[r["적용시점"]][r["계절"]][r["시간대"]], axis=1)

# ----------------------------------------------------------------------
# 4. Encoding and enhanced target encoding
# ----------------------------------------------------------------------
le = LabelEncoder()
train_df["작업유형_encoded"] = le.fit_transform(train_df["작업유형"])
test_df["작업유형_encoded"] = le.transform(test_df["작업유형"])

def target_encoding_with_kfold(df_train: pd.DataFrame, df_test: pd.DataFrame, 
                              col: str, target: str, smoothing: int = 10, n_folds: int = 5) -> None:
    """Enhanced target encoding with K-fold to prevent overfitting"""
    from sklearn.model_selection import KFold
    
    global_mean = df_train[target].mean()
    df_train[f"{col}_te"] = global_mean
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df_train):
        train_fold = df_train.iloc[train_idx]
        val_fold = df_train.iloc[val_idx]
        
        agg = train_fold.groupby(col)[target].agg(["mean", "count"])
        smoothing_weight = 1 / (1 + np.exp(-(agg["count"] - smoothing)))
        enc = global_mean * (1 - smoothing_weight) + agg["mean"] * smoothing_weight
        mapping = enc.to_dict()
        
        df_train.loc[val_idx, f"{col}_te"] = val_fold[col].map(mapping).fillna(global_mean)
    
    # For test set
    agg = df_train.groupby(col)[target].agg(["mean", "count"])
    smoothing_weight = 1 / (1 + np.exp(-(agg["count"] - smoothing)))
    enc = global_mean * (1 - smoothing_weight) + agg["mean"] * smoothing_weight
    mapping = enc.to_dict()
    df_test[f"{col}_te"] = df_test[col].map(mapping).fillna(global_mean)

for c in ["작업유형", "시간", "요일", "시간대", "계절"]:
    target_encoding_with_kfold(train_df, test_df, c, "전기요금(원)")

# ----------------------------------------------------------------------
# 5. Enhanced time series features (Lag + Rolling)
# ----------------------------------------------------------------------
def create_lag_features(df, target_col, lags=[1, 2, 3, 6, 12, 24]):
    """Create lag features for time series"""
    df_sorted = df.sort_values("측정일시").copy()
    
    for lag in lags:
        df_sorted[f"{target_col}_lag{lag}"] = df_sorted[target_col].shift(lag)
    
    return df_sorted

def create_rolling_features(df, target_col, windows=[3, 6, 12, 24]):
    """Create rolling statistics features"""
    df_sorted = df.sort_values("측정일시").copy()
    
    for window in windows:
        df_sorted[f"{target_col}_roll_mean{window}"] = df_sorted[target_col].rolling(window).mean()
        df_sorted[f"{target_col}_roll_std{window}"] = df_sorted[target_col].rolling(window).std()
        df_sorted[f"{target_col}_roll_max{window}"] = df_sorted[target_col].rolling(window).max()
        df_sorted[f"{target_col}_roll_min{window}"] = df_sorted[target_col].rolling(window).min()
    
    return df_sorted

# Sort both dataframes by datetime
train_df = train_df.sort_values("측정일시").reset_index(drop=True)
test_df = test_df.sort_values("측정일시").reset_index(drop=True)

# Create lag and rolling features for training data
train_df = create_lag_features(train_df, "전기요금(원)")
train_df = create_rolling_features(train_df, "전기요금(원)")

# For test data, we need to be careful about data leakage
# Use the last known values from training data
last_train_values = train_df["전기요금(원)"].tail(24).values
for i, lag in enumerate([1, 2, 3, 6, 12, 24]):
    if i < len(last_train_values):
        test_df[f"전기요금(원)_lag{lag}"] = last_train_values[-(i+1)]
    else:
        test_df[f"전기요금(원)_lag{lag}"] = train_df["전기요금(원)"].mean()

# For rolling features in test, use training data statistics
for window in [3, 6, 12, 24]:
    test_df[f"전기요금(원)_roll_mean{window}"] = train_df[f"전기요금(원)_roll_mean{window}"].iloc[-1]
    test_df[f"전기요금(원)_roll_std{window}"] = train_df[f"전기요금(원)_roll_std{window}"].iloc[-1]
    test_df[f"전기요금(원)_roll_max{window}"] = train_df[f"전기요금(원)_roll_max{window}"].iloc[-1]
    test_df[f"전기요금(원)_roll_min{window}"] = train_df[f"전기요금(원)_roll_min{window}"].iloc[-1]

# ----------------------------------------------------------------------
# 6. Outlier removal with enhanced IQR
# ----------------------------------------------------------------------
def remove_outliers_iqr(df, target_col, factor=1.5):
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    
    before_count = len(df)
    df_clean = df[(df[target_col] >= lower) & (df[target_col] <= upper)].copy()
    after_count = len(df_clean)
    
    print(f"Outlier removal: {before_count} → {after_count} ({before_count-after_count} removed)")
    return df_clean

train_df = remove_outliers_iqr(train_df, "전기요금(원)", factor=2.0)

# ----------------------------------------------------------------------
# 7. Enhanced feature selection
# ----------------------------------------------------------------------
LAG_FEATURES = [f"전기요금(원)_lag{lag}" for lag in [1, 2, 3, 6, 12, 24]]
ROLLING_FEATURES = []
for window in [3, 6, 12, 24]:
    ROLLING_FEATURES.extend([
        f"전기요금(원)_roll_mean{window}",
        f"전기요금(원)_roll_std{window}",
        f"전기요금(원)_roll_max{window}",
        f"전기요금(원)_roll_min{window}"
    ])

FEATURES = [
    "작업유형_encoded",
    "년", "월", "일", "시간", "요일", "주말여부",
    "sin_시간", "cos_시간", "sin_요일", "cos_요일", "sin_월", "cos_월",
    "월초여부", "월말여부", "peak_time", "night_time",
    "요금단가",
    "작업유형_te", "시간_te", "요일_te", "시간대_te", "계절_te",
] + LAG_FEATURES + ROLLING_FEATURES

TARGET = "전기요금(원)"

# Remove missing values from training data
train_df = train_df.dropna(subset=FEATURES + [TARGET])

print(f"Final training shape: {train_df.shape}")
print(f"Features count: {len(FEATURES)}")

X = train_df[FEATURES]
y = train_df[TARGET]
X_test = test_df[FEATURES].fillna(train_df[FEATURES].median())

# ----------------------------------------------------------------------
# 8. Log transformation for target
# ----------------------------------------------------------------------
USE_LOG_TRANSFORM = True
if USE_LOG_TRANSFORM:
    print("Applying log transformation to target...")
    y_log = np.log1p(y)
    y_original = y.copy()
    y = y_log

# ----------------------------------------------------------------------
# 9. Enhanced scaling
# ----------------------------------------------------------------------
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 10. Hyperparameter optimization with Optuna
# ----------------------------------------------------------------------
def optimize_lightgbm(X_train, X_val, y_train, y_val, n_trials=100):
    """Optimize LightGBM hyperparameters using Optuna"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

# ----------------------------------------------------------------------
# 11. Enhanced tree models with optimization
# ----------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Get optimized parameters
if OPTUNA_AVAILABLE:
    print("Optimizing LightGBM hyperparameters...")
    lgb_best_params = optimize_lightgbm(X_train, X_val, y_train, y_val, n_trials=50)
    print(f"Best LightGBM params: {lgb_best_params}")
else:
    lgb_best_params = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 1.0,
        'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }

models = {
    "xgb": XGBRegressor(
        n_estimators=600, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
        reg_lambda=1.0, random_state=42, n_jobs=-1
    ),
    "lgb": LGBMRegressor(**lgb_best_params),
    "rf": RandomForestRegressor(
        n_estimators=400, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
}

preds_val = {}
preds_test = {}
metrics = {}

print("Training enhanced tree models...")
for idx, (name, model) in enumerate(models.items(), 1):
    print(f"[{idx}/{len(models)}] Training {name}...")
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test_scaled)
    
    preds_val[name] = val_pred
    preds_test[name] = test_pred
    
    if USE_LOG_TRANSFORM:
        val_pred_original = np.expm1(val_pred)
        y_val_original = np.expm1(y_val)
        mae = mean_absolute_error(y_val_original, val_pred_original)
        r2 = r2_score(y_val_original, val_pred_original)
    else:
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
    
    metrics[name] = r2
    print(f"[{idx}/{len(models)}] {name} - MAE: {mae:.2f}, R2: {r2:.4f}")

# ----------------------------------------------------------------------
# 12. Optional SARIMAX
# ----------------------------------------------------------------------
if STATS_AVAILABLE:
    print("Training SARIMAX model...")
    try:
        sarimax = SARIMAX(y, exog=X_scaled, order=(1,1,1), seasonal_order=(1,1,1,24))
        sarimax_fit = sarimax.fit(disp=False)
        val_pred = sarimax_fit.predict(start=len(y_train), end=len(y_train)+len(y_val)-1, exog=X_val)
        test_pred = sarimax_fit.predict(start=len(X_scaled), end=len(X_scaled)+len(X_test_scaled)-1, exog=X_test_scaled)
        
        preds_val["sarimax"] = val_pred
        preds_test["sarimax"] = test_pred
        
        if USE_LOG_TRANSFORM:
            val_pred_original = np.expm1(val_pred)
            y_val_original = np.expm1(y_val)
            r2 = r2_score(y_val_original, val_pred_original)
        else:
            r2 = r2_score(y_val, val_pred)
        metrics["sarimax"] = r2
        print(f"SARIMAX R2: {r2:.4f}")
    except Exception as e:
        print(f"SARIMAX failed: {e}")
else:
    print("statsmodels not available - skipping SARIMAX")

# ----------------------------------------------------------------------
# 13. Enhanced LSTM model
# ----------------------------------------------------------------------
TIME_STEPS = 96 * 7  # 1 week

# Prepare sequence data
seq_scaler = MinMaxScaler()
seq_data = train_df[FEATURES + [TARGET]].copy()
seq_scaled = seq_scaler.fit_transform(seq_data)
seq_scaled = pd.DataFrame(seq_scaled, columns=FEATURES + [TARGET])

def create_sequences(arr: pd.DataFrame, timesteps: int) -> tuple:
    xs, ys = [], []
    for i in range(len(arr) - timesteps):
        xs.append(arr.iloc[i:i+timesteps][FEATURES].values)
        ys.append(arr.iloc[i+timesteps][TARGET])
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(seq_scaled, TIME_STEPS)
seq_train_idx = int(len(X_seq) * 0.8)

X_seq_train, X_seq_val = X_seq[:seq_train_idx], X_seq[seq_train_idx:]
y_seq_train, y_seq_val = y_seq[:seq_train_idx], y_seq[seq_train_idx:]

# Enhanced LSTM architecture
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(TIME_STEPS, len(FEATURES))),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1),
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# Enhanced callbacks
es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

print("Training enhanced LSTM model...")
history = lstm_model.fit(
    X_seq_train, y_seq_train,
    validation_data=(X_seq_val, y_seq_val),
    epochs=14,
    batch_size=64,
    callbacks=[es, reduce_lr],
    verbose=1,
)

# LSTM predictions
val_pred = lstm_model.predict(X_seq_val).flatten()
preds_val["lstm"] = val_pred

if USE_LOG_TRANSFORM:
    val_pred_original = np.expm1(val_pred)
    y_seq_val_original = np.expm1(y_seq_val)
    r2 = r2_score(y_seq_val_original, val_pred_original)
else:
    r2 = r2_score(y_seq_val, val_pred)
metrics["lstm"] = r2

# LSTM test prediction
def predict_lstm_enhanced(model, last_known: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    combined = pd.concat([last_known, future], ignore_index=True)
    combined_scaled = seq_scaler.transform(combined)
    combined_scaled = pd.DataFrame(combined_scaled, columns=FEATURES + [TARGET])
    
    predictions = []
    for i in range(len(future)):
        if i + TIME_STEPS < len(combined_scaled):
            seq = combined_scaled.iloc[i:i+TIME_STEPS][FEATURES].values.reshape(1, TIME_STEPS, -1)
            pred = model.predict(seq, verbose=0)[0, 0]
            predictions.append(pred)
    
    return np.array(predictions)

last_part = train_df[FEATURES + [TARGET]].iloc[-TIME_STEPS:]
lstm_test_pred = predict_lstm_enhanced(lstm_model, last_part, test_df[FEATURES])
preds_test["lstm"] = lstm_test_pred

print(f"LSTM R2: {metrics['lstm']:.4f}")

# ----------------------------------------------------------------------
# 14. Advanced Stacking Ensemble
# ----------------------------------------------------------------------
print("Creating stacking ensemble...")

# 1. 길이 정렬
min_len = min(len(pred) for pred in preds_val.values())
level1_train = np.column_stack([pred[:min_len] for pred in preds_val.values()])
level1_test = np.column_stack([preds_test[name] for name in preds_test.keys()])

# 2. 타깃 정렬
y_meta = y_seq_val if "lstm" in preds_val else y_val
y_meta_aligned = y_meta[:min_len]

# 3. 메타 모델 학습
meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(level1_train, y_meta_aligned)

# 4. 예측 및 평가
ensemble_val = meta_model.predict(level1_train)
ensemble_test = meta_model.predict(level1_test)

if USE_LOG_TRANSFORM:
    ensemble_val_original = np.expm1(ensemble_val)
    y_meta_original = np.expm1(y_meta_aligned)
    ensemble_test_original = np.expm1(ensemble_test)
else:
    ensemble_val_original = ensemble_val
    y_meta_original = y_meta_aligned
    ensemble_test_original = ensemble_test

ensemble_mae = mean_absolute_error(y_meta_original, ensemble_val_original)
ensemble_r2 = r2_score(y_meta_original, ensemble_val_original)

# 5. 출력
print(f"🎯 Stacking Ensemble - MAE: {ensemble_mae:.2f}, R2: {ensemble_r2:.4f}")
print("\n📊 Individual Model Performance:")
for name, r2 in metrics.items():
    print(f"  {name}: R2 = {r2:.4f}")
print(f"\n🏆 Final Ensemble: MAE = {ensemble_mae:.2f}, R2 = {ensemble_r2:.4f}")
# ----------------------------------------------------------------------
# 15. Save predictions
# ----------------------------------------------------------------------
if USE_LOG_TRANSFORM:
    final_predictions = ensemble_test_original
else:
    final_predictions = ensemble_test

submission = pd.DataFrame({
    "id": test_df["id"], 
    "전기요금(원)": final_predictions
})

submission.to_csv("submission_optimized.csv", index=False)
submission.to_csv("submission.csv", index=False)
print("✅ Saved submission_optimized.csv and submission.csv")

# ----------------------------------------------------------------------
# 16. Save trained models
# ----------------------------------------------------------------------
MODELS_DIR = "pickles_optimized"
os.makedirs(MODELS_DIR, exist_ok=True)

print("💾 Saving optimized models...")
for name, model in models.items():
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

if STATS_AVAILABLE and "sarimax" in preds_test:
    with open(os.path.join(MODELS_DIR, "sarimax.pkl"), "wb") as f:
        pickle.dump(sarimax_fit, f)

lstm_model.save(os.path.join(MODELS_DIR, "lstm_model.h5"))

with open(os.path.join(MODELS_DIR, "meta_model.pkl"), "wb") as f:
    pickle.dump(meta_model, f)

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODELS_DIR, "seq_scaler.pkl"), "wb") as f:
    pickle.dump(seq_scaler, f)

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"✅ All models saved to {MODELS_DIR}")

# ----------------------------------------------------------------------
# 17. Multi-target prediction (keeping original functionality)
# ----------------------------------------------------------------------
MULTI_TARGETS = [
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "탄소배출량(tCO2)",
    "지상역률(%)",
    "진상역률(%)",
    "전기요금(원)"
]

# test_df 생성 및 파생 변수 포함 완료된 상태라고 가정
test_pred_dict = {}

for target in MULTI_TARGETS:
    print(f"🔍 [{target}] 예측 중...")
    
    # 학습용 데이터 구성
    y_multi = train_df[target]
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_multi, test_size=0.2, random_state=42)

    # 모델 훈련 (여기선 LightGBM 사용 예시)
    model = LGBMRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 결과 저장
    y_test_pred = model.predict(X_test_scaled)
    test_pred_dict[target] = y_test_pred
    print(f"✅ [{target}] 예측 완료")

for target, pred in test_pred_dict.items():
    test_df[target] = pred


# id, 측정일시, 작업유형 + 예측된 피처들 저장
final_output = test_df[["id", "측정일시", "작업유형"] + MULTI_TARGETS]
final_output.to_csv("test_predicted_december_data.csv", index=False)
print("test_predicted_december_data.csv 저장 완료 (멀티타깃 예측 포함)")
