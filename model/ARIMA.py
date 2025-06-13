
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATS_AVAILABLE = True
except Exception:
    STATS_AVAILABLE = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
BASE_DIR = "../data"
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
print("✔️ [1] 데이터 로딩 중...")

# ----------------------------------------------------------------------
# 2. Datetime features
# ----------------------------------------------------------------------
for df in [train_df, test_df]:
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["시간"] = df["측정일시"].dt.hour
    df["요일"] = df["측정일시"].dt.weekday
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["sin_시간"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_시간"] = np.cos(2 * np.pi * df["시간"] / 24)

print("✔️ [2] 날짜 변수 처리 중...")

# ----------------------------------------------------------------------
# 3. Tariff calculation
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

print("✔️ [3] 계절/시간대/요금단가 처리 중...")

# ----------------------------------------------------------------------
# 4. Encoding and target encoding
# ----------------------------------------------------------------------

print("✔️ [4] 라벨 인코딩 및 타겟 인코딩 중...")

le = LabelEncoder()
train_df["작업유형_encoded"] = le.fit_transform(train_df["작업유형"])
test_df["작업유형_encoded"] = le.transform(test_df["작업유형"])


def target_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame, col: str, target: str, smoothing: int = 10) -> None:
    global_mean = df_train[target].mean()
    agg = df_train.groupby(col)[target].agg(["mean", "count"])
    smoothing_weight = 1 / (1 + np.exp(-(agg["count"] - smoothing)))
    enc = global_mean * (1 - smoothing_weight) + agg["mean"] * smoothing_weight
    mapping = enc.to_dict()
    df_train[f"{col}_te"] = df_train[col].map(mapping)
    df_test[f"{col}_te"] = df_test[col].map(mapping)


for c in ["작업유형", "시간", "요일", "시간대"]:
    target_encoding(train_df, test_df, c, "전기요금(원)")


# ----------------------------------------------------------------------
# 5. Outlier removal with IQR
# ----------------------------------------------------------------------
print("✔️ [5] 이상치 제거 중...")

q1 = train_df["전기요금(원)"].quantile(0.25)
q3 = train_df["전기요금(원)"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
train_df = train_df[(train_df["전기요금(원)"] >= lower) & (train_df["전기요금(원)"] <= upper)].copy()

# ----------------------------------------------------------------------
# 6. Feature selection
# ----------------------------------------------------------------------
print("✔️ [6] 피처 선택 및 정의 완료")

FEATURES = [
    "작업유형_encoded",
    "월", "일", "시간", "요일", "주말여부",
    "sin_시간", "cos_시간",
    "요금단가",
    "작업유형_te", "시간_te", "요일_te", "시간대_te",
]
TARGET = "전기요금(원)"

X = train_df[FEATURES]
y = train_df[TARGET]
X_test = test_df[FEATURES]

# ----------------------------------------------------------------------
# 7. Scaling
# ----------------------------------------------------------------------
print("✔️ [7] 스케일링 적용 중 (RobustScaler)...")

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 8. Tree based models
# ----------------------------------------------------------------------
print("✔️ [8] 트리 기반 모델 학습 시작")

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "xgb": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    "lgb": LGBMRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1),
    "rf": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
}

preds_val = {}
preds_test = {}
metrics = {}

for name, model in models.items():
    print(f"  🔁 {name.upper()} 모델 학습 중...")
    model.fit(X_train, y_train)
    print(f"  ✅ {name.upper()} 학습 완료, R2: {r2_score(y_val, model.predict(X_val)):.4f}")
    val_pred = model.predict(X_val)
    preds_test[name] = model.predict(X_test_scaled)
    preds_val[name] = val_pred
    metrics[name] = r2_score(y_val, val_pred)

# ----------------------------------------------------------------------
# 9. Optional SARIMAX
# ----------------------------------------------------------------------
print("✔️ [9] SARIMAX 학습 시도 중...")
if STATS_AVAILABLE:
    print("  🔁 SARIMAX 모델 학습 시작")
    sarimax = SARIMAX(y, exog=X_scaled, order=(1,1,1), seasonal_order=(1,1,1,24))
    sarimax_fit = sarimax.fit(disp=False)
    val_pred = sarimax_fit.predict(start=len(y_train), end=len(y_train)+len(y_val)-1, exog=X_val)
    preds_val["sarimax"] = val_pred
    preds_test["sarimax"] = sarimax_fit.predict(start=len(X_scaled), end=len(X_scaled)+len(X_test_scaled)-1, exog=X_test_scaled)
    metrics["sarimax"] = r2_score(y_val, val_pred)
    print(f"  ✅ SARIMAX R2: {metrics['sarimax']:.4f}")
else:
    print("statsmodels not available - skipping SARIMAX")

# ----------------------------------------------------------------------
# 10. LSTM model
# ----------------------------------------------------------------------
print("✔️ [10] LSTM 시퀀스 데이터 준비 중...")

TIME_STEPS = 96 * 7

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
seq_train, seq_val = int(len(X_seq) * 0.8), int(len(X_seq) * 0.8)
X_seq_train, X_seq_val = X_seq[:seq_train], X_seq[seq_train:]
y_seq_train, y_seq_val = y_seq[:seq_train], y_seq[seq_train:]

print("  🔁 LSTM 모델 학습 시작 (최대 20 epoch)...")

lstm_model = Sequential([
    LSTM(64, input_shape=(TIME_STEPS, len(FEATURES))),
    Dense(32, activation="relu"),
    Dense(1),
])
lstm_model.compile(optimizer="adam", loss="mse")

es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lstm_model.fit(X_seq_train, y_seq_train, validation_data=(X_seq_val, y_seq_val), epochs=20, batch_size=32, callbacks=[es], verbose=0)

val_pred = lstm_model.predict(X_seq_val).flatten()
metrics["lstm"] = r2_score(y_seq_val, val_pred)

print(f"  ✅ LSTM R2: {metrics['lstm']:.4f}")

def predict_lstm(model, last_known: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    combined = pd.concat([last_known, future], ignore_index=True)
    combined_scaled = seq_scaler.transform(combined)
    combined_scaled = pd.DataFrame(combined_scaled, columns=FEATURES + [TARGET])
    seqs = [combined_scaled.iloc[i:i+TIME_STEPS][FEATURES].values for i in range(len(combined_scaled) - TIME_STEPS)]
    seqs = np.array(seqs)
    preds = model.predict(seqs).flatten()
    return preds[-len(future):]


last_part = train_df[FEATURES + [TARGET]].iloc[-TIME_STEPS:]
lstm_test_pred = predict_lstm(lstm_model, last_part, test_df[FEATURES])

preds_test["lstm"] = lstm_test_pred
preds_val["lstm"] = val_pred

# ----------------------------------------------------------------------
# 11. Weighted ensemble
# ----------------------------------------------------------------------
print("✔️ [11] 앙상블 가중치 계산 중...")

weights = {name: max(score, 0) for name, score in metrics.items()}
total = sum(weights.values())
if total == 0:
    weights = {name: 1/len(weights) for name in weights}
else:
    weights = {name: w/total for name, w in weights.items()}

val_ens = np.zeros_like(list(preds_val.values())[0])
for name, pred in preds_val.items():
    val_ens += weights[name] * pred

ens_r2 = r2_score(y_seq_val if "lstm" in preds_val else y_val, val_ens)
print("Ensemble R2:", round(ens_r2, 4))

# Test prediction
test_pred = np.zeros(len(test_df))
for name, pred in preds_test.items():
    test_pred += weights[name] * pred

submission = pd.DataFrame({"id": test_df["id"], "전기요금(원)": test_pred})
submission.to_csv("submission_optimal.csv", index=False)
print("Saved submission_optimal.csv")
