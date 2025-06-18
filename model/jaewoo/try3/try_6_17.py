"""Optimal model for LS electric electricity bill prediction.

This script combines advanced preprocessing steps with multiple models:
- LSTM on 96x7 time windows
- Tree based models (XGB, LightGBM, RandomForest)
- Optional SARIMAX if statsmodels is installed

The final prediction is an ensemble of the available models.
"""

import os
import pickle
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
    STATS_AVAILABLE = False
except Exception:
    STATS_AVAILABLE = False
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
BASE_DIR = "../data"
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

# ----------------------------------------------------------------------
# 2. Datetime features
# ----------------------------------------------------------------------
for df in [train_df, test_df]:
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["요일"] = df["측정일시"].dt.weekday
    df["시간"] = df["측정일시"].dt.hour
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["주말여부"] = df["요일"].isin([0,6]).astype(int)
    df["sin_시간"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_시간"] = np.cos(2 * np.pi * df["시간"] / 24)
    LAG_STEP = 96 * 7  # 15분 간격 x 96 x 7일
    train_df["lag_7d_요금"] = train_df["전기요금(원)"].shift(LAG_STEP)

train_df.dropna(subset=["lag_7d_요금"], inplace=True)
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

# ----------------------------------------------------------------------
# 4. Encoding and target encoding
# ----------------------------------------------------------------------
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
q1 = train_df["전기요금(원)"].quantile(0.25)
q3 = train_df["전기요금(원)"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
train_df = train_df[(train_df["전기요금(원)"] >= lower) & (train_df["전기요금(원)"] <= upper)].copy()

# ----------------------------------------------------------------------
# 6. Feature selection
# ----------------------------------------------------------------------
FEATURES = [
    "작업유형_encoded",
    "월", "시간", "요일", "주말여부",
    "sin_시간", "cos_시간","lag_7d_요금"
]
TARGET = "전기요금(원)"

X = train_df[FEATURES]
y = train_df[TARGET]
test_df["lag_7d_요금"] = np.nan  # 초기화
X_test = test_df[FEATURES]

# ----------------------------------------------------------------------
# 7. Scaling
# ----------------------------------------------------------------------
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 8. Tree based models
# ----------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "xgb": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    "lgb": LGBMRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1),
    "rf": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
}

preds_val = {}
preds_test = {}
metrics = {}

print("Training tree models...")
n_models = len(models)
for idx, (name, model) in enumerate(models.items(), 1):
    print(f"[{idx}/{n_models}] {name} start")
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    preds_test[name] = model.predict(X_test_scaled)
    preds_val[name] = val_pred
    metrics[name] = r2_score(y_val, val_pred)
    print(f"[{idx}/{n_models}] {name} done")

# ----------------------------------------------------------------------
# 9. Optional SARIMAX
# ----------------------------------------------------------------------
if STATS_AVAILABLE:
    sarimax = SARIMAX(y, exog=X_scaled, order=(1,1,1), seasonal_order=(1,1,1,24))
    sarimax_fit = sarimax.fit(disp=False)
    val_pred = sarimax_fit.predict(start=len(y_train), end=len(y_train)+len(y_val)-1, exog=X_val)
    preds_val["sarimax"] = val_pred
    preds_test["sarimax"] = sarimax_fit.predict(start=len(X_scaled), end=len(X_scaled)+len(X_test_scaled)-1, exog=X_test_scaled)
    metrics["sarimax"] = r2_score(y_val, val_pred)
else:
    print("statsmodels not available - skipping SARIMAX")

# ----------------------------------------------------------------------
# 10. LSTM model
# ----------------------------------------------------------------------
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
split_idx = int(len(X_seq) * 0.8)
X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[split_idx:]
y_seq_train, y_seq_val = y_seq[:split_idx], y_seq[split_idx:]

lstm_model = Sequential([
    LSTM(64, input_shape=(TIME_STEPS, len(FEATURES))),
    Dense(32, activation="relu"),
    Dense(1),
])
lstm_model.compile(optimizer="adam", loss="mse")

es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
print("Training LSTM model...")
lstm_model.fit(
    X_seq_train,
    y_seq_train,
    validation_data=(X_seq_val, y_seq_val),
    epochs=20,
    batch_size=32,
    callbacks=[es],
    verbose=1,
)

val_pred = lstm_model.predict(X_seq_val).flatten()
metrics["lstm"] = r2_score(y_seq_val, val_pred)

# ✅ 12월 예측 시: lag가 없는 경우는 예측값으로 대체
def autoregressive_lstm_predict(model, train_df, test_df, features, scaler, time_steps):
    preds = []
    last_sequence = train_df[features + [TARGET]].iloc[-time_steps:].copy()

    for i in range(len(test_df)):
        row = test_df.iloc[i].copy()
        current_time = row["측정일시"]
        lag_time = current_time - pd.Timedelta(days=7)

        # 이전에 예측한 값 중 lag_time이 있으면 사용
        matched_idx = next((j for j, r in enumerate(test_df.iloc[:i].itertuples())
                            if r.측정일시 == lag_time), None)

        if matched_idx is not None:
            row["lag_7d_요금"] = preds[matched_idx]
        else:
            # train_df에서 가져올 수 있는 경우 사용
            past_match = train_df[train_df["측정일시"] == lag_time]
            if not past_match.empty:
                row["lag_7d_요금"] = past_match.iloc[0]["전기요금(원)"]
            else:
                row["lag_7d_요금"] = train_df["전기요금(원)"].mean()

        # 시퀀스에 새로운 row 추가
        combined = pd.concat([last_sequence, pd.DataFrame([row])], ignore_index=True)
        scaled = scaler.transform(combined[features + [TARGET]])
        seq_input = scaled[-time_steps:, :-1].reshape(1, time_steps, len(features))
        pred = model.predict(seq_input)[0, 0]
        preds.append(pred)

        # 예측값 삽입 후 시퀀스 갱신
        row[TARGET] = pred
        last_sequence = pd.concat([last_sequence, pd.DataFrame([row])], ignore_index=True).iloc[1:]

    return np.array(preds)


auto_lstm_preds = autoregressive_lstm_predict(
    model=lstm_model,
    train_df=train_df,
    test_df=test_df.copy(),
    features=FEATURES,
    scaler=seq_scaler,
    time_steps=TIME_STEPS
)
preds_test["lstm"] = auto_lstm_preds
preds_val["lstm"] = val_pred

dummy = np.zeros((len(auto_lstm_preds), len(FEATURES) + 1))  # +1은 TARGET(전기요금)
dummy[:, -1] = auto_lstm_preds  # 전기요금만 채우고 나머지는 0

# 2. 역변환 수행
inversed = seq_scaler.inverse_transform(dummy)[:, -1]  # 마지막 열만 추출 (전기요금)

# 3. 저장
pd.DataFrame({
    "id": test_df["id"],
    "예측_전기요금(원)": inversed
}).to_csv("lstm_unscaled.csv", index=False)
# ----------------------------------------------------------------------
# 11. GRU model
# ----------------------------------------------------------------------
gru_model = Sequential([
    GRU(64, input_shape=(TIME_STEPS, len(FEATURES))),
    Dense(32, activation="relu"),
    Dense(1),
])
gru_model.compile(optimizer="adam", loss="mse")

print("Training GRU model...")
gru_model.fit(
    X_seq_train,
    y_seq_train,
    validation_data=(X_seq_val, y_seq_val),
    epochs=20,
    batch_size=32,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    verbose=1,
)

# Validation 예측 및 R2 기록
val_pred_gru = gru_model.predict(X_seq_val).flatten()
metrics["gru"] = r2_score(y_seq_val, val_pred_gru)
preds_val["gru"] = val_pred_gru

def autoregressive_gru_predict(model, train_df, test_df, features, scaler, time_steps):
    preds = []
    last_sequence = train_df[features + [TARGET]].iloc[-time_steps:].copy()

    for i in range(len(test_df)):
        row = test_df.iloc[i].copy()
        current_time = row["측정일시"]
        lag_time = current_time - pd.Timedelta(days=7)

        matched_idx = next((j for j, r in enumerate(test_df.iloc[:i].itertuples())
                            if r.측정일시 == lag_time), None)

        if matched_idx is not None:
            row["lag_7d_요금"] = preds[matched_idx]
        else:
            past_match = train_df[train_df["측정일시"] == lag_time]
            if not past_match.empty:
                row["lag_7d_요금"] = past_match.iloc[0]["전기요금(원)"]
            else:
                row["lag_7d_요금"] = train_df["전기요금(원)"].mean()

        combined = pd.concat([last_sequence, pd.DataFrame([row])], ignore_index=True)
        scaled = scaler.transform(combined[features + [TARGET]])
        seq_input = scaled[-time_steps:, :-1].reshape(1, time_steps, len(features))
        pred = model.predict(seq_input)[0, 0]
        preds.append(pred)

        row[TARGET] = pred
        last_sequence = pd.concat([last_sequence, pd.DataFrame([row])], ignore_index=True).iloc[1:]

    return np.array(preds)

# 예측 및 역변환
auto_gru_preds = autoregressive_gru_predict(
    model=gru_model,
    train_df=train_df,
    test_df=test_df.copy(),
    features=FEATURES,
    scaler=seq_scaler,
    time_steps=TIME_STEPS
)
preds_test["gru"] = auto_gru_preds

# 역변환 후 저장
dummy_gru = np.zeros((len(auto_gru_preds), len(FEATURES) + 1))
dummy_gru[:, -1] = auto_gru_preds
inversed_gru = seq_scaler.inverse_transform(dummy_gru)[:, -1]

pd.DataFrame({
    "id": test_df["id"],
    "예측_전기요금_GRU(원)": inversed_gru
}).to_csv("gru_unscaled.csv", index=False)
# ----------------------------------------------------------------------
# 12. Weighted ensemble
# ----------------------------------------------------------------------
weights = {name: max(score, 0) for name, score in metrics.items()}
total = sum(weights.values())
if total == 0:
    weights = {name: 1/len(weights) for name in weights}
else:
    weights = {name: w/total for name, w in weights.items()}

# Determine correct validation target
is_seq = "lstm" in preds_val
y_true = y_seq_val if is_seq else y_val
target_len = len(y_true)

# Initialize
val_ens = np.zeros(target_len)

# Ensemble
for name, pred in preds_val.items():
    if len(pred) != target_len:
        print(f"⚠️ {name} prediction length mismatch: {len(pred)} vs {target_len}")
        continue
    val_ens += weights[name] * pred

# Evaluate
ens_r2 = r2_score(y_true, val_ens)
print("✅ Ensemble R2:", round(ens_r2, 4))

# Test prediction
test_pred = np.zeros(len(test_df))
for name, pred in preds_test.items():
    test_pred += weights[name] * pred


submission = pd.DataFrame({"id": test_df["id"], "전기요금(원)": test_pred})
submission.to_csv("submission_optimal.csv", index=False)
submission.to_csv("submission.csv", index=False)
print("Saved submission_optimal.csv and submission.csv")

# ----------------------------------------------------------------------
# 13. Save trained models
# ----------------------------------------------------------------------
MODELS_DIR = "pickles"
os.makedirs(MODELS_DIR, exist_ok=True)
print("Saving trained models...")
for name, model in models.items():
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {name}.pkl")
if STATS_AVAILABLE:
    with open(os.path.join(MODELS_DIR, "sarimax.pkl"), "wb") as f:
        pickle.dump(sarimax_fit, f)
    print("Saved sarimax.pkl")
with open(os.path.join(MODELS_DIR, "lstm.pkl"), "wb") as f:
    pickle.dump(lstm_model, f)
print("Saved lstm.pkl")
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("Saved scaler.pkl")
with open(os.path.join(MODELS_DIR, "seq_scaler.pkl"), "wb") as f:
    pickle.dump(seq_scaler, f)
print("Saved seq_scaler.pkl")
print(f"Saved trained models to {MODELS_DIR}")




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


