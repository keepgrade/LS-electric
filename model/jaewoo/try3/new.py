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
from scipy.ndimage import uniform_filter1d
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATS_AVAILABLE = False
except Exception:
    STATS_AVAILABLE = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
BASE_DIR = "../data"
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
# train_df["측정일시"] = pd.to_datetime(train_df["측정일시"])
# test_df = train_df[(train_df["측정일시"] >= "2024-11-01") & (train_df["측정일시"] < "2024-12-01")]
# answer = test_df[["id","전기요금(원)"]]
# test_df = test_df[["측정일시","id","작업유형"]]
# train_df = train_df[train_df["측정일시"] < "2024-11-01"]
# ----------------------------------------------------------------------
# 2. Datetime features
# ----------------------------------------------------------------------
for df in [train_df, test_df]:
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["월"] = df["측정일시"].dt.month
    df["시간"] = df["측정일시"].dt.hour
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.weekday
    df["주말여부"] = df["요일"].isin([0,6]).astype(int)
    df["sin_시간"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_시간"] = np.cos(2 * np.pi * df["시간"] / 24)

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
q1 = train_df["전기요금(원)"].quantile(0.05)
q3 = train_df["전기요금(원)"].quantile(0.95)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
train_df = train_df[(train_df["전기요금(원)"] >= lower) & (train_df["전기요금(원)"] <= upper)].copy()

# ----------------------------------------------------------------------
# 6. Feature selection
# ----------------------------------------------------------------------
FEATURES = [
    "작업유형_encoded",
    "월", "요일", "주말여부",
    "sin_시간", "cos_시간",
    "요금단가",
   "시간_te"
]
TARGET = "전기요금(원)"

X = train_df[FEATURES]
y = train_df[TARGET]
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

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# 탐색 범위 정의
param_grids = {
    "xgb": {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.03, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "lgb": {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.03, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "rf": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "max_features": ["sqrt", "log2", None],
    }
}

# 모델 정의
model_classes = {
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
    "lgb": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "rf": RandomForestRegressor(random_state=42, n_jobs=-1)
}

# 최적 모델 학습 및 저장
best_models = {}
preds_val = {}
preds_test = {}
metrics = {}

for name in ["xgb", "lgb", "rf"]:
    print(f"🔍 Tuning {name.upper()}...")
    search = RandomizedSearchCV(
        model_classes[name],
        param_distributions=param_grids[name],
        n_iter=10,
        scoring=mae_scorer,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_models[name] = best_model

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test_scaled)

    preds_val[name] = val_pred
    preds_test[name] = test_pred
    metrics[name] = r2_score(y_val, val_pred)

    print(f"✅ {name.upper()} Best Params: {search.best_params_}")
    print(f"📊 {name.upper()} R²: {metrics[name]:.4f}")

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
FEATURES = [
    "작업유형_encoded",
    "월","요일", "주말여부",
    "sin_시간", "cos_시간"
]
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

dummy_val = np.zeros((len(val_pred), len(FEATURES) + 1))
dummy_val[:, -1] = val_pred
val_pred_inversed = seq_scaler.inverse_transform(dummy_val)[:, -1]

dummy_true = np.zeros((len(y_seq_val), len(FEATURES) + 1))
dummy_true[:, -1] = y_seq_val
y_true_inversed = seq_scaler.inverse_transform(dummy_true)[:, -1]

residual_val = y_true_inversed - val_pred_inversed

val_features = seq_scaled.iloc[-len(val_pred):][FEATURES].copy()
val_features["lstm_pred"] = val_pred_inversed
residual_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
residual_model.fit(val_features, residual_val)

test_features = test_df[FEATURES].copy()


dummy1 = np.zeros((len(val_pred), len(FEATURES) + 1))  # +1은 TARGET(전기요금)
dummy1[:, -1] = val_pred  # 전기요금만 채우고 나머지는 0

# 2. 역변환 수행
inversed1 = seq_scaler.inverse_transform(dummy1)[:, -1]  # 마지막 열만 추출 (전기요금)

metrics["lstm"] = r2_score(y_seq_val, val_pred)

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

dummy = np.zeros((len(lstm_test_pred), len(FEATURES) + 1))  # +1은 TARGET(전기요금)
dummy[:, -1] = lstm_test_pred  # 전기요금만 채우고 나머지는 0

# 2. 역변환 수행
inversed = seq_scaler.inverse_transform(dummy)[:, -1]  # 마지막 열만 추출 (전기요금)
# 1. 원본 예측 (이미 존재)
inversed_raw = inversed

test_features["lstm_pred"] = inversed
residual_correction = residual_model.predict(test_features)
corrected_lstm = inversed + residual_correction

# 2. Smooth만 적용
inversed_smooth = uniform_filter1d(inversed_raw, size=5)

# 3. Clip만 적용 (실제 lower_bound는 앞서 추천한 300으로 사용)
lower_bound = 600
upper_bound = 13651.88
inversed_clipped = np.clip(inversed_raw, lower_bound, upper_bound)

# 4. Smooth + Clip 적용
inversed_smooth_clipped = np.clip(inversed_smooth, lower_bound, upper_bound)
# true = answer["전기요금(원)"].values
# MAE 계산
# mae_smooth = mean_absolute_error(true, inversed_smooth)
# mae_clipped = mean_absolute_error(true, inversed_clipped)
# mae_smooth_clipped = mean_absolute_error(true, inversed_smooth_clipped)

# mae_lstm = mean_absolute_error(true, inversed)
# print(f"lstm_mae : {mae_lstm}")

# mae_lstm = mean_absolute_error(true, corrected_lstm)
# print(f"lstm_mae : {mae_lstm}")

preds_test["lstm"] = inversed_clipped
preds_val["lstm"] = inversed1

# ----------------------------------------------------------------------
# 11. Weighted ensemble
# ----------------------------------------------------------------------
weights = {name: max(score, 0) for name, score in metrics.items()}
print("📊 각 모델의 설명력 (R² score):")
for name, score in sorted(metrics.items(), key=lambda x: -x[1]):
    print(f"- {name}: R² = {score:.4f}")
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

# Test prediction
test_pred = np.zeros(len(test_df))
for name, pred in preds_test.items():
    # mae = mean_absolute_error(true, pred)
    print(f"{name}:{mae}")
    test_pred += weights[name] * pred


submission = pd.DataFrame({
    "id": test_df["id"].values,
    "전기요금(원)": test_pred
})

# submission1 = pd.DataFrame({
#     "id": test_df["id"].values,
#     "측정일시" : test_df["측정일시"],
#     "전기요금(원)": test_pred,
#     "실제요금":answer["전기요금(원)"]
# })
# for name, pred in preds_test.items():
#     submission1[f"{name}"] = pred
# mae_asem = mean_absolute_error(answer["전기요금(원)"], submission["전기요금(원)"])
# print(f"asem_mae : {mae_asem}")
# submission1.to_csv("submission_optimal.csv", index=False)
submission.to_csv("submission3.csv", index=False)
print("Saved submission_optimal.csv and submission.csv")

# ----------------------------------------------------------------------
# 12. Save trained models
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

