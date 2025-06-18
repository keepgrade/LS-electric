# 📦 1. 라이브러리 임포트
import os
import pickle
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 📂 2. 데이터 불러오기
BASE_DIR = "./data"
train_df = pd.read_csv(os.path.join(BASE_DIR, "train_with_weather.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test_with_weather.csv"))

# 📅 3. 시간 관련 파생변수 생성
for df in [train_df, test_df]:
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["시간"] = df["측정일시"].dt.hour
    df["요일"] = df["측정일시"].dt.weekday
    df["휴일여부"] = df["요일"].isin([0, 6]).astype(int)
    df["sin_시간"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_시간"] = np.cos(2 * np.pi * df["시간"] / 24)

# 💰 4. 시간대 기반 요금단가 생성
def get_season(month):
    if month in [6, 7, 8]: return "여름"
    elif month in [3, 4, 5, 9, 10]: return "봄가을"
    else: return "겨울"

def get_time_zone(hour, season):
    if season in ["여름", "봄가을"]:
        if 22 <= hour or hour < 8: return "경부하"
        if (8 <= hour < 11) or (12 <= hour < 13) or (18 <= hour < 22): return "중간부하"
        return "최대부하"
    else:
        if 22 <= hour or hour < 8: return "경부하"
        if (8 <= hour < 9) or (12 <= hour < 16) or (19 <= hour < 22): return "중간부하"
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

# 🔡 5. 범주형 변수 인코딩 및 타겟 인코딩
def target_encoding(df_train, df_test, col, target, smoothing=10):
    global_mean = df_train[target].mean()
    agg = df_train.groupby(col)[target].agg(["mean", "count"])
    weight = 1 / (1 + np.exp(-(agg["count"] - smoothing)))
    enc = global_mean * (1 - weight) + agg["mean"] * weight
    mapping = enc.to_dict()
    df_train[f"{col}_te"] = df_train[col].map(mapping)
    df_test[f"{col}_te"] = df_test[col].map(mapping)

le = LabelEncoder()
train_df["작업유형_encoded"] = le.fit_transform(train_df["작업유형"])
test_df["작업유형_encoded"] = le.transform(test_df["작업유형"])

for col in ["작업유형", "시간", "요일", "시간대"]:
    target_encoding(train_df, test_df, col, "전기요금(원)")

# 📊 6. 이상치 제거
q1 = train_df["전기요금(원)"].quantile(0.25)
q3 = train_df["전기요금(원)"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
train_df = train_df[(train_df["전기요금(원)"] >= lower) & (train_df["전기요금(원)"] <= upper)]

# 🧾 7. Feature/Target 정의 및 정규화
FEATURES = [
    "작업유형_encoded", "월", "일", "시간", "요일", "휴일여부",
    "sin_시간", "cos_시간", "요금단가",
    "작업유형_te", "시간_te", "요일_te", "시간대_te"
]
TARGETS = ["전력사용량(kWh)", "탄소배출량(tCO2)", "전기요금(원)"]

X = train_df[FEATURES]
y = train_df[TARGETS]
X_test = test_df[FEATURES]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 🌲 8. 트리 모델 학습 (3개 타겟별 개별 예측)
models = {}
metrics = {}
preds_test = {}
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

for target in TARGETS:
    for name, model in {
        "xgb": XGBRegressor(n_estimators=200, max_depth=4),
        "lgb": LGBMRegressor(n_estimators=200, max_depth=4),
        "rf": RandomForestRegressor(n_estimators=200, max_depth=6)
    }.items():
        model.fit(X_train, y_train[target])
        pred = model.predict(X_val)
        score = r2_score(y_val[target], pred)
        models[f"{name}_{target}"] = model
        metrics[f"{name}_{target}"] = round(score, 4)
        preds_test[f"{name}_{target}"] = model.predict(X_test_scaled)

# 📈 9. LSTM을 위한 시계열 데이터 구성 및 정규화
TIME_STEPS = 96 * 7
seq_scaler = MinMaxScaler()
seq_data = pd.concat([train_df[FEATURES], train_df[TARGETS]], axis=1)
seq_scaled = seq_scaler.fit_transform(seq_data)
scaled_df = pd.DataFrame(seq_scaled, columns=FEATURES + TARGETS)

# 🔄 10. 시계열 시퀀스 생성 함수
def create_sequences(data, timesteps):
    xs, ys = [], []
    for i in range(len(data) - timesteps):
        xs.append(data.iloc[i:i+timesteps][FEATURES].values)
        ys.append(data.iloc[i+timesteps][TARGETS].values)
    return np.array(xs), np.array(ys)

X_seq, y_seq = create_sequences(scaled_df, TIME_STEPS)
split = int(0.8 * len(X_seq))
X_seq_train, X_seq_val = X_seq[:split], X_seq[split:]
y_seq_train, y_seq_val = y_seq[:split], y_seq[split:]

# 🧠 11. Multi-output LSTM 모델 정의 및 학습
model = Sequential([
    LSTM(128, input_shape=(TIME_STEPS, len(FEATURES))),
    Dense(64, activation='relu'),
    Dense(len(TARGETS))
])
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_seq_train, y_seq_train, validation_data=(X_seq_val, y_seq_val),
          epochs=20, batch_size=32, callbacks=[es], verbose=1)

# 💾 12. 모델 및 스케일러 저장
os.makedirs("./pickles", exist_ok=True)
with open("./pickles/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("./pickles/seq_scaler.pkl", "wb") as f:
    pickle.dump(seq_scaler, f)
model.save("./pickles/lstm_multi_output.h5")

print("✅ 학습 완료 및 모델 저장 완료")













# ✅ 슬라이딩 방식으로 12월 요금 예측 수행 (full_scaled 저장 없이)
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# 📂 1. 데이터 로드
BASE_DIR = "./data"
train_df = pd.read_csv(os.path.join(BASE_DIR, "train_with_weather.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test_with_weather.csv"))

# 📅 2. 시간 파생 변수 생성
for df in [train_df, test_df]:
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["시간"] = df["측정일시"].dt.hour
    df["요일"] = df["측정일시"].dt.weekday
    df["주말여부"] = (df["요일"] >= 5).astype(int)
    df["sin_시간"] = np.sin(2 * np.pi * df["시간"] / 24)
    df["cos_시간"] = np.cos(2 * np.pi * df["시간"] / 24)

# 💰 3. 요금단가 계산 함수
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

def get_season(month):
    return "여름" if month in [6, 7, 8] else "봄가을" if month in [3, 4, 5, 9, 10] else "겨울"
def get_time_zone(hour, season):
    if season in ["여름", "봄가을"]:
        if 22 <= hour or hour < 8: return "경부하"
        if (8 <= hour < 11) or (12 <= hour < 13) or (18 <= hour < 22): return "중간부하"
        return "최대부하"
    else:
        if 22 <= hour or hour < 8: return "경부하"
        if (8 <= hour < 9) or (12 <= hour < 16) or (19 <= hour < 22): return "중간부하"
        return "최대부하"

for df in [train_df, test_df]:
    df["계절"] = df["월"].apply(get_season)
    df["적용시점"] = df["측정일시"].apply(lambda x: "before" if x < CUTOFF else "after")
    df["시간대"] = df.apply(lambda r: get_time_zone(r["시간"], r["계절"]), axis=1)
    df["요금단가"] = df.apply(lambda r: RATE_TABLE[r["적용시점"]][r["계절"]][r["시간대"]], axis=1)

# 🔡 4. 인코딩
le = LabelEncoder()
train_df["작업유형_encoded"] = le.fit_transform(train_df["작업유형"])
test_df["작업유형_encoded"] = le.transform(test_df["작업유형"])

def target_encoding(df_train, df_test, col, target):
    global_mean = df_train[target].mean()
    agg = df_train.groupby(col)[target].agg(["mean", "count"])
    weight = 1 / (1 + np.exp(-(agg["count"] - 10)))
    enc = global_mean * (1 - weight) + agg["mean"] * weight
    mapping = enc.to_dict()
    df_train[f"{col}_te"] = df_train[col].map(mapping)
    df_test[f"{col}_te"] = df_test[col].map(mapping)

for col in ["작업유형", "시간", "요일", "시간대"]:
    target_encoding(train_df, test_df, col, "전기요금(원)")

# 📐 5. 입력 변수와 시퀀스 정의
FEATURES = [
    "작업유형_encoded", "월", "일", "시간", "요일", "주말여부",
    "sin_시간", "cos_시간", "요금단가",
    "작업유형_te", "시간_te", "요일_te", "시간대_te"
]
TARGETS = ["전력사용량(kWh)", "탄소배출량(tCO2)", "전기요금(원)"]
TIME_STEPS = 96 * 7

# 🔄 6. 스케일러 로드 및 정규화
with open("./pickles/seq_scaler.pkl", "rb") as f:
    seq_scaler = pickle.load(f)

full_input = pd.concat([
    pd.concat([train_df[FEATURES], train_df[TARGETS]], axis=1),
    pd.concat([test_df[FEATURES], pd.DataFrame(0, index=range(len(test_df)), columns=TARGETS)], axis=1)
], ignore_index=True)

full_scaled = seq_scaler.transform(full_input)
scaled_df = pd.DataFrame(full_scaled, columns=FEATURES + TARGETS)

# 🧠 7. LSTM 로드 및 슬라이딩 예측
model = load_model("./pickles/lstm_multi_output.h5", compile=False)
pred_list = []

last_sequence = scaled_df.iloc[-(TIME_STEPS + len(test_df)):-len(test_df)][FEATURES].values.copy()

for i in range(len(test_df)):
    input_seq = last_sequence[-TIME_STEPS:]
    pred_scaled = model.predict(input_seq[np.newaxis, :, :], verbose=0)

    dummy_input = np.zeros((1, len(FEATURES)))
    full_row = np.concatenate([dummy_input, pred_scaled], axis=1)
    inverse_row = seq_scaler.inverse_transform(full_row)[0][-len(TARGETS):]

    pred_list.append(inverse_row)

    # 슬라이딩 적용: 다음 입력 업데이트
    next_input = scaled_df.iloc[-len(test_df) + i][FEATURES].values  # 날씨 등은 사용
    last_sequence = np.vstack([last_sequence[1:], next_input])

# 📤 8. 결과 저장
pred_df = pd.DataFrame(pred_list, columns=TARGETS)
submission = pd.DataFrame({
    "id": test_df["id"],
    "전기요금(원)": pred_df["전기요금(원)"]
})
submission.to_csv("submission_lstm.csv", index=False)
print("✅ 슬라이딩 예측 완료 → submission_lstm.csv 저장됨")


# -----------------------------------------------------------------
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 📂 데이터 로드 및 시계열 scaler 로딩
test_df = pd.read_csv("./data/test_with_weather.csv")
with open("./pickles/seq_scaler.pkl", "rb") as f:
    seq_scaler = pickle.load(f)

# 📄 피처 설정
FEATURES = [
    "작업유형_encoded", "월", "일", "시간", "요일", "주말여부",
    "sin_시간", "cos_시간", "요금단가",
    "작업유형_te", "시간_te", "요일_te", "시간대_te"
]
TARGETS = ["전력사용량(kWh)", "탄소배출량(tCO2)", "전기요금(원)"]
TIME_STEPS = 96 * 7

# 전체 시퀀스 데이터 재구성
full_input = pd.read_csv("./data/full_scaled.csv")  # 미리 scaling된 전체 시퀀스를 저장한 파일
full_scaled = full_input.values
scaled_df = pd.DataFrame(full_scaled, columns=FEATURES + TARGETS)

# 슬라이딩 예측 시작점
X_input = scaled_df.iloc[-TIME_STEPS:][FEATURES].values
X_pred = []

model = load_model("/mnt/data/lstm_multi_output.h5", compile=False)

for _ in range(len(test_df)):
    x = np.expand_dims(X_input, axis=0)  # (1, TIME_STEPS, features)
    pred_scaled = model.predict(x, verbose=0)[0]

    # 예측값을 역정규화 없이 다시 붙이기 위해 스케일된 상태로 유지
    next_row = np.concatenate([X_input[-1], pred_scaled])  # (features + targets,)
    full_scaled_next = np.append(X_input[1:], [next_row[:len(FEATURES)]], axis=0)
    X_input = full_scaled_next

    X_pred.append(pred_scaled)

# 결과 역정규화
X_pred = np.array(X_pred)
dummy = np.zeros((len(X_pred), len(FEATURES)))
recon = np.concatenate([dummy, X_pred], axis=1)
inv_preds = seq_scaler.inverse_transform(recon)[:, -len(TARGETS):]
pred_df = pd.DataFrame(inv_preds, columns=TARGETS)
pred_df["id"] = test_df["id"]

# 전기요금 예측만 저장
submission = pred_df[["id", "전기요금(원)"]]
submission.to_csv("./data/submission_lstm_sliding.csv", index=False)




