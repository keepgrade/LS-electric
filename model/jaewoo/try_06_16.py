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

# 💰 3. 요금단가 계산
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

# 🔡 4. 인코딩 및 타겟 인코딩
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

# 예측을 위한 입력 저장
FEATURES = [
    "작업유형_encoded", "월", "일", "시간", "요일", "주말여부",
    "sin_시간", "cos_시간", "요금단가",
    "작업유형_te", "시간_te", "요일_te", "시간대_te"
]
TARGETS = ["전력사용량(kWh)", "탄소배출량(tCO2)", "전기요금(원)"]

# 정규화 및 LSTM 로드 후 예측
with open("./pickles/seq_scaler.pkl", "rb") as f:
    seq_scaler = pickle.load(f)

# concat full data
full_input = pd.concat([
    pd.concat([train_df[FEATURES], train_df[TARGETS]], axis=1),
    pd.concat([test_df[FEATURES], pd.DataFrame(0, index=range(len(test_df)), columns=TARGETS)], axis=1)
], ignore_index=True)

full_scaled = seq_scaler.transform(full_input)
scaled_df = pd.DataFrame(full_scaled, columns=FEATURES + TARGETS)

# 슬라이딩 예측
TIME_STEPS = 96 * 7
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

    next_input = scaled_df.iloc[-len(test_df) + i][FEATURES].values
    last_sequence = np.vstack([last_sequence[1:], next_input])

# 결과 DataFrame 생성
pred_df = pd.DataFrame(pred_list, columns=TARGETS)
test_augmented = test_df.copy()
test_augmented["전력사용량_LSTM"] = pred_df["전력사용량(kWh)"]
test_augmented["탄소배출량_LSTM"] = pred_df["탄소배출량(tCO2)"]
