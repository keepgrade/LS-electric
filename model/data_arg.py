import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 로딩
train_df = pd.read_csv("../data/train.csv")

# 2. 파생 변수 생성 (시간 관련)
train_df["측정일시"] = pd.to_datetime(train_df["측정일시"])
train_df["월"] = train_df["측정일시"].dt.month
train_df["일"] = train_df["측정일시"].dt.day
train_df["시간"] = train_df["측정일시"].dt.hour
train_df["요일"] = train_df["측정일시"].dt.weekday
train_df["sin_시간"] = np.sin(2 * np.pi * train_df["시간"] / 24)
train_df["cos_시간"] = np.cos(2 * np.pi * train_df["시간"] / 24)

# 3. 사용할 피처 및 타겟 정의
FEATURES = ["월", "일", "시간", "요일", "sin_시간", "cos_시간"]
TARGET = "전기요금(원)"

# 4. 스케일링
scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_df[FEATURES + [TARGET]]), columns=FEATURES + [TARGET])

# 5. 시계열 증강 함수
def create_augmented_sequences(data, features, target, time_steps=672, n_slices=2, noise_level=0.01):
    xs, ys = [], []
    for i in range(len(data) - time_steps - 1):
        base_seq = data.iloc[i:i+time_steps].copy()
        base_target = data.iloc[i+time_steps][target]
        xs.append(base_seq[features].values)
        ys.append(base_target)
        for _ in range(n_slices):
            sliced = base_seq.sample(frac=1).sort_index()
            noisy = sliced[features].values + np.random.normal(0, noise_level, sliced[features].shape)
            xs.append(noisy)
            ys.append(base_target)
    return np.array(xs), np.array(ys)

# 6. 증강 시퀀스 생성
X_seq, y_seq = create_augmented_sequences(train_scaled, FEATURES, TARGET, time_steps=96*7, n_slices=2)

# 7. 훈련/검증 분리
split_idx = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

# 8. LSTM 모델 정의 및 학습
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(32, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[es])

# 9. 검증 성능
val_pred = model.predict(X_val).flatten()
print("📊 LSTM with Augmentation R²:", r2_score(y_val, val_pred))






# 10. test.csv 불러오기 및 피처 생성
test_df = pd.read_csv("../data/test.csv")
test_df["측정일시"] = pd.to_datetime(test_df["측정일시"])
test_df["월"] = test_df["측정일시"].dt.month
test_df["일"] = test_df["측정일시"].dt.day
test_df["시간"] = test_df["측정일시"].dt.hour
test_df["요일"] = test_df["측정일시"].dt.weekday
test_df["sin_시간"] = np.sin(2 * np.pi * test_df["시간"] / 24)
test_df["cos_시간"] = np.cos(2 * np.pi * test_df["시간"] / 24)


FEATURES = ["월", "일", "시간", "요일", "sin_시간", "cos_시간"]


# 학습 시점에 전기요금(원)은 제외
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(train_df[FEATURES])  # TARGET은 제외
joblib.dump(scaler, "models/minmax_scaler.pkl")

# 테스트 데이터도 같은 FEATURES로만 transform
test_scaled = pd.DataFrame(scaler.transform(test_df[FEATURES]), columns=FEATURES)

# 스케일링
test_scaled = pd.DataFrame(scaler.transform(test_df[FEATURES]), columns=FEATURES)



# LSTM 입력 시퀀스 구성
def create_lstm_sequences(test_features_df, last_known, time_steps=96*7):
    combined = pd.concat([last_known, test_features_df], ignore_index=True)
    seqs = []
    for i in range(len(test_features_df)):
        start_idx = i
        end_idx = i + time_steps
        seq = combined.iloc[start_idx:end_idx][FEATURES].values
        seqs.append(seq)
    return np.array(seqs[-len(test_features_df):])  # 가장 마지막 시점 기준

# 최근 학습 데이터에서 마지막 시퀀스 가져오기
last_known_scaled = train_scaled[FEATURES].iloc[-(96*7):]

# 시퀀스 생성
X_test_seq = create_lstm_sequences(test_scaled, last_known_scaled)

# 예측
test_pred = model.predict(X_test_seq).flatten()

# submission 저장
submission = pd.DataFrame({
    "id": test_df["id"],
    "전기요금(원)": test_pred
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 저장 완료")

# 모델 및 스케일러 저장
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/minmax_scaler.pkl")
print("✅ 모델 및 스케일러 저장 완료")
