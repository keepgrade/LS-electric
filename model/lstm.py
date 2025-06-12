# ================================
# 📦 1. 모듈 임포트
# ================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


# ================================
# 📂 2. 데이터 불러오기 및 정렬
# ================================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])
test_df['측정일시'] = pd.to_datetime(test_df['측정일시'])
train_df = train_df.sort_values('측정일시')
test_df = test_df.sort_values('측정일시')


# ================================
# 📊 3. 변수 정의
# ================================
features = [
    '전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
    '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)'
]
target_col = '전기요금(원)'
TIME_STEPS = 10



# ================================
# 🪜 4. 시계열 데이터셋 생성 함수
# ================================
def create_sequences(data, target_col, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data.iloc[i:i+time_steps][features].values)
        y.append(data.iloc[i+time_steps][target_col])
    return np.array(X), np.array(y)

# ================================
# 🔄 5. 정규화 및 시계열 데이터 준비
# ================================
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(train_df[features])
y_scaled = y_scaler.fit_transform(train_df[[target_col]])

scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df[target_col] = y_scaled

X, y = create_sequences(scaled_df, target_col, time_steps=TIME_STEPS)

# ================================
# ✂️ 6. Train/Validation 분리
# ================================
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ================================
# 🤖 7. LSTM 모델 정의 및 학습
# ================================
model = Sequential()
model.add(LSTM(64, input_shape=(TIME_STEPS, len(features))))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ================================
# 📈 8. 성능 평가 (역변환 포함)
# ================================
y_pred = model.predict(X_val).flatten()
y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_val_inv = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_val_inv, y_pred_inv)
mse = mean_squared_error(y_val_inv, y_pred_inv)
rmse = np.sqrt(mse)
r2 = r2_score(y_val_inv, y_pred_inv)

print(f"✅ MAE: {mae:.2f}")
print(f"📉 MSE: {mse:.2f}")
print(f"📏 RMSE: {rmse:.2f}")
print(f"📊 R² Score: {r2:.4f}")


# ================================
# 🧪 9. 테스트 데이터 예측 준비
# ================================
for col in features:
    test_df[col] = train_df[col].mean()

combined_df = pd.concat([train_df[-TIME_STEPS:][features], test_df[features]], ignore_index=True)
combined_scaled = X_scaler.transform(combined_df)
combined_scaled = pd.DataFrame(combined_scaled, columns=features)

X_seq = [combined_scaled.iloc[i:i+TIME_STEPS].values for i in range(len(combined_scaled) - TIME_STEPS)]
X_seq = np.array(X_seq)


# ================================
# 🤖 10. 테스트 예측 및 저장
# ================================
preds = model.predict(X_seq).flatten()
preds_inverse = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

test_df['전기요금(원)'] = preds_inverse[:len(test_df)]
submission = test_df[['id', '전기요금(원)']]
submission.to_csv("submission.csv", index=False)
print("📁 submission.csv 파일 저장 완료")