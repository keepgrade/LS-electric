import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# ✅ 데이터 로딩 (로컬 경로)
data_path = "../data/train.csv"  # 또는 "C:/Users/username/Desktop/train_with_weather.csv"
df = pd.read_csv(data_path)
df.set_index('측정일시', inplace=True)
df.index = pd.to_datetime(df.index)
df.sort_values(['id'], inplace=True)

# ✅ id=29855에서 0인 열 보간
row_idx = df[df['id'] == 29855].index[0]
zero_columns = df.loc[row_idx][df.loc[row_idx] == 0].index.tolist()
df_interp = df.copy()
df_interp[zero_columns] = df_interp[zero_columns].replace(0, pd.NA)
df_interp[zero_columns] = df_interp[zero_columns].interpolate(method='time')

# ✅ 예측할 변수
target_cols = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
               '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '기온', '습도']

# ✅ 정규화
scalers = {}
scaled_df = pd.DataFrame(index=df_interp.index)
for col in target_cols:
    scaler = MinMaxScaler()
    scaled_df[col] = scaler.fit_transform(df_interp[[col]])
    scalers[col] = scaler

# ✅ 시퀀스 설정
seq_len = 96 * 7  # 7일치
prediction_df_list = []

# ✅ LSTM 예측 (12월) - 예측값만 저장
for target in target_cols:
    X, y, timestamps = [], [], []
    for i in range(seq_len, len(scaled_df) - 1):
        input_seq = scaled_df.iloc[i - seq_len:i].values
        target_value = scaled_df[target].iloc[i + 1]
        timestamp = scaled_df.index[i + 1]
        X.append(input_seq)
        y.append(target_value)
        timestamps.append(timestamp)

    X = np.array(X)
    y = np.array(y)
    timestamps = np.array(timestamps)

    train_idx = [i for i, t in enumerate(timestamps) if t < pd.Timestamp("2024-12-01")]
    test_idx = [i for i, t in enumerate(timestamps) if pd.Timestamp("2024-12-01") <= t < pd.Timestamp("2025-01-01")]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, test_timestamps = X[test_idx], timestamps[test_idx]

    if len(X_train) < 10 or len(X_test) == 0:
        print(f"❌ {target}: 학습/테스트 데이터 부족. 건너뜀")
        continue

    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=50 if len(X_train) >= 100 else 30,
        batch_size=32 if len(X_train) >= 100 else 8,
        validation_split=0.1 if len(X_train) >= 100 else 0.0,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred_scaled = model.predict(X_test).flatten()
    scaler = scalers[target]
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    prediction_df = pd.DataFrame({
        '측정일시': test_timestamps,
        f"{target}_예측": y_pred
    })
    prediction_df_list.append(prediction_df)

# ✅ 통합 예측값 저장 (로컬)
result = prediction_df_list[0]
for df_part in prediction_df_list[1:]:
    result = pd.merge(result, df_part, on='측정일시')

output_path = "./LSTM_12월예측_예측값만.csv"
result.to_csv(output_path, index=False)
print(f"✅ 예측값 저장 완료 → {output_path}")
print(result.head())
