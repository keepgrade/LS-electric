# 📦 패키지 임포트
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from datetime import datetime
import os

# 📂 경로 설정
BASE_DIR = "./data"  # 또는 절대경로로 지정
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")

# ================================
# 📊 1. 데이터 불러오기
# ================================
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# ================================
# 🕒 2. datetime 파싱 + 시계열 파생
# ================================
for df in [train_df, test_df]:
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['월'] = df['측정일시'].dt.month
    df['일'] = df['측정일시'].dt.day
    df['시간'] = df['측정일시'].dt.hour
    df['요일'] = df['측정일시'].dt.weekday
    df['주말여부'] = (df['요일'] >= 5).astype(int)
    df['sin_시간'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_시간'] = np.cos(2 * np.pi * df['시간'] / 24)

# ================================
# 🌞 3. 계절/시간대 + 요금단가 반영
# ================================
def get_season(month):
    if month in [6, 7, 8]:
        return '여름'
    elif month in [3, 4, 5, 9, 10]:
        return '봄가을'
    else:
        return '겨울'

def get_time_zone(hour, season):
    if season in ['여름', '봄가을']:
        if 22 <= hour or hour < 8:
            return '경부하'
        elif (8 <= hour < 11) or (12 <= hour < 13) or (18 <= hour < 22):
            return '중간부하'
        else:
            return '최대부하'
    else:
        if 22 <= hour or hour < 8:
            return '경부하'
        elif (8 <= hour < 9) or (12 <= hour < 16) or (19 <= hour < 22):
            return '중간부하'
        else:
            return '최대부하'

rate_table = {
    'before': {
        '여름': {'경부하': 93.1, '중간부하': 146.3, '최대부하': 216.6},
        '봄가을': {'경부하': 93.1, '중간부하': 115.2, '최대부하': 138.9},
        '겨울': {'경부하': 100.4, '중간부하': 146.5, '최대부하': 193.4}
    },
    'after': {
        '여름': {'경부하': 110.0, '중간부하': 163.2, '최대부하': 233.5},
        '봄가을': {'경부하': 110.0, '중간부하': 132.1, '최대부하': 155.8},
        '겨울': {'경부하': 117.3, '중간부하': 163.4, '최대부하': 210.3}
    }
}

cutoff_date = datetime(2024, 10, 24)

for df in [train_df, test_df]:
    df['계절'] = df['월'].apply(get_season)
    df['적용시점'] = df['측정일시'].apply(lambda x: 'before' if x < cutoff_date else 'after')
    df['시간대'] = df.apply(lambda row: get_time_zone(row['시간'], row['계절']), axis=1)
    df['요금단가'] = df.apply(lambda row: rate_table[row['적용시점']][row['계절']][row['시간대']], axis=1)

# ================================
# 🔤 4. 작업유형 인코딩
# ================================
le = LabelEncoder()
train_df['작업유형_encoded'] = le.fit_transform(train_df['작업유형'])
test_df['작업유형_encoded'] = le.transform(test_df['작업유형'])

# 🎯 타겟 인코딩
type_mean = train_df.groupby('작업유형')['전기요금(원)'].mean().to_dict()
train_df['작업유형_target'] = train_df['작업유형'].map(type_mean)
test_df['작업유형_target'] = test_df['작업유형'].map(type_mean)

# ================================
# 🎯 5. 모델 학습
# ================================
features = [
    '작업유형_encoded', '작업유형_target',
    '월', '일', '요일', '주말여부',
    'sin_시간', 'cos_시간',
    '요금단가'
]
target = '전기요금(원)'

X = train_df[features]
y = train_df[target]
X_test = test_df[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# ================================
# 📊 6. 평가 및 예측 저장
# ================================
val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, val_pred)
mse = mean_squared_error(y_val, val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_pred)

print("✅ 평가 지표")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# 📁 결과 저장
test_df['전기요금(원)'] = model.predict(X_test)
test_df[['id', '전기요금(원)']].to_csv("submission.csv", index=False)
