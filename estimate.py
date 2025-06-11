# ================================
# 📦 1. 모듈 임포트
# ================================
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ================================
# 📂 2. 데이터 불러오기
# ================================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# ================================
# 🕒 3. 날짜 및 시간 파생 변수
# ================================
for df in [train_df, test_df]:
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['월'] = df['측정일시'].dt.month
    df['일'] = df['측정일시'].dt.day
    df['시간'] = df['측정일시'].dt.hour
    df['요일'] = df['측정일시'].dt.weekday
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)
    df['sin_시간'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_시간'] = np.cos(2 * np.pi * df['시간'] / 24)

# ================================
# 🌦️ 4. 계절 및 시간대 파생
# ================================
def get_season(month):
    if month in [6, 7, 8]:
        return '여름'
    elif month in [3, 4, 5, 9, 10]:
        return '봄가을'
    else:
        return '겨울'

def get_time_zone(hour, month):
    season = get_season(month)
    if season in ['여름', '봄가을']:
        if 22 <= hour or hour < 8:
            return '경부하'
        elif (8 <= hour < 11) or (12 <= hour < 13) or (18 <= hour < 22):
            return '중간부하'
        else:
            return '최대부하'
    else:  # 겨울
        if 22 <= hour or hour < 8:
            return '경부하'
        elif (8 <= hour < 9) or (12 <= hour < 16) or (19 <= hour < 22):
            return '중간부하'
        else:
            return '최대부하'

train_df['계절'] = train_df['월'].apply(get_season)
test_df['계절'] = test_df['월'].apply(get_season)

train_df['시간대'] = train_df.apply(lambda row: get_time_zone(row['시간'], row['월']), axis=1)
test_df['시간대'] = test_df.apply(lambda row: get_time_zone(row['시간'], row['월']), axis=1)

# ================================
# 💰 5. 요금단가 계산
# ================================
rate_table = {
    'before': {
        '여름':    {'경부하': 93.1, '중간부하': 146.3, '최대부하': 216.6},
        '봄가을': {'경부하': 93.1, '중간부하': 115.2, '최대부하': 138.9},
        '겨울':   {'경부하': 100.4, '중간부하': 146.5, '최대부하': 193.4}
    },
    'after': {
        '여름':    {'경부하': 110.0, '중간부하': 163.2, '최대부하': 233.5},
        '봄가을': {'경부하': 110.0, '중간부하': 132.1, '최대부하': 155.8},
        '겨울':   {'경부하': 117.3, '중간부하': 163.4, '최대부하': 210.3}
    }
}

cutoff_date = datetime(2024, 10, 24)

for df in [train_df, test_df]:
    df['적용시점'] = df['측정일시'].apply(lambda x: 'before' if x < cutoff_date else 'after')
    df['요금단가'] = df.apply(lambda row: rate_table[row['적용시점']][row['계절']][row['시간대']], axis=1)

# ================================
# 🔤 6. 작업유형 인코딩
# ================================
le = LabelEncoder()
train_df['작업유형_encoded'] = le.fit_transform(train_df['작업유형'])
test_df['작업유형_encoded'] = le.transform(test_df['작업유형'])

# 🎯 타겟 인코딩
type_mean = train_df.groupby('작업유형')['전기요금(원)'].mean().to_dict()
train_df['작업유형_target'] = train_df['작업유형'].map(type_mean)
test_df['작업유형_target'] = test_df['작업유형'].map(type_mean)

# ================================
# 🧼 7. 모델 입력 정의
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

# ================================
# ✂️ 8. 데이터 분할
# ================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 🤖 9. 모델 학습
# ================================
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)  # ← 간단하게!

# ================================
# 📈 10. 성능 평가
# ================================
val_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, val_pred)
mse = mean_squared_error(y_val, val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_pred)

print(f"✅ MAE (평균절대오차): {mae:.2f}")
print(f"📉 MSE (평균제곱오차): {mse:.2f}")
print(f"📏 RMSE (제곱근 평균제곱오차): {rmse:.2f}")
print(f"📊 R² Score (결정계수): {r2:.4f}")

# ================================
# 📁 11. 예측 및 제출파일 저장
# ================================
test_df['전기요금(원)'] = model.predict(X_test)
submission = test_df[['id', '전기요금(원)']]
submission.to_csv("submission.csv", index=False)
print("📁 submission.csv 파일 저장 완료")



print(test_df)