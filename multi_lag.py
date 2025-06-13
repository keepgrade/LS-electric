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
# 📂 2. 데이터 로드
# ================================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# ================================
# 🧼 3. 공통 전처리 함수 정의
# ================================
def preprocess_datetime(df):
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['월'] = df['측정일시'].dt.month
    df['일'] = df['측정일시'].dt.day
    df['시간'] = df['측정일시'].dt.hour
    df['요일'] = df['측정일시'].dt.weekday
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)
    df['sin_시간'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_시간'] = np.cos(2 * np.pi * df['시간'] / 24)
    return df

train_df = preprocess_datetime(train_df)
test_df = preprocess_datetime(test_df)

# ================================
# 🌦️ 4. 시간대/계절별 요금 단가 계산
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

def apply_tariff(df):
    df['계절'] = df['월'].apply(get_season)
    df['시간대'] = df.apply(lambda row: get_time_zone(row['시간'], row['월']), axis=1)
    df['적용시점'] = df['측정일시'].apply(lambda x: 'before' if x < cutoff_date else 'after')
    df['요금단가'] = df.apply(lambda row: rate_table[row['적용시점']][row['계절']][row['시간대']], axis=1)
    return df

train_df = apply_tariff(train_df)
test_df = apply_tariff(test_df)

# ================================
# 🔤 5. 작업유형 인코딩 및 타겟 인코딩
# ================================
le = LabelEncoder()
train_df['작업유형_encoded'] = le.fit_transform(train_df['작업유형'])
test_df['작업유형_encoded'] = le.transform(test_df['작업유형'])

type_mean = train_df.groupby('작업유형')['전기요금(원)'].mean().to_dict()
train_df['작업유형_target'] = train_df['작업유형'].map(type_mean)
test_df['작업유형_target'] = test_df['작업유형'].map(type_mean)

# ================================
# 🔁 6. 다양한 피처 LAG 및 ROLLING 피처 생성
# ================================

# LAG 적용할 피처들 정의
lag_features = ['전기요금(원)', '요금단가', '작업유형_target', '시간', '요일', '주말여부']
lag_periods = [1, 3, 6, 12, 24]  # 1시간, 3시간, 6시간, 12시간, 24시간(1일) 전

# 🎯 TRAIN 데이터에 LAG 피처 적용
print("🔄 Train 데이터에 LAG 피처 생성 중...")
for feature in lag_features:
    for lag in lag_periods:
        train_df[f'{feature}_lag_{lag}'] = train_df[feature].shift(lag)

# 🎯 ROLLING 피처 생성 (Train만)
rolling_features = ['전기요금(원)', '요금단가', '작업유형_target']
rolling_windows = [3, 6, 12, 24]  # 3시간, 6시간, 12시간, 24시간 윈도우

print("🎯 Train 데이터에 ROLLING 피처 생성 중...")
for feature in rolling_features:
    for window in rolling_windows:
        train_df[f'{feature}_rolling_mean_{window}'] = train_df[feature].rolling(window=window).mean()
        train_df[f'{feature}_rolling_std_{window}'] = train_df[feature].rolling(window=window).std()
        train_df[f'{feature}_rolling_max_{window}'] = train_df[feature].rolling(window=window).max()
        train_df[f'{feature}_rolling_min_{window}'] = train_df[feature].rolling(window=window).min()

# 🎯 변화율 피처 생성 (Train만)
print("📈 Train 데이터에 변화율 피처 생성 중...")
for feature in ['전기요금(원)', '요금단가']:
    train_df[f'{feature}_diff_1'] = train_df[feature].diff(1)
    train_df[f'{feature}_diff_3'] = train_df[feature].diff(3)
    train_df[f'{feature}_pct_change_1'] = train_df[feature].pct_change(1)
    train_df[f'{feature}_pct_change_3'] = train_df[feature].pct_change(3)

# 결측치 제거 (LAG/ROLLING 피처로 인한 결측치)
print("🧹 결측치 제거 중...")
train_df_clean = train_df.dropna().copy()
print(f"✅ Train 데이터: {len(train_df)} → {len(train_df_clean)} (결측치 제거 후)")

# ================================
# 🧪 7. TEST 데이터에 LAG/ROLLING 피처 적용
# ================================
print("🔄 Test 데이터에 LAG/ROLLING 피처 생성 중...")

# 마지막 N개 값을 사용하여 테스트 데이터의 LAG 피처 생성
def create_test_lag_features(train_data, test_data, lag_features, lag_periods):
    """테스트 데이터에 LAG 피처를 생성하는 함수"""
    test_df_with_lag = test_data.copy()
    
    for feature in lag_features:
        if feature in train_data.columns:
            train_series = train_data[feature]
            
            for lag in lag_periods:
                if len(train_series) >= lag:
                    # 마지막 lag 개의 값을 사용
                    lag_values = train_series.iloc[-lag:].values
                    
                    # 테스트 데이터 길이에 맞춰 반복
                    test_length = len(test_data)
                    if test_length <= len(lag_values):
                        test_df_with_lag[f'{feature}_lag_{lag}'] = lag_values[:test_length]
                    else:
                        # 순환적으로 값 할당
                        repeated_values = np.tile(lag_values, (test_length // len(lag_values)) + 1)
                        test_df_with_lag[f'{feature}_lag_{lag}'] = repeated_values[:test_length]
                else:
                    # 데이터가 부족한 경우 평균값 사용
                    test_df_with_lag[f'{feature}_lag_{lag}'] = train_series.mean()
    
    return test_df_with_lag

# TEST 데이터에 LAG 피처 적용
test_df = create_test_lag_features(train_df_clean, test_df, lag_features, lag_periods)

# TEST 데이터에 ROLLING 피처 적용 (Train 데이터의 마지막 값들 사용)
print("🎯 Test 데이터에 ROLLING 피처 생성 중...")
for feature in rolling_features:
    if feature in train_df_clean.columns:
        train_series = train_df_clean[feature]
        
        for window in rolling_windows:
            if len(train_series) >= window:
                # 마지막 window 개의 값으로 통계 계산
                last_values = train_series.iloc[-window:]
                test_df[f'{feature}_rolling_mean_{window}'] = last_values.mean()
                test_df[f'{feature}_rolling_std_{window}'] = last_values.std()
                test_df[f'{feature}_rolling_max_{window}'] = last_values.max()
                test_df[f'{feature}_rolling_min_{window}'] = last_values.min()
            else:
                # 데이터가 부족한 경우 전체 평균 사용
                test_df[f'{feature}_rolling_mean_{window}'] = train_series.mean()
                test_df[f'{feature}_rolling_std_{window}'] = train_series.std()
                test_df[f'{feature}_rolling_max_{window}'] = train_series.max()
                test_df[f'{feature}_rolling_min_{window}'] = train_series.min()

# TEST 데이터에 변화율 피처 적용
print("📈 Test 데이터에 변화율 피처 생성 중...")
for feature in ['전기요금(원)', '요금단가']:
    if feature in train_df_clean.columns:
        train_series = train_df_clean[feature]
        
        # 마지막 값과 그 이전 값들의 차이 계산
        if len(train_series) >= 2:
            test_df[f'{feature}_diff_1'] = train_series.iloc[-1] - train_series.iloc[-2]
        if len(train_series) >= 4:
            test_df[f'{feature}_diff_3'] = train_series.iloc[-1] - train_series.iloc[-4]
        if len(train_series) >= 2:
            test_df[f'{feature}_pct_change_1'] = (train_series.iloc[-1] - train_series.iloc[-2]) / train_series.iloc[-2]
        if len(train_series) >= 4:
            test_df[f'{feature}_pct_change_3'] = (train_series.iloc[-1] - train_series.iloc[-4]) / train_series.iloc[-4]

# ================================
# 🧠 8. 피처 선택 및 모델 학습
# ================================

# 사용할 피처들 정의
base_features = [
    '작업유형_encoded', '작업유형_target',
    '월', '일', '요일', '주말여부',
    'sin_시간', 'cos_시간', '요금단가'
]

# LAG 피처들
lag_feature_names = []
for feature in lag_features:
    if feature != '전기요금(원)':  # 타겟 변수는 제외
        for lag in lag_periods:
            lag_feature_names.append(f'{feature}_lag_{lag}')

# ROLLING 피처들
rolling_feature_names = []
for feature in rolling_features:
    if feature != '전기요금(원)':  # 타겟 변수는 제외
        for window in rolling_windows:
            rolling_feature_names.extend([
                f'{feature}_rolling_mean_{window}',
                f'{feature}_rolling_std_{window}',
                f'{feature}_rolling_max_{window}',
                f'{feature}_rolling_min_{window}'
            ])

# 변화율 피처들
diff_feature_names = []
for feature in ['요금단가']:  # 전기요금은 타겟이므로 제외
    diff_feature_names.extend([
        f'{feature}_diff_1', f'{feature}_diff_3',
        f'{feature}_pct_change_1', f'{feature}_pct_change_3'
    ])

# 전체 피처 리스트
all_features = base_features + lag_feature_names + rolling_feature_names + diff_feature_names

# 실제 존재하는 피처만 선택
available_features = [f for f in all_features if f in train_df_clean.columns and f in test_df.columns]

print(f"📊 사용 가능한 피처 수: {len(available_features)}")
print(f"🎯 기본 피처: {len(base_features)}")
print(f"🔄 LAG 피처: {len([f for f in lag_feature_names if f in available_features])}")
print(f"🎯 ROLLING 피처: {len([f for f in rolling_feature_names if f in available_features])}")
print(f"📈 변화율 피처: {len([f for f in diff_feature_names if f in available_features])}")

target = '전기요금(원)'

# 데이터 준비
X = train_df_clean[available_features]
y = train_df_clean[target]
X_test = test_df[available_features]

# 결측치 확인 및 처리
print(f"🔍 Train 결측치: {X.isna().sum().sum()}")
print(f"🔍 Test 결측치: {X_test.isna().sum().sum()}")

# 결측치가 있다면 0으로 채우기
X = X.fillna(0)
X_test = X_test.fillna(0)

# Train/Validation 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
print("🚀 모델 학습 시작...")
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ================================
# 📊 9. 성능 평가
# ================================
val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, val_pred)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
r2 = r2_score(y_val, val_pred)

print("\n" + "="*50)
print("📊 모델 성능 평가")
print("="*50)
print(f"✅ MAE: {mae:.2f}")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📊 R² Score: {r2:.4f}")

# 피처 중요도 상위 20개 출력
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 상위 20개 중요 피처:")
print(feature_importance.head(20).to_string(index=False))

# ================================
# 📁 10. 결과 저장
# ================================
print("\n🔮 예측 결과 생성 중...")
test_predictions = model.predict(X_test)
test_df['전기요금(원)'] = test_predictions

submission = test_df[['id', '전기요금(원)']]
submission.to_csv("submission_multi_lag.csv", index=False)

print("="*50)
print("📁 submission_multi_lag.csv 저장 완료")
print(f"📈 예측 범위: {test_predictions.min():.2f} ~ {test_predictions.max():.2f}")
print(f"📊 예측 평균: {test_predictions.mean():.2f}")
print("="*50)