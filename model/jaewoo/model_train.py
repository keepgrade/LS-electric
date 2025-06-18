# 전체 코드: 기존 '전기요금(원)' 열을 기준으로 MAE 비교

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ✅ 데이터 로딩
train_df = pd.read_csv('./data/train_with_weather.csv', parse_dates=['측정일시'])
lstm_pred_df = pd.read_csv('./data/colab_예측_9월.csv', parse_dates=['측정일시'])

# ✅ 인덱스 설정
train_df.set_index('측정일시', inplace=True)
lstm_pred_df.set_index('측정일시', inplace=True)

# ✅ 파생변수 생성 (공통: 요일, is_holiday)
train_df['요일'] = train_df.index.dayofweek
train_df['is_holiday'] = train_df['요일'].isin([6, 0]).astype(int)
lstm_pred_df['요일'] = lstm_pred_df.index.dayofweek
lstm_pred_df['is_holiday'] = lstm_pred_df['요일'].isin([6, 0]).astype(int)

# ✅ 작업유형 생성 (랜덤, 동일한 방식)
np.random.seed(42)
load_types = ['경부하', '중간부하', '최대부하']
train_df['작업유형'] = np.random.choice(load_types, size=len(train_df))
lstm_pred_df['작업유형'] = np.random.choice(load_types, size=len(lstm_pred_df))

# ✅ 학습/테스트 분리
train_data = train_df[train_df.index.month <= 8].copy()
test_data = lstm_pred_df[lstm_pred_df.index.month == 9].copy()
actual_9월 = train_df[train_df.index.month == 9]['전기요금(원)'].copy()

# ✅ 피쳐 정의
feature_cols = [
    '전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
    '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '기온', '습도',
    '요일', 'is_holiday'
]

# ✅ 학습 데이터 구성
X_train = pd.get_dummies(train_data[feature_cols + ['작업유형']])
y_train = train_data['전기요금(원)']

# ✅ 테스트 데이터 구성 (예측값 활용)
test_feature_cols = [f + '_예측' for f in feature_cols[:8]] + ['요일', 'is_holiday']
X_test = test_data[test_feature_cols].copy()
X_test.columns = feature_cols
X_test = pd.get_dummies(pd.concat([X_test, test_data['작업유형']], axis=1))

# ✅ 컬럼 정렬
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# ✅ 모델 학습 및 예측
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
test_pred = model.predict(X_test)

# ✅ MAE 계산
mae = mean_absolute_error(actual_9월.values.flatten(), test_pred)
mae



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import os

# ✅ 데이터 로드
train_df = pd.read_csv('./data/train_with_weather.csv', parse_dates=['측정일시'])
pred_df = pd.read_csv('./data/lstm_6feature.csv', parse_dates=['측정일시'])
train_df
# ✅ 인덱스 설정 및 파생 변수 생성
for df in [train_df, pred_df]:
    df.set_index('측정일시', inplace=True)
    df['요일'] = df.index.dayofweek
    df['is_holiday'] = df['요일'].isin([6, 0]).astype(int)

# ✅ 작업유형 랜덤 생성 (일관성 유지)
np.random.seed(42)
load_types = ['경부하', '중간부하', '최대부하']
train_df['작업유형'] = np.random.choice(load_types, size=len(train_df))
pred_df['작업유형'] = np.random.choice(load_types, size=len(pred_df))

# ✅ 변수 정의
exclude_cols = ['지상무효전력량(kVarh)', '진상무효전력량(kVarh)']
base_features = ['전력사용량(kWh)', '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '기온', '습도']
time_features = ['요일', 'is_holiday']
feature_cols = base_features + time_features

# ✅ 학습/테스트 분리
train_data = train_df[train_df.index.month <= 11].copy()
test_data = pred_df[pred_df.index.month == 12].copy()

# ✅ 학습 데이터
X_train = pd.get_dummies(train_data[feature_cols + ['작업유형']])
y_train = train_data['전기요금(원)']

# ✅ 테스트 데이터
test_feature_cols = [f + '_예측' for f in base_features] + time_features
X_test = test_data[test_feature_cols].copy()
X_test.columns = feature_cols
X_test = pd.get_dummies(pd.concat([X_test, test_data['작업유형']], axis=1))

# ✅ 컬럼 정렬
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# ✅ 하이퍼파라미터 탐색
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
scorer = make_scorer(mean_squared_error, greater_is_better=False)
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring=scorer, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# ✅ 예측
test_pred = best_model.predict(X_test)
test_result_df = test_data.copy()
test_result_df['전기요금(예측)'] = test_pred

# ✅ 저장
os.makedirs('./data', exist_ok=True)
save_path = './data/전기요금_예측결과_12월_최적화.csv'
test_result_df[['전기요금(예측)']].to_csv(save_path)

save_path


import pandas as pd

# 데이터 로딩
df = pd.read_csv('./data/train_with_weather.csv', parse_dates=['측정일시'])
df.set_index('측정일시', inplace=True)

# 일별 평균 전력사용량 계산
daily_avg = df['전력사용량(kWh)'].resample('D').mean()

# 요일 정보 추가 (0: 월요일, 6: 일요일)
daily_avg_df = daily_avg.to_frame(name='일별평균전력량(kWh)')
daily_avg_df['요일'] = daily_avg_df.index.dayofweek

# 조건 필터링: 평균 5 이하 & 요일이 월/일 제외
filtered = daily_avg_df[
    (daily_avg_df['일별평균전력량(kWh)'] <= 10) &
    (~daily_avg_df['요일'].isin([0, 6]))
]

print(filtered)

filtered.reset_index()["측정일시"]












import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# ✅ 경로 설정
train_path = "./data/train.csv"
test_path = "./data/test.csv"
sarima_model_path = "./data/sarima_model.pkl"
sarima_forecast_path = "./data/power_forecast_december.csv"
submission_path = "./data/submission_full_pipeline.csv"

# ✅ 데이터 불러오기 및 정렬
train = pd.read_csv(train_path, parse_dates=["측정일시"]).sort_values("측정일시")
test = pd.read_csv(test_path, parse_dates=["측정일시"]).sort_values("측정일시")
train.set_index("측정일시", inplace=True)
test.set_index("측정일시", inplace=True)

# ✅ 15분 단위 빈도 지정 (경고 제거)
train = train.asfreq("15min")
test = test.asfreq("15min")

# ✅ SARIMA 학습 대상
train_power = train["전력사용량(kWh)"]
train_power_months = train_power[train_power.index.month <= 11]

# ✅ SARIMA 모델 학습
sarima_model = SARIMAX(
    train_power_months,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 96),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_result = sarima_model.fit(disp=True)

# ✅ SARIMA 모델 저장
joblib.dump(sarima_result, sarima_model_path)

# ✅ 12월 예측 시계열 생성
forecast_index = pd.date_range("2024-12-01 00:00:00", "2024-12-31 23:45:00", freq="15min")
forecast_len = len(forecast_index)

# ✅ 전력사용량 예측 및 저장
power_forecast = sarima_result.forecast(steps=forecast_len)
power_forecast_df = pd.DataFrame({
    "측정일시": forecast_index,
    "전력사용량(kWh)": power_forecast.values
})
power_forecast_df.to_csv(sarima_forecast_path, index=False)

# ✅ 테스트셋에 예측값 추가
test["전력사용량(kWh)"] = power_forecast.values

# ✅ 시간대별 요금 단가 함수
def get_tariff_info(dt):
    m, h = dt.month, dt.hour
    if m in [12, 1, 2, 11]:
        if 22 <= h or h < 8:
            return "경부하", 100.4
        elif 8 <= h < 9 or 12 <= h < 16 or 19 <= h < 22:
            return "중간부하", 146.5
        else:
            return "최대부하", 193.4
    elif m in [6, 7, 8]:
        if 22 <= h or h < 8:
            return "경부하", 93.1
        elif 8 <= h < 11 or 12 <= h < 13 or 18 <= h < 22:
            return "중간부하", 146.3
        else:
            return "최대부하", 216.6
    else:
        if 22 <= h or h < 8:
            return "경부하", 93.1
        elif 8 <= h < 11 or 12 <= h < 13 or 18 <= h < 22:
            return "중간부하", 115.2
        else:
            return "최대부하", 138.9

# ✅ 파생 변수 생성
train["요일"] = train.index.dayofweek
train["is_holiday"] = train["요일"].isin([6, 0]).astype(int)
train["부하시간대"], train["단가"] = zip(*train.index.map(get_tariff_info))
train["전기요금(원)"] = train["전력사용량(kWh)"] * train["단가"]

test["요일"] = test.index.dayofweek
test["is_holiday"] = test["요일"].isin([6, 0]).astype(int)
test["부하시간대"], test["단가"] = zip(*test.index.map(get_tariff_info))

# ✅ 학습 및 예측
features = ["전력사용량(kWh)", "요일", "is_holiday"]
X_train = train[features]
y_train = train["전기요금(원)"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

X_test = test[features]
test["전기요금(예측)"] = model.predict(X_test)

# ✅ 제출 파일 생성
submission = test[["id", "전기요금(예측)"]].reset_index(drop=True)
submission.to_csv(submission_path, index=False)

print("✅ SARIMA 모델 저장:", sarima_model_path)
print("✅ 전력 예측 결과 저장:", sarima_forecast_path)
print("✅ 최종 제출 파일 저장:", submission_path)
