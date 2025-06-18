import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Malgun Gothic'
# 데이터 로딩
train_df = pd.read_csv("./data/train.csv")  # 경로는 실제 파일 위치에 맞게 조정
train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])

# 파생 변수 생성
train_df['요일'] = train_df['측정일시'].dt.day_name()
train_df['시간'] = train_df['측정일시'].dt.hour
train_df['월'] = train_df['측정일시'].dt.month

# 수치형 변수 리스트
numeric_cols = [
    '전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
    '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '전기요금(원)'
]
train_df.columns
# 요일 순서 지정
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 시각화 함수 정의
def plot_comparison_by_timeunit(df, col, time_unit, order=None):
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x=time_unit, y=col, order=order)
    plt.title(f"{time_unit}별 {col} 분포")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 시각화 실행
for col in numeric_cols:
    plot_comparison_by_timeunit(train_df, col, '요일', order=weekday_order)
    plot_comparison_by_timeunit(train_df, col, '시간')
    plot_comparison_by_timeunit(train_df, col, '월')


import seaborn as sns
import matplotlib.pyplot as plt

# 동일한 전력사용량을 기준으로 요금의 월별 차이 확인
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=train_df,
    x='전력사용량(kWh)',
    y='전기요금(원)',
    hue='월',
    palette='tab10',
    alpha=0.6
)
plt.title("전력사용량(kWh) vs 전기요금(원) (월별 구분)")
plt.tight_layout()
plt.show()



# 요금제별 단가 정의: [고압A/B], [선택Ⅰ/Ⅱ], [2024.01.01, 2024.10.24]
# 구조: (요금제명, 날짜구간, 부하 → 계절 → 단가)
import numpy as np
from datetime import datetime

# 날짜 구간 나누기
before_change = train_df['측정일시'] < pd.to_datetime("2024-10-24")
after_change = ~before_change

# 시간대 분류 함수는 이미 정의되어 있음: get_time_block

# 요금표 (고압A/B 선택1,2 각각 2시기)
tariffs = {
    "고압A_선택1_20240101": {
        "summer": {"경부하": 99.5, "중간부하": 152.4, "최대부하": 234.5},
        "spring_fall": {"경부하": 99.5, "중간부하": 122.0, "최대부하": 152.7},
        "winter": {"경부하": 106.5, "중간부하": 152.6, "최대부하": 210.1},
    },
    "고압A_선택1_20241024": {
        "summer": {"경부하": 116.4, "중간부하": 169.3, "최대부하": 251.4},
        "spring_fall": {"경부하": 116.4, "중간부하": 138.9, "최대부하": 169.6},
        "winter": {"경부하": 123.4, "중간부하": 169.5, "최대부하": 227.0},
    },
    "고압A_선택2_20240101": {
        "summer": {"경부하": 94.0, "중간부하": 146.9, "최대부하": 229.0},
        "spring_fall": {"경부하": 94.0, "중간부하": 116.5, "최대부하": 147.2},
        "winter": {"경부하": 101.0, "중간부하": 147.1, "최대부하": 204.6},
    },
    "고압A_선택2_20241024": {
        "summer": {"경부하": 110.9, "중간부하": 163.8, "최대부하": 245.9},
        "spring_fall": {"경부하": 110.9, "중간부하": 133.4, "최대부하": 164.1},
        "winter": {"경부하": 117.9, "중간부하": 164.0, "최대부하": 221.5},
    },
    "고압B_선택1_20240101": {
        "summer": {"경부하": 109.4, "중간부하": 161.7, "최대부하": 242.9},
        "spring_fall": {"경부하": 109.4, "중간부하": 131.7, "최대부하": 162.0},
        "winter": {"경부하": 116.4, "중간부하": 161.7, "최대부하": 217.9},
    },
    "고압B_선택1_20241024": {
        "summer": {"경부하": 126.3, "중간부하": 178.6, "최대부하": 259.8},
        "spring_fall": {"경부하": 126.3, "중간부하": 148.6, "최대부하": 178.9},
        "winter": {"경부하": 133.3, "중간부하": 178.6, "최대부하": 234.8},
    },
    "고압B_선택2_20240101": {
        "summer": {"경부하": 105.6, "중간부하": 157.9, "최대부하": 239.1},
        "spring_fall": {"경부하": 105.6, "중간부하": 127.9, "최대부하": 158.2},
        "winter": {"경부하": 112.6, "중간부하": 157.9, "최대부하": 214.1},
    },
    "고압B_선택2_20241024": {
        "summer": {"경부하": 122.5, "중간부하": 174.8, "최대부하": 256.0},
        "spring_fall": {"경부하": 122.5, "중간부하": 144.8, "최대부하": 175.1},
        "winter": {"경부하": 129.5, "중간부하": 174.8, "최대부하": 231.0},
    }
}

# 오차 계산 함수
def estimate_tariff_error(df, rate_set):
    est_fees = []
    for _, row in df.iterrows():
        hour = row['시간']
        month = row['월']
        usage = row['전력사용량(kWh)']
        time_block, season_key = get_time_block(hour, month)
        rate = rate_set[season_key][time_block]
        est_fees.append(usage * rate)
    return np.mean(np.abs(df['전기요금(원)'] - est_fees))

# 전체 요금제 평균 오차 비교
results = {}
for name, rate_set in tariffs.items():
    if "20240101" in name:
        df_sub = train_df[before_change]
    else:
        df_sub = train_df[after_change]

    error = estimate_tariff_error(df_sub, rate_set)
    results[name] = error

# 가장 오차가 작은 요금제 정렬
sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
sorted_results


pd.read_csv("./data/test.csv")





import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import holidays

# 복사 및 기본 파생변수 생성
df = train_df.copy()
df['월'] = df['측정일시'].dt.month
df['요일'] = df['측정일시'].dt.dayofweek
df['hour'] = df['측정일시'].dt.hour
kr_holidays = holidays.KR(years=[2024])
df['휴일'] = df['측정일시'].dt.date.apply(lambda x: x in kr_holidays or pd.to_datetime(x).weekday() >= 5)

# 계절/시간대 분류 함수
def get_time_block(hour, month):
    if month in [6, 7, 8]:
        season = "summer"
    elif month in [3, 4, 5, 9, 10]:
        season = "spring_fall"
    else:
        season = "winter"

    if season in ["summer", "spring_fall"]:
        if 22 <= hour or hour < 8:
            block = "경부하"
        elif 8 <= hour < 11 or 12 <= hour < 13 or 18 <= hour < 22:
            block = "중간부하"
        else:
            block = "최대부하"
    else:
        if 22 <= hour or hour < 8:
            block = "경부하"
        elif 8 <= hour < 9 or 12 <= hour < 16 or 19 <= hour < 22:
            block = "중간부하"
        else:
            block = "최대부하"
    return block, season

# 고압A 선택3 요금 단가
tariff_a3_20240101 = {
    "summer": {"경부하": 93.1, "중간부하": 146.3, "최대부하": 216.6},
    "spring_fall": {"경부하": 93.1, "중간부하": 115.2, "최대부하": 138.9},
    "winter": {"경부하": 100.4, "중간부하": 146.5, "최대부하": 193.4}
}

tariff_a3_20241024 = {
    "summer": {"경부하": 110.0, "중간부하": 163.2, "최대부하": 233.5},
    "spring_fall": {"경부하": 110.0, "중간부하": 132.1, "최대부하": 155.8},
    "winter": {"경부하": 117.3, "중간부하": 163.4, "최대부하": 210.3}
}

# 요금제 단가 파생변수 생성
def estimate_a3_fee_v2(row):
    block, season = get_time_block(row['hour'], row['월'])
    기준일 = pd.to_datetime("2024-10-24")
    if row['측정일시'] < 기준일:
        rate = tariff_a3_20240101[season][block]
    else:
        rate = tariff_a3_20241024[season][block]
    return rate

df['요금제'] = df.apply(estimate_a3_fee_v2, axis=1)

# 학습/평가 데이터 분리 (6월 제외 학습, 6월 테스트)
train_data = df[df['월'].isin([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])]
test_data = df[df['월'] == 6]

X_train = train_data[['월', '요일', '휴일', 'hour', '작업유형', '요금제']]
y_train = train_data['전기요금(원)']
X_test = test_data[['월', '요일', '휴일', 'hour', '작업유형', '요금제']]
y_test = test_data['전기요금(원)']

# 전처리 및 모델 정의
categorical_features = ['요일', '휴일', '작업유형']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough'
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# 학습 및 평가
final_results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    final_results[name] = mse

final_results





import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import holidays

train_df = pd.read_csv("./data/train.csv")

# 원본 데이터 복사
df = train_df.copy()
df['측정일시'] = pd.to_datetime(df['측정일시'])
# 기본 파생변수 생성
df['월'] = df['측정일시'].dt.month
df['요일'] = df['측정일시'].dt.dayofweek
df['hour'] = df['측정일시'].dt.hour
kr_holidays = holidays.KR(years=[2024])
df['휴일'] = df['측정일시'].dt.date.apply(lambda x: x in kr_holidays or pd.to_datetime(x).weekday() >= 5)

# 시간대-계절별 구분 함수
def get_time_block(hour, month):
    if month in [6, 7, 8]:
        season = "summer"
    elif month in [3, 4, 5, 9, 10]:
        season = "spring_fall"
    else:
        season = "winter"
    if season in ["summer", "spring_fall"]:
        if 22 <= hour or hour < 8:
            block = "경부하"
        elif 8 <= hour < 11 or 12 <= hour < 13 or 18 <= hour < 22:
            block = "중간부하"
        else:
            block = "최대부하"
    else:
        if 22 <= hour or hour < 8:
            block = "경부하"
        elif 8 <= hour < 9 or 12 <= hour < 16 or 19 <= hour < 22:
            block = "중간부하"
        else:
            block = "최대부하"
    return block, season

# 고압A 선택3 요금제
tariff_a3_20240101 = {
    "summer": {"경부하": 93.1, "중간부하": 146.3, "최대부하": 216.6},
    "spring_fall": {"경부하": 93.1, "중간부하": 115.2, "최대부하": 138.9},
    "winter": {"경부하": 100.4, "중간부하": 146.5, "최대부하": 193.4}
}
tariff_a3_20241024 = {
    "summer": {"경부하": 110.0, "중간부하": 163.2, "최대부하": 233.5},
    "spring_fall": {"경부하": 110.0, "중간부하": 132.1, "최대부하": 155.8},
    "winter": {"경부하": 117.3, "중간부하": 163.4, "최대부하": 210.3}
}

def estimate_a3_fee(row):
    block, season = get_time_block(row['hour'], row['월'])
    기준일 = pd.to_datetime("2024-10-24")
    rate = tariff_a3_20240101[season][block] if row['측정일시'] < 기준일 else tariff_a3_20241024[season][block]
    return rate

df['요금제'] = df.apply(estimate_a3_fee, axis=1)

# 6월만 테스트 데이터로 분리, 나머지는 학습
train_data = df[df['월'].isin([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])]
test_data = df[df['월'] == 6]

X_train = train_data[['월', '요일', '휴일', 'hour', '작업유형', '요금제']]
y_train = train_data['전기요금(원)']
X_test = test_data[['월', '요일', '휴일', 'hour', '작업유형', '요금제']]
y_test = test_data['전기요금(원)']

# 전처리 및 모델 정의
categorical_features = ['요일', '휴일', '작업유형']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# MAE로 평가
mae_results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_results[name] = mae

mae_results














import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import holidays

# 데이터 불러오기
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# 날짜 변환
train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])
test_df['측정일시'] = pd.to_datetime(test_df['측정일시'])

# 파생변수 생성 함수
def create_features(df):
    df['월'] = df['측정일시'].dt.month
    df['요일'] = df['측정일시'].dt.dayofweek
    kr_holidays = holidays.KR(years=[2024])
    df['휴일'] = df['측정일시'].dt.date.apply(lambda x: x in kr_holidays or pd.to_datetime(x).weekday() >= 5)
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

# 부하 요금표 (작업유형을 부하로 간주)
tariff_map_20240101 = {
    "Light_Load": {"summer": 93.1, "spring_fall": 93.1, "winter": 100.4},
    "Medium_Load": {"summer": 146.3, "spring_fall": 115.2, "winter": 146.5},
    "Maximum_Load": {"summer": 216.6, "spring_fall": 138.9, "winter": 193.4}
}
tariff_map_20241024 = {
    "Light_Load": {"summer": 110.0, "spring_fall": 110.0, "winter": 117.3},
    "Medium_Load": {"summer": 163.2, "spring_fall": 132.1, "winter": 163.4},
    "Maximum_Load": {"summer": 233.5, "spring_fall": 155.8, "winter": 210.3}
}

# 계절 구분
def get_season(month):
    if month in [6, 7, 8]:
        return "summer"
    elif month in [3, 4, 5, 9, 10]:
        return "spring_fall"
    else:
        return "winter"

# 요금제 계산 함수
def compute_tariff(row):
    기준일 = pd.to_datetime("2024-10-24")
    season = get_season(row['월'])
    if row['측정일시'] < 기준일:
        return tariff_map_20240101[row['작업유형']][season]
    else:
        return tariff_map_20241024[row['작업유형']][season]

# 요금제 적용
train_df['요금제'] = train_df.apply(compute_tariff, axis=1)
test_df['요금제'] = test_df.apply(compute_tariff, axis=1)

# 최종 모델 feature 지정 (작업유형 제거됨)
X_train = train_df[['월', '요일', '휴일', '요금제']]
y_train = train_df['전기요금(원)']
X_test = test_df[['월', '요일', '휴일', '요금제']]

# 전처리 및 모델 정의
categorical_features = ['요일', '휴일']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# 예측 수행
predictions = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    predictions[name] = preds

# 최종 결과 저장 (XGBoost 기준)
test_df['예측_전기요금(원)_XGB'] = predictions['XGBoost']
submission = test_df[['id', '예측_전기요금(원)_XGB']].copy()
submission.columns = ['id', '전기요금(원)']
submission.to_csv("./submission.csv", index=False)








import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import holidays

# 데이터 불러오기
train_df = pd.read_csv("./data/train.csv")
train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])

# 파생변수 생성
def create_features(df):
    df['월'] = df['측정일시'].dt.month
    df['요일'] = df['측정일시'].dt.dayofweek
    kr_holidays = holidays.KR(years=[2024])
    df['휴일'] = df['측정일시'].dt.date.apply(lambda x: x in kr_holidays or pd.to_datetime(x).weekday() >= 5)
    return df

train_df = create_features(train_df)

# 고압A 선택3 요금제 단가표
tariff_map_20240101 = {
    "Light_Load": {"summer": 93.1, "spring_fall": 93.1, "winter": 100.4},
    "Medium_Load": {"summer": 146.3, "spring_fall": 115.2, "winter": 146.5},
    "Maximum_Load": {"summer": 216.6, "spring_fall": 138.9, "winter": 193.4}
}
tariff_map_20241024 = {
    "Light_Load": {"summer": 110.0, "spring_fall": 110.0, "winter": 117.3},
    "Medium_Load": {"summer": 163.2, "spring_fall": 132.1, "winter": 163.4},
    "Maximum_Load": {"summer": 233.5, "spring_fall": 155.8, "winter": 210.3}
}

# 계절 구분
def get_season(month):
    if month in [6, 7, 8]:
        return "summer"
    elif month in [3, 4, 5, 9, 10]:
        return "spring_fall"
    else:
        return "winter"

# 요금제 계산 함수
def compute_tariff(row):
    기준일 = pd.to_datetime("2024-10-24")
    season = get_season(row['월'])
    if row['측정일시'] < 기준일:
        return tariff_map_20240101[row['작업유형']][season]
    else:
        return tariff_map_20241024[row['작업유형']][season]

# 요금제 적용
train_df['요금제'] = train_df.apply(compute_tariff, axis=1)

# 6월을 테스트로 분리
train_data = train_df[train_df['월'] != 6]
test_data = train_df[train_df['월'] == 6]

X_train = train_data[['월', '요일', '휴일', '요금제']]
y_train = train_data['전기요금(원)']
X_test = test_data[['월', '요일', '휴일', '요금제']]
y_test = test_data['전기요금(원)']

# 전처리
categorical_features = ['요일', '휴일']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# 모델 정의
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

# MAE 평가
mae_results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mae_results[name] = mae
    print(f"{name} → MAE: {mae:.2f}")

# 최종 결과 출력
print("\n✅ 전체 MAE 결과:")
print(mae_results)



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# 1. Load & Sort
# -----------------------
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])
test_df['측정일시'] = pd.to_datetime(test_df['측정일시'])

train_df = train_df.sort_values("측정일시").reset_index(drop=True)
test_df = test_df.sort_values("측정일시").reset_index(drop=True)

# -----------------------
# 2. Create Time Features
# -----------------------
def create_time_features(df):
    df['hour'] = df['측정일시'].dt.hour
    df['minute'] = df['측정일시'].dt.minute
    df['dayofweek'] = df['측정일시'].dt.dayofweek
    return df

train_df = create_time_features(train_df)
test_df = create_time_features(test_df)

# -----------------------
# 3. Combine last 24 rows of train with test
# -----------------------
window_size = 24
combined_df = pd.concat([train_df.tail(window_size), test_df], ignore_index=True)

# -----------------------
# 4. Select Features and Normalize
# -----------------------
feature_cols = ['hour', 'minute', 'dayofweek']
target_cols = ['지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
               '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)']

X_all_raw = combined_df[feature_cols].values
y_train_raw = train_df[target_cols].values  # Only training labels

# Scaling
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_all_scaled = X_scaler.fit_transform(X_all_raw)
y_train_scaled = y_scaler.fit_transform(y_train_raw)

# -----------------------
# 5. Create Sequences
# -----------------------
def create_sequences(X, y=None, window_size=24):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        if y is not None:
            ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys) if y is not None else np.array(Xs)

# 학습 데이터: 전체 시퀀스에서 train 길이만큼만 사용
X_seq_all, y_seq_all = create_sequences(X_all_scaled, None, window_size)
X_seq_train = X_seq_all[:len(train_df) - window_size]
y_seq_train = y_train_scaled[window_size:]

# 테스트 데이터: 마지막 test 길이만큼 시퀀스 자르기
X_seq_test = X_seq_all[-len(test_df):]

# -----------------------
# 6. LSTM Model
# -----------------------
model = Sequential([
    LSTM(64, input_shape=(window_size, X_seq_train.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(len(target_cols))  # 5개 출력
])

model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_seq_train, y_seq_train, validation_split=0.2,
          epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

# -----------------------
# 7. Predict and Inverse Transform
# -----------------------
y_pred_scaled = model.predict(X_seq_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# -----------------------
# 8. Save Predictions
# -----------------------
result_df = test_df.copy()
result_df[target_cols] = y_pred
result_df.to_csv("test_predicted.csv", index=False)

print(result_df[target_cols].head())


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Load data
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./test_predicted.csv")
train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])
test_df['측정일시'] = pd.to_datetime(test_df['측정일시'])

# Feature engineering
def create_features(df):
    df['월'] = df['측정일시'].dt.month
    df['요일'] = df['측정일시'].dt.dayofweek
    df['hour'] = df['측정일시'].dt.hour
    df['휴일'] = df['요일'].isin([0, 6])  # Monday, Sunday = 휴일
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

# 요금제 계산 함수
def get_time_block(hour, month):
    if month in [6, 7, 8]: season = "summer"
    elif month in [3, 4, 5, 9, 10]: season = "spring_fall"
    else: season = "winter"
    if season in ["summer", "spring_fall"]:
        if 22 <= hour or hour < 8: block = "경부하"
        elif 8 <= hour < 11 or 12 <= hour < 13 or 18 <= hour < 22: block = "중간부하"
        else: block = "최대부하"
    else:
        if 22 <= hour or hour < 8: block = "경부하"
        elif 8 <= hour < 9 or 12 <= hour < 16 or 19 <= hour < 22: block = "중간부하"
        else: block = "최대부하"
    return block, season

tariff_a3_20240101 = {
    "summer": {"경부하": 93.1, "중간부하": 146.3, "최대부하": 216.6},
    "spring_fall": {"경부하": 93.1, "중간부하": 115.2, "최대부하": 138.9},
    "winter": {"경부하": 100.4, "중간부하": 146.5, "최대부하": 193.4}
}
tariff_a3_20241024 = {
    "summer": {"경부하": 110.0, "중간부하": 163.2, "최대부하": 233.5},
    "spring_fall": {"경부하": 110.0, "중간부하": 132.1, "최대부하": 155.8},
    "winter": {"경부하": 117.3, "중간부하": 163.4, "최대부하": 210.3}
}

def estimate_a3_fee(row):
    block, season = get_time_block(row['hour'], row['월'])
    기준일 = pd.to_datetime("2024-10-24")
    return tariff_a3_20240101[season][block] if row['측정일시'] < 기준일 else tariff_a3_20241024[season][block]

train_df['요금제'] = train_df.apply(estimate_a3_fee, axis=1)
test_df['요금제'] = test_df.apply(estimate_a3_fee, axis=1)

# Features
lstm_cols = ['지상무효전력량(kVarh)', '진상무효전력량(kVarh)', '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)']
feature_cols = ['월', '요일', '휴일', 'hour', '요금제'] + lstm_cols
X_train = train_df[feature_cols]
y_train = train_df['전기요금(원)']
X_test = test_df[feature_cols]

# Preprocessing
categorical_features = ['요일', '휴일']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Model 정의 및 파라미터 설정
models = {
    "Linear Regression": (LinearRegression(), {}),
    "Random Forest": (
        RandomForestRegressor(random_state=42),
        {
            'regressor__n_estimators': randint(100, 300),
            'regressor__max_depth': randint(3, 15)
        }
    ),
    "XGBoost": (
        XGBRegressor(random_state=42),
        {
            'regressor__n_estimators': randint(100, 300),
            'regressor__max_depth': randint(3, 15),
            'regressor__learning_rate': uniform(0.01, 0.2)
        }
    ),
    "LightGBM": (
        LGBMRegressor(random_state=42),
        {
            'regressor__n_estimators': randint(100, 300),
            'regressor__max_depth': randint(3, 15),
            'regressor__learning_rate': uniform(0.01, 0.2)
        }
    )
}

# MAE 저장
mae_results = {}
final_predictions = {}

# 모델별 학습 및 예측
for name, (model, param_dist) in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    if param_dist:
        search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                    n_iter=20, cv=3, scoring='neg_mean_absolute_error',
                                    n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    preds = best_model.predict(X_test)
    final_predictions[name] = preds

    # 평가용도 - train을 다시 예측해서 MAE 측정
    train_pred = best_model.predict(X_train)
    mae = mean_absolute_error(y_train, train_pred)
    mae_results[name] = mae

# MAE 출력
print("모델별 MAE 결과:")
for name, score in mae_results.items():
    print(f"{name}: {score:.2f}")

# 가장 좋은 모델 선택
best_model_name = min(mae_results, key=mae_results.get)
print(f"\n✅ 가장 성능 좋은 모델: {best_model_name}")

# 해당 모델의 예측 결과 저장
test_df['예측_전기요금(원)'] = final_predictions[best_model_name]
submission = test_df[['id', '예측_전기요금(원)']].copy()
submission.columns = ['id', '전기요금(원)']
submission.to_csv("./data/submission_best_model.csv", index=False)
print("\n📁 최종 예측 결과 저장 완료: ./data/submission_best_model.csv")


submission.head()
submission.info()
a = pd.read_csv("./data/test.csv")

a.info()

b = pd.read_csv("./submission.csv")
b.info()

c = pd.read_csv("./data/submission_best_model.csv")
c.info()


import pandas as pd

# 데이터 로드
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
weather = pd.read_csv("./data/weather.csv")
train.head()
weather.head()
# datetime 변환
train['측정일시'] = pd.to_datetime(train['측정일시'])
test['측정일시'] = pd.to_datetime(test['측정일시'])
weather['일시'] = pd.to_datetime(weather['일시'], errors='coerce')

# ⏱️ 정확히 15분 간격으로 반올림 (round) 처리
weather['측정일시'] = weather['일시'].dt.round('15T')

# 🧹 혹시 모를 결측치 제거
weather = weather.dropna(subset=['측정일시', '기온(°C)', '습도(%)'])

# 평균 집계
weather_grouped = weather.groupby('측정일시')[['기온(°C)', '습도(%)']].mean().reset_index()
weather_grouped = weather_grouped.rename(columns={'기온(°C)': '기온', '습도(%)': '습도'})

# 병합
train_merged = pd.merge(train, weather_grouped, on='측정일시', how='left')
test_merged = pd.merge(test, weather_grouped, on='측정일시', how='left')

# 결측 보간 (옵션)
train_merged[['기온', '습도']] = train_merged[['기온', '습도']].fillna(method='ffill').fillna(method='bfill')
test_merged[['기온', '습도']] = test_merged[['기온', '습도']].fillna(method='ffill').fillna(method='bfill')

# 저장
train_merged.to_csv("./data/train_with_weather.csv", index=False)
test_merged.to_csv("./data/test_with_weather.csv", index=False)

# 확인
print(train_merged[['측정일시', '기온', '습도']].head(10))
print("\n✅ 병합 완료")

train_merged.info()
train_merged.columns


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# 데이터 로딩
df = pd.read_csv("./data/train_with_weather.csv")

# datetime으로 변환 및 인덱스 설정
df['측정일시'] = pd.to_datetime(df['측정일시'])
df.set_index('측정일시', inplace=True)

# 스케일링 대상 수치형 열만 선택
target_cols = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
               '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '기온', '습도']

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[target_cols])
scaled_df = pd.DataFrame(scaled, index=df.index, columns=target_cols)

# 학습/테스트 데이터 구성
X_train, y_train, X_test, y_test, test_timestamps = [], [], [], [], []

for current_time in scaled_df.index:
    if current_time.month == 9:
        past_times = [current_time - timedelta(weeks=w) for w in range(8, 0, -1)]
        if all(t in scaled_df.index for t in past_times):
            X_test.append(scaled_df.loc[past_times].values)
            y_test.append(scaled_df.loc[current_time].values)
            test_timestamps.append(current_time)
    elif current_time.month < 9:
        past_times = [current_time - timedelta(weeks=w) for w in range(8, 0, -1)]
        if all(t in scaled_df.index for t in past_times):
            X_train.append(scaled_df.loc[past_times].values)
            y_train.append(scaled_df.loc[current_time].values)

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)
test_timestamps = np.array(test_timestamps)

print(X_train.shape, X_test.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Build an enhanced LSTM model with regularization and normalization
model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(8, 8)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(8)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_split=0.1, 
                    callbacks=[early_stop],
                    verbose=0)

# Predict on September test set
y_pred_scaled = model.predict(X_test)

# Inverse transform predictions and targets
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

# Create DataFrame for comparison
# 수치형 target 컬럼만 명시적으로 사용
target_cols = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
               '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '기온', '습도']

# 수정된 결과 데이터프레임 생성
results_df = pd.DataFrame(y_true, columns=target_cols)
results_df['측정일시'] = test_timestamps
results_df = results_df.set_index('측정일시')

pred_df = pd.DataFrame(y_pred, columns=[col + '_예측' for col in target_cols])
pred_df['측정일시'] = test_timestamps
pred_df = pred_df.set_index('측정일시')

# 결과 병합
full_result = pd.concat([results_df, pred_df], axis=1)

import ace_tools as tools 
tools.display_dataframe_to_user(name="LSTM 9월 예측 결과 비교", dataframe=full_result)


# Full corrected version of single-target LSTM per variable with proper training/test split
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load and preprocess data
df = pd.read_csv("./data/train_with_weather.csv", encoding="utf-8")
df['측정일시'] = pd.to_datetime(df['측정일시'])
df.set_index('측정일시', inplace=True)

# Target variables
target_cols = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
               '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '기온', '습도']

# Normalize
scaler = MinMaxScaler()
scaled_all = scaler.fit_transform(df[target_cols])
data_scaled = pd.DataFrame(scaled_all, columns=target_cols, index=df.index)

# Define sequence length (7 days of 15-minute intervals)
seq_len = 96 * 7

# Prepare combined result container
results_combined = []

# Iterate over each variable
for target in target_cols:
    X, y, timestamps = [], [], []

    for i in range(seq_len, len(data_scaled) - 1):
        input_seq = data_scaled.iloc[i - seq_len:i].values
        target_value = data_scaled[target].iloc[i + 1]
        timestamp = data_scaled.index[i + 1]

        X.append(input_seq)
        y.append(target_value)
        timestamps.append(timestamp)

    X = np.array(X)
    y = np.array(y)
    timestamps = np.array(timestamps)

    # Split: before Sep = train, Sep = test
    train_idx = [i for i, t in enumerate(timestamps) if t < pd.Timestamp("2024-09-01")]
    test_idx = [i for i, t in enumerate(timestamps) if pd.Timestamp("2024-09-01") <= t < pd.Timestamp("2024-10-01")]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_timestamps = timestamps[test_idx]

    if len(X_train) < 10 or len(X_test) == 0:
        print(f"❌ {target}: 학습/테스트 데이터가 부족합니다. 건너뜁니다.")
        continue

    # Build LSTM model
    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train with conditional validation
    if len(X_train) < 100:
        model.fit(X_train, y_train, epochs=30, batch_size=8, callbacks=[early_stop], verbose=1)
    else:
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=1)

    # Predict and inverse transform
    y_pred_scaled = model.predict(X_test).flatten().reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    target_scaler = MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[target_cols.index(target)], scaler.scale_[target_cols.index(target)]
    y_pred = y_pred_scaled * target_scaler.scale_ + target_scaler.min_
    y_true = y_test_scaled * target_scaler.scale_ + target_scaler.min_

    # Store results
    temp_df = pd.DataFrame({
        '측정일시': test_timestamps,
        target: y_true.flatten(),
        f"{target}_예측": y_pred.flatten()
    })
    results_combined.append(temp_df)

# Merge all results
final_result = results_combined[0]
for df_part in results_combined[1:]:
    final_result = pd.merge(final_result, df_part, on='측정일시')

final_result.set_index('측정일시', inplace=True)

# Save to CSV
os.makedirs("./data", exist_ok=True)
output_path = "./data/LSTM_개별모델_예측_vs_실제.csv"
final_result.to_csv(output_path, encoding="utf-8-sig")

output_path


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 원본 학습 데이터에서 스케일러 학습
train_df = pd.read_csv("./data/train_with_weather.csv", encoding="utf-8")
target_cols = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)',
               '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)', '기온', '습도']

scaler = MinMaxScaler()
scaler.fit(train_df[target_cols])

# 2. 예측 결과 불러오기
scaled_df = pd.read_csv("./data/LSTM_개별모델_예측_vs_실제.csv", encoding="utf-8-sig")

# 3. 스케일된 값들만 따로 추출
scaled_only = scaled_df[[col for col in scaled_df.columns if col in target_cols or col.endswith("_예측")]]

# 4. 역변환 가능한 구조로 재정렬
# 예: ["전력사용량(kWh)", "전력사용량(kWh)_예측", "지상무효전력량(kVarh)", "지상무효전력량(kVarh)_예측", ...]
recovered_parts = []
for col in target_cols:
    pair = scaled_df[[col, f"{col}_예측"]].copy()
    scaler_col = MinMaxScaler()
    scaler_col.fit(train_df[[col]])  # 개별 컬럼만 학습
    pair[[col, f"{col}_예측"]] = scaler_col.inverse_transform(pair[[col, f"{col}_예측"]])
    recovered_parts.append(pair)

# 5. 컬럼 통합
restored_df = pd.concat([scaled_df['측정일시']] + recovered_parts, axis=1)
restored_df.to_csv("./data/LSTM_예측_복구_정상.csv", index=False, encoding="utf-8-sig")
print("✅ 복원 완료: LSTM_예측_복구_정상.csv")








