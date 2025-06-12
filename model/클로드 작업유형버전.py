
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor  # 필요시 주석 해제
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("📊 전기요금 예측 모델 - 성능 개선 버전")
print("=" * 50)

# 📂 경로 설정
BASE_DIR = "./data"
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")

# ================================
# 📊 1. 데이터 불러오기
# ================================
print("1️⃣ 데이터 로딩 중...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"   - 훈련 데이터: {train_df.shape}")
print(f"   - 테스트 데이터: {test_df.shape}")

# ================================
# 🕒 2. datetime 파싱 + 기본 시계열 파생변수
# ================================
print("2️⃣ 시계열 피처 생성 중...")
for df in [train_df, test_df]:
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['월'] = df['측정일시'].dt.month
    df['일'] = df['측정일시'].dt.day
    df['시간'] = df['측정일시'].dt.hour
    df['요일'] = df['측정일시'].dt.weekday  # 0=월요일, 6=일요일
    
    # 🏭 공장 특성 반영: 일요일(6), 월요일(0)에 전기 덜 사용
    df['공장휴무일'] = ((df['요일'] == 0) | (df['요일'] == 6)).astype(int)
    df['평일여부'] = (df['요일'].between(1, 5)).astype(int)  # 화~토요일
    
    # 📅 날짜 관련 피처
    df['월말여부'] = (df['일'] >= 28).astype(int)
    df['월초여부'] = (df['일'] <= 3).astype(int)
    df['월중순여부'] = (df['일'].between(11, 20)).astype(int)
    
    # 🌀 주기성 피처 (sin/cos 변환)
    df['sin_시간'] = np.sin(2 * np.pi * df['시간'] / 24)
    df['cos_시간'] = np.cos(2 * np.pi * df['시간'] / 24)
    df['sin_월'] = np.sin(2 * np.pi * df['월'] / 12)
    df['cos_월'] = np.cos(2 * np.pi * df['월'] / 12)
    df['sin_일'] = np.sin(2 * np.pi * df['일'] / 31)
    df['cos_일'] = np.cos(2 * np.pi * df['일'] / 31)
    df['sin_요일'] = np.sin(2 * np.pi * df['요일'] / 7)
    df['cos_요일'] = np.cos(2 * np.pi * df['요일'] / 7)

# ================================
# 🌞 3. 계절/시간대 구분 + 요금단가 계산
# ================================
print("3️⃣ 계절별 요금 체계 및 작업유형별 부하 매핑 중...")

def get_season(month):
    """계절 구분 함수"""
    if month in [6, 7, 8]:
        return '여름'
    elif month in [3, 4, 5, 9, 10]:
        return '봄가을'
    else:
        return '겨울'

def map_work_type_to_load(work_type):
    """작업유형을 부하 구분으로 매핑하는 함수"""
    load_mapping = {
        'Light_Load': '경부하',
        'Medium_Load': '중간부하', 
        'Maximum_Load': '최대부하'
    }
    return load_mapping.get(work_type, '경부하')  # 기본값은 경부하

# 📊 요금 단가 테이블
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
    # 작업유형으로부터 시간대(부하구분) 매핑
    df['시간대'] = df['작업유형'].apply(map_work_type_to_load)
    df['요금단가'] = df.apply(lambda row: rate_table[row['적용시점']][row['계절']][row['시간대']], axis=1)

# ================================
# 🔤 4. 작업유형 인코딩 + Target Encoding
# ================================
print("4️⃣ 작업유형 인코딩 및 Target Encoding 중...")

# 기본 라벨 인코딩
le = LabelEncoder()
train_df['작업유형_encoded'] = le.fit_transform(train_df['작업유형'])
test_df['작업유형_encoded'] = le.transform(test_df['작업유형'])

# 🎯 Target Encoding (과적합 방지를 위한 Smoothing 적용)
def target_encoding_with_smoothing(df_train, df_test, categorical_col, target_col, smoothing=10):
    """Target Encoding with Smoothing"""
    # 전체 평균
    global_mean = df_train[target_col].mean()
    
    # 그룹별 통계
    group_stats = df_train.groupby(categorical_col)[target_col].agg(['mean', 'count']).reset_index()
    group_stats.columns = [categorical_col, 'target_mean', 'count']
    
    # Smoothing 적용
    group_stats['smoothed_target'] = (
        (group_stats['target_mean'] * group_stats['count'] + global_mean * smoothing) /
        (group_stats['count'] + smoothing)
    )
    
    # 매핑
    mapping = dict(zip(group_stats[categorical_col], group_stats['smoothed_target']))
    
    return mapping

# 다양한 Target Encoding 적용
target_encodings = {}

# 작업유형별 Target Encoding
target_encodings['작업유형'] = target_encoding_with_smoothing(train_df, test_df, '작업유형', '전기요금(원)')
train_df['작업유형_target'] = train_df['작업유형'].map(target_encodings['작업유형'])
test_df['작업유형_target'] = test_df['작업유형'].map(target_encodings['작업유형'])

# 시간별 Target Encoding
target_encodings['시간'] = target_encoding_with_smoothing(train_df, test_df, '시간', '전기요금(원)')
train_df['시간_target'] = train_df['시간'].map(target_encodings['시간'])
test_df['시간_target'] = test_df['시간'].map(target_encodings['시간'])

# 요일별 Target Encoding
target_encodings['요일'] = target_encoding_with_smoothing(train_df, test_df, '요일', '전기요금(원)')
train_df['요일_target'] = train_df['요일'].map(target_encodings['요일'])
test_df['요일_target'] = test_df['요일'].map(target_encodings['요일'])

# 시간대별 Target Encoding
target_encodings['시간대'] = target_encoding_with_smoothing(train_df, test_df, '시간대', '전기요금(원)')
train_df['시간대_target'] = train_df['시간대'].map(target_encodings['시간대'])
test_df['시간대_target'] = test_df['시간대'].map(target_encodings['시간대'])

# ================================
# 🔄 5. 상호작용 피처 생성
# ================================
print("5️⃣ 상호작용 피처 생성 중...")
# LabelEncoder 인코딩
le_season = LabelEncoder()
train_df['계절_encoded'] = le_season.fit_transform(train_df['계절'])
test_df['계절_encoded'] = le_season.transform(test_df['계절'])

# 상호작용 피처 생성
for df in [train_df, test_df]:
    df['부하구분_encoded'] = df['작업유형'].map({
        'Light_Load': 0, 'Medium_Load': 1, 'Maximum_Load': 2
    })
    df['요금단가_작업유형'] = df['요금단가'] * df['작업유형_encoded']
    df['요금단가_시간'] = df['요금단가'] * df['시간']
    df['부하구분_시간'] = df['부하구분_encoded'] * df['시간']
    df['부하구분_공장휴무일'] = df['부하구분_encoded'] * df['공장휴무일']
    df['계절_부하구분'] = df['계절_encoded'] * df['부하구분_encoded']

# ================================
# 🧹 6. 이상치 처리
# ================================
print("6️⃣ 이상치 처리 중...")

# IQR 방법으로 이상치 탐지
Q1 = train_df['전기요금(원)'].quantile(0.25)
Q3 = train_df['전기요금(원)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"   - 이상치 범위: {lower_bound:.2f} ~ {upper_bound:.2f}")
outliers_count = len(train_df[(train_df['전기요금(원)'] < lower_bound) | (train_df['전기요금(원)'] > upper_bound)])
print(f"   - 이상치 개수: {outliers_count}")

# 이상치 제거 (너무 극단적인 값만 제거)
train_df_clean = train_df[(train_df['전기요금(원)'] >= lower_bound) & (train_df['전기요금(원)'] <= upper_bound)].copy()
print(f"   - 정제 후 훈련 데이터: {train_df_clean.shape[0]} (제거: {len(train_df) - len(train_df_clean)}개)")

# ================================
# 🎯 7. 피처 선택 및 데이터 준비
# ================================
print("7️⃣ 피처 선택 및 데이터 준비 중...")

# 사용할 피처 정의
features = [
    # 기본 피처
    '작업유형_encoded', '부하구분_encoded', '월', '일', '시간', '요일',
    
    # 공장 특성 피처
    '공장휴무일', '평일여부',
    
    # 날짜 관련 피처
    '월말여부', '월초여부', '월중순여부',
    
    # 주기성 피처
    'sin_시간', 'cos_시간', 'sin_월', 'cos_월', 'sin_요일', 'cos_요일',
    
    # 요금 관련 피처
    '요금단가', '계절_encoded',
    
    # Target Encoding 피처
    '작업유형_target', '시간_target', '요일_target', '시간대_target',
    
    # 상호작용 피처
    '요금단가_작업유형', '요금단가_시간', 
    '공장휴무일_작업유형', '공장휴무일_시간',
    '부하구분_시간', '부하구분_공장휴무일', '계절_부하구분',
    
    # 시간대 피처
    '오전시간', '오후시간', '저녁시간', '새벽시간'
]

target = '전기요금(원)'

# 피처 존재 확인
missing_features = [f for f in features if f not in train_df_clean.columns]
if missing_features:
    print(f"   ⚠️  누락된 피처: {missing_features}")
    features = [f for f in features if f in train_df_clean.columns]

print(f"   - 사용할 피처 개수: {len(features)}")

# 데이터 분할
X = train_df_clean[features]
y = train_df_clean[target]
X_test = test_df[features]

# 스케일링 (RobustScaler 사용 - 이상치에 덜 민감)
print("8️⃣ 피처 스케일링 중...")
scaler = RobustScaler()

# 연속형 변수만 스케일링
numeric_features = ['요금단가', '작업유형_target', '시간_target', '요일_target', '시간대_target']
numeric_features = [f for f in numeric_features if f in features]

if numeric_features:
    X_scaled = X.copy()
    X_test_scaled = X_test.copy()
    
    X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
else:
    X_scaled = X
    X_test_scaled = X_test

# 훈련/검증 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"   - 훈련 세트: {X_train.shape}")
print(f"   - 검증 세트: {X_val.shape}")

# ================================
# 🤖 8. 모델 훈련 (앙상블)
# ================================
print("9️⃣ 모델 훈련 중...")

# 여러 모델 정의
models = {
    'xgb': XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'lgb': LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    'rf': RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

# 모델별 훈련 및 검증
model_predictions = {}
model_scores = {}

for name, model in models.items():
    print(f"   - {name.upper()} 모델 훈련 중...")
    
    # 모델 훈련
    model.fit(X_train, y_train)
    
    # 검증 예측
    val_pred = model.predict(X_val)
    
    # 성능 평가
    mae = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2 = r2_score(y_val, val_pred)
    
    model_scores[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    # 테스트 예측
    test_pred = model.predict(X_test_scaled)
    model_predictions[name] = test_pred
    
    print(f"     MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# ================================
# 🎯 9. 앙상블 예측
# ================================
print("🔟 앙상블 예측 중...")

# 성능 기반 가중평균 (R2 스코어 기반)
weights = {}
total_r2 = sum([scores['R2'] for scores in model_scores.values()])

for name, scores in model_scores.items():
    weights[name] = scores['R2'] / total_r2

print("   - 모델별 가중치:")
for name, weight in weights.items():
    print(f"     {name.upper()}: {weight:.3f}")

# 가중평균 앙상블
ensemble_pred = np.zeros(len(X_test_scaled))
for name, pred in model_predictions.items():
    ensemble_pred += weights[name] * pred

# 앙상블 검증 성능 계산
ensemble_val_pred = np.zeros(len(X_val))
for name, model in models.items():
    val_pred = model.predict(X_val)
    ensemble_val_pred += weights[name] * val_pred

ensemble_mae = mean_absolute_error(y_val, ensemble_val_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
ensemble_r2 = r2_score(y_val, ensemble_val_pred)

print("\n📊 최종 앙상블 성능:")
print(f"   MAE: {ensemble_mae:.2f}")
print(f"   RMSE: {ensemble_rmse:.2f}")
print(f"   R²: {ensemble_r2:.4f}")

# ================================
# 💾 10. 결과 저장
# ================================
print("1️⃣1️⃣ 결과 저장 중...")

# 제출 파일 생성
test_df['전기요금(원)'] = ensemble_pred
submission = test_df[['id', '전기요금(원)']].copy()

# 음수 값 처리 (전기요금은 음수가 될 수 없음)
submission['전기요금(원)'] = np.maximum(submission['전기요금(원)'], 0)

# 파일 저장
submission.to_csv("submission_improved.csv", index=False)

print(f"   - 제출 파일 저장: submission_improved.csv")
print(f"   - 예측값 범위: {submission['전기요금(원)'].min():.2f} ~ {submission['전기요금(원)'].max():.2f}")

# ================================
# 📈 11. 피처 중요도 분석
# ================================
print("1️⃣2️⃣ 피처 중요도 분석...")

# XGBoost 모델의 피처 중요도
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': models['xgb'].feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 상위 10개 중요 피처:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"   {i+1:2d}. {row['feature']:20s}: {row['importance']:.4f}")

print("\n✅ 모든 처리가 완료되었습니다!")
print("📁 'submission_improved.csv' 파일을 확인하세요.")