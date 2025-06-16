import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 0. 설정
BASE_DIR = '../data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess(base_dir, is_train=True):
    """데이터 로딩 및 전처리"""
    df = pd.read_csv(os.path.join(base_dir, 'train.csv' if is_train else 'test.csv'))

    # 시간 변수 생성
    df["측정일시"] = pd.to_datetime(df["측정일시"])
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["시간"] = df["측정일시"].dt.hour
    df["분"] = df["측정일시"].dt.minute
    df["요일"] = df["측정일시"].dt.weekday
    df["휴일여부"] = df["요일"].isin([5, 6]).astype(int)  # 수정: 토요일(5), 일요일(6)
    df["주말"] = df["휴일여부"]  # 일관성을 위해 추가
    
    # 순환 시간 변수
    df["time_step"] = df["시간"] * 4 + df["분"] // 15
    df["sin_time_step"] = np.sin(2 * np.pi * df["time_step"] / 96)
    df["cos_time_step"] = np.cos(2 * np.pi * df["time_step"] / 96)
    
    # 계절성 추가
    df["sin_month"] = np.sin(2 * np.pi * df["월"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["월"] / 12)
    df["sin_day"] = np.sin(2 * np.pi * df["일"] / 31)
    df["cos_day"] = np.cos(2 * np.pi * df["일"] / 31)

    # 타깃 변수 검증
    if is_train and '전기요금(원)' not in df.columns:
        raise KeyError("학습 데이터에 '전기요금(원)' 컬럼이 없습니다.")
    
    # 계절 및 시간대, 요금단가 계산
    def get_season(month):
        if month in [6, 7, 8]:
            return "여름"
        elif month in [3, 4, 5, 9, 10]:
            return "봄가을"
        else:
            return "겨울"

    def get_time_zone(hour, season):
        if season in ["여름", "봄가을"]:
            if 22 <= hour or hour < 8:
                return "경부하"
            elif (8 <= hour < 11) or (12 <= hour < 13) or (18 <= hour < 22):
                return "중간부하"
            else:
                return "최대부하"
        else:  # 겨울
            if 22 <= hour or hour < 8:
                return "경부하"
            elif (8 <= hour < 9) or (12 <= hour < 16) or (19 <= hour < 22):
                return "중간부하"
            else:
                return "최대부하"

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
    df["계절"] = df["월"].apply(get_season)
    df["적용시점"] = df["측정일시"].apply(lambda x: "before" if x < CUTOFF else "after")
    df["시간대"] = df.apply(lambda r: get_time_zone(r["시간"], r["계절"]), axis=1)
    df["요금단가"] = df.apply(lambda r: RATE_TABLE[r["적용시점"]][r["계절"]][r["시간대"]], axis=1)

    return df

def add_time_features(df, target='전기요금(원)', lags=[1, 24, 168], rolls=[24, 168]):
    """시계열 특성 생성 - 데이터 리키지 방지"""
    df = df.sort_values('측정일시').reset_index(drop=True)
    
    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df[target].shift(lag)
    
    # Rolling features - 최소 기간을 lag보다 크게 설정하여 리키지 방지
    for w in rolls:
        min_periods = max(w // 4, 10)  # 최소 기간을 더 보수적으로 설정
        df[f'roll_mean_{w}'] = df[target].shift(1).rolling(w, min_periods=min_periods).mean()
        df[f'roll_std_{w}'] = df[target].shift(1).rolling(w, min_periods=min_periods).std()
        df[f'roll_max_{w}'] = df[target].shift(1).rolling(w, min_periods=min_periods).max()
        df[f'roll_min_{w}'] = df[target].shift(1).rolling(w, min_periods=min_periods).min()
    
    return df

def evaluate_model(y_true, y_pred, model_name="Model"):
    """모델 평가 메트릭"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return mae, rmse, r2

# 1. 데이터 로딩 및 전처리
print("데이터 로딩 중...")
train_df = load_and_preprocess(BASE_DIR, True)
test_df = load_and_preprocess(BASE_DIR, False)

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# 2. 작업유형 인코딩 (필요한 경우)
if '작업유형' in train_df.columns:
    le_work = LabelEncoder()
    train_df['작업유형_encoded'] = le_work.fit_transform(train_df['작업유형'])
    test_df['작업유형_encoded'] = le_work.transform(test_df['작업유형'])
    joblib.dump(le_work, os.path.join(MODEL_DIR, 'work_type_encoder.pkl'))

# 3. 시계열 특성 생성
print("시계열 특성 생성 중...")
train_df = add_time_features(train_df)

# 테스트 데이터의 lag 특성 처리 개선
if len(test_df) > 0:
    print("테스트 데이터 lag 특성 처리 중...")
    # 마지막 train 데이터를 기준으로 테스트 데이터의 lag 계산
    last_train_values = train_df['전기요금(원)'].tail(168).values  # 최대 lag만큼
    
    # 테스트 데이터에 임시로 타겟값 추가 (lag 계산용)
    test_df['전기요금(원)'] = np.nan
    
    # 전체 데이터 결합
    full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('측정일시')
    
    # lag 특성만 다시 계산
    for lag in [1, 24, 168]:
        full_df[f'lag_{lag}'] = full_df['전기요금(원)'].shift(lag)
    
    # rolling 특성도 다시 계산
    for w in [24, 168]:
        min_periods = max(w // 4, 10)
        full_df[f'roll_mean_{w}'] = full_df['전기요금(원)'].shift(1).rolling(w, min_periods=min_periods).mean()
        full_df[f'roll_std_{w}'] = full_df['전기요금(원)'].shift(1).rolling(w, min_periods=min_periods).std()
        full_df[f'roll_max_{w}'] = full_df['전기요금(원)'].shift(1).rolling(w, min_periods=min_periods).max()
        full_df[f'roll_min_{w}'] = full_df['전기요금(원)'].shift(1).rolling(w, min_periods=min_periods).min()
    
    # 테스트 데이터만 추출
    test_df = full_df[full_df['id'].isin(test_df['id'])].copy()
    test_df = test_df.drop('전기요금(원)', axis=1)  # 임시 컬럼 제거

# 4. 라벨 인코딩
le = LabelEncoder()
train_df['요일_le'] = le.fit_transform(train_df['요일'])
test_df['요일_le'] = le.transform(test_df['요일'])
joblib.dump(le, os.path.join(MODEL_DIR, 'weekday_encoder.pkl'))

# 5. 결측치 및 이상치 처리
print(f"결측치 제거 전 train 데이터: {len(train_df)}")
train_df = train_df.dropna()
print(f"결측치 제거 후 train 데이터: {len(train_df)}")

# 이상치 제거 (상위/하위 1% 제거)
q1 = train_df['전기요금(원)'].quantile(0.01)
q99 = train_df['전기요금(원)'].quantile(0.99)
print(f"이상치 제거 범위: {q1:.2f} ~ {q99:.2f}")
train_df = train_df[(train_df['전기요금(원)'] >= q1) & (train_df['전기요금(원)'] <= q99)]
print(f"이상치 제거 후 train 데이터: {len(train_df)}")

# 6. 피처 선택
BASE_FEATURES = ['월', '일', '시간', '주말', '요일_le', '요금단가']
TIME_FEATURES = ['sin_time_step', 'cos_time_step', 'sin_month', 'cos_month', 'sin_day', 'cos_day']
LAG_FEATURES = ['lag_1', 'lag_24', 'lag_168']
ROLL_FEATURES = ['roll_mean_24', 'roll_std_24', 'roll_max_24', 'roll_min_24',
                 'roll_mean_168', 'roll_std_168', 'roll_max_168', 'roll_min_168']

# 작업유형이 있는 경우 추가
if '작업유형_encoded' in train_df.columns:
    BASE_FEATURES.append('작업유형_encoded')

FEATURES = BASE_FEATURES + TIME_FEATURES + LAG_FEATURES + ROLL_FEATURES
print(f"사용할 피처 수: {len(FEATURES)}")
print("피처 목록:", FEATURES)

TARGET = '전기요금(원)'

# 테스트 데이터의 결측치 처리
test_df[FEATURES] = test_df[FEATURES].fillna(test_df[FEATURES].median())

# 7. 스케일링
scaler = RobustScaler()
X_train = scaler.fit_transform(train_df[FEATURES])
y_train = np.log1p(train_df[TARGET].values)  # 로그 변환
X_test = scaler.transform(test_df[FEATURES])

joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

# 8. 모델 정의 및 하이퍼파라미터 조정
models = {
    'lgb': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    ),
    'xgb': XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ),
    'rf': RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# 9. TimeSeriesSplit Cross-validation with proper gap
print("\n=== Cross Validation 시작 ===")
# 시계열 데이터에서 gap을 두어 데이터 리키지 방지
tscv = TimeSeriesSplit(n_splits=5, gap=24)  # 24시간 gap 추가
oof_preds = {name: np.zeros(len(y_train)) for name in models}
cv_scores = {name: [] for name in models}

# 결측치 처리 - CV 전에 수행
print(f"CV 전 결측치 확인: {pd.DataFrame(X_train).isna().sum().sum()}")
if pd.DataFrame(X_train).isna().sum().sum() > 0:
    print("결측치 발견 - 중앙값으로 대체")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.pkl'))

for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
    print(f"\n--- Fold {fold + 1} ---")
    print(f"Train 기간: {tr_idx[0]} ~ {tr_idx[-1]} ({len(tr_idx)} samples)")
    print(f"Valid 기간: {va_idx[0]} ~ {va_idx[-1]} ({len(va_idx)} samples)")
    
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    
    for name, model in models.items():
        # 모델 훈련
        if name == 'lgb':
            model.fit(X_tr, y_tr, 
                     eval_set=[(X_va, y_va)], 
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        elif name == 'xgb':
            # XGBoost 최신 버전 호환성
            try:
                # 최신 버전 방식
                model.fit(X_tr, y_tr,
                         eval_set=[(X_va, y_va)],
                         callbacks=[xgb.callback.EarlyStopping(rounds=100)])
            except:
                # 구버전 호환성
                try:
                    model.fit(X_tr, y_tr,
                             eval_set=[(X_va, y_va)],
                             early_stopping_rounds=100,
                             verbose=False)
                except:
                    # 기본 훈련
                    model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr)
        
        # 예측
        va_pred = model.predict(X_va)
        oof_preds[name][va_idx] = va_pred
        
        # 평가 (원래 스케일로 변환)
        mae, rmse, r2 = evaluate_model(
            np.expm1(y_va), np.expm1(va_pred), f"{name.upper()} Fold {fold+1}"
        )
        cv_scores[name].append(mae)

# 10. 최종 OOF 평가
print("\n=== 최종 CV 결과 ===")
for name in models:
    avg_mae = np.mean(cv_scores[name])
    std_mae = np.std(cv_scores[name])
    print(f"{name.upper()} - 평균 MAE: {avg_mae:.4f} (±{std_mae:.4f})")
    
    # 전체 OOF 평가
    evaluate_model(
        np.expm1(y_train), np.expm1(oof_preds[name]), f"{name.upper()} OOF"
    )

# 11. 앙상블
oof_stack = np.column_stack([oof_preds[name] for name in models])
final_oof = np.mean(oof_stack, axis=1)

print("\n=== 앙상블 결과 ===")
evaluate_model(np.expm1(y_train), np.expm1(final_oof), "Ensemble OOF")

# 12. 최종 모델 훈련 및 테스트 예측
print("\n=== 최종 예측 생성 ===")
test_preds = []

for name, model in models.items():
    print(f"{name.upper()} 최종 모델 훈련 중...")
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_preds.append(test_pred)
    
    # 모델 저장
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_final.pkl"))
    print(f"{name.upper()} 모델 저장 완료")

# 앙상블 예측
test_preds_stack = np.column_stack(test_preds)
final_test_pred = np.mean(test_preds_stack, axis=1)

# 13. Submission 파일 생성
submission = pd.DataFrame({
    'id': test_df['id'],
    '전기요금(원)': np.expm1(final_test_pred)
})

submission.to_csv('submission.csv', index=False)
print(f"\n✅ submission.csv 저장 완료!")
print(f"예측값 범위: {submission['전기요금(원)'].min():.2f} ~ {submission['전기요금(원)'].max():.2f}")
print(f"예측값 평균: {submission['전기요금(원)'].mean():.2f}")

# 14. 피처 중요도 출력 (LightGBM 기준)
if 'lgb' in models:
    feature_importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': models['lgb'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== 상위 10개 중요 피처 ===")
    print(feature_importance.head(10).to_string(index=False))