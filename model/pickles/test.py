import pandas as pd

# ✅ 인코딩을 cp949 또는 euc-kr로 명시
df = pd.read_csv("test_predictions.csv", encoding='cp949')  # 또는 encoding='euc-kr'

# ✅ id 컬럼이 없을 때만 추가
if 'id' not in df.columns:
    df.insert(0, 'id', range(len(df)))

# ✅ 컬럼 이름 변경
if '예측 전기요금(원)' in df.columns:
    df.rename(columns={"예측 전기요금(원)": "prediction"}, inplace=True)

# ✅ 저장
df.to_csv("submission.csv", index=False)
print("✅ 'submission.csv' 파일이 성공적으로 저장되었습니다.")
