from shiny import App, render, ui, reactive
from shinywidgets import render_widget, output_widget
import os
import asyncio
import tempfile
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import FigureWidget
from plotly.subplots import make_subplots
from pandas.tseries.offsets import Week
from docxtpl import RichText
from generate_report import generate_report
# 경고 무시
warnings.filterwarnings("ignore")

# asyncio의 CancelledError 무시하는 방식
def silence_cancelled_error():
    def handle_exception(loop, context):
        if isinstance(context.get("exception"), asyncio.CancelledError):
            return  # 무시
        else:
            loop.default_exception_handler(context)

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)

# main entrypoint에서 실행
silence_cancelled_error()

# ───────────────────────────────────────────────────────
# 1) 경로 설정
# ───────────────────────────────────────────────────────
# app.py가 위치한 폴더를 기준으로 상대 경로 설정
BASE_DIR = Path(__file__).resolve().parent       # 👉 dashboard/
DATA_DIR = BASE_DIR / "data"                     # 👉 dashboard/data/

DF_FINAL = DATA_DIR / "df_final.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test_predicted_december_data.csv"
SUMMARY_CSV = DATA_DIR / "monthly_summary.csv"

print("📂 BASE_DIR:", BASE_DIR)
print("📂 DATA_DIR:", DATA_DIR)
print("📄 DF_FINAL:", DF_FINAL)

# 파일 존재 여부 검증 (선택적으로 사용 가능)
for path in [TRAIN_CSV, TEST_CSV, DF_FINAL]:
    if not path.exists():
        print(f"❌ 파일 없음: {path}")
    else:
        print(f"✅ 파일 확인됨: {path}")

# ───────────────────────────────────────────────────────
# 2) 데이터 로드 함수
# ───────────────────────────────────────────────────────
def load_data():
    try:
        df = pd.read_csv(TRAIN_CSV)
        # 날짜 컬럼 변환
        if "측정일시" in df.columns:
            df["측정일시"] = pd.to_datetime(df["측정일시"])
        elif "datetime" in df.columns:
            df["측정일시"] = pd.to_datetime(df["datetime"])
        # 컬럼명 통일
        rename_map = {}
        if "전력사용량(kWh)" in df.columns:
            rename_map["전력사용량(kWh)"] = "전력사용량"
        if "power_usage" in df.columns:
            rename_map["power_usage"] = "전력사용량"
        if "전기요금(원)" in df.columns:
            rename_map["전기요금(원)"] = "전기요금"
        if "cost" in df.columns:
            rename_map["cost"] = "전기요금"
        if "탄소배출량(tCO2)" in df.columns:
            rename_map["탄소배출량(tCO2)"] = "탄소배출량"
        if "co2" in df.columns:
            rename_map["co2"] = "탄소배출량"
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        # 작업유형 없는 경우 랜덤 생성
        if "작업유형" not in df.columns:
            df["작업유형"] = np.random.choice(["Light_Load", "Heavy_Load"], len(df))
        return df
    except FileNotFoundError:
        # 더미 데이터
        dates = pd.date_range(start="2025-05-01", end="2025-06-30", freq="H")
        return pd.DataFrame({
            "측정일시": dates,
            "전력사용량": np.random.normal(341203, 50000, len(dates)),
            "전기요금":     np.random.normal(120327, 20000, len(dates)),
            "탄소배출량":   np.random.normal(328, 30, len(dates)),
            "작업유형":     np.random.choice(["Light_Load", "Heavy_Load"], len(dates))
        })

# ───────────────────────────────────────────────────────
# 3) 글로벌 데이터프레임
# ───────────────────────────────────────────────────────
df_train  = load_data()
test_df   = pd.read_csv(TEST_CSV)
test_df["측정일시"] = pd.to_datetime(test_df["측정일시"])
# test_df 컬럼명 통일
for orig, std in [("전력사용량(kWh)","전력사용량"),("전기요금(원)","전기요금"),("탄소배출량(tCO2)","탄소배출량")]:
    if orig in test_df.columns:
        test_df.rename(columns={orig: std}, inplace=True)

# final_df 로드
final_df = pd.read_csv(DF_FINAL)
final_df["측정일시"] = pd.to_datetime(final_df["측정일시"], errors="coerce")




# ───────────────────────────────────────────────────────
# 스트리머 / 누적기 정의 (반드시 server() 위에 위치!)
# ───────────────────────────────────────────────────────
class Streamer:
    def __init__(self, df):
        # df를 시간순으로 정렬하고 내부 인덱스 초기화
        self.df = df.sort_values("측정일시").reset_index(drop=True)
        self.index = 0

    def get_next_batch(self, n=1):
        if self.index >= len(self.df):
            return None
        batch = self.df.iloc[self.index : self.index + n]
        self.index += n
        return batch

    def get_current_data(self):
        return self.df.iloc[: self.index].copy()


class Accumulator:
    def __init__(self):
        self.df = pd.DataFrame()

    def accumulate(self, batch):
        self.df = pd.concat([self.df, batch], ignore_index=True)

    def get(self):
        return self.df.copy()


# ───────────────────────────────────────────────────────
# 4) Baseline 계산 함수
# ───────────────────────────────────────────────────────
def get_november_baseline(df):
    nov = df[(df["측정일시"] >= "2024-11-01") & (df["측정일시"] < "2024-12-01")].copy()
    daily_total   = nov.groupby(nov["측정일시"].dt.date)["전력사용량"].sum().mean()
    weekly_total  = nov.groupby(nov["측정일시"].dt.to_period("W"))["전력사용량"].sum().mean()
    monthly_total = nov["전력사용량"].sum()
    cost_daily    = nov.groupby(nov["측정일시"].dt.date)["전기요금"].sum().mean()
    cost_weekly   = nov.groupby(nov["측정일시"].dt.to_period("W"))["전기요금"].sum().mean()
    cost_monthly  = nov["전기요금"].sum()
    return {
        "power": {"daily": daily_total, "weekly": weekly_total, "monthly": monthly_total},
        "cost":  {"daily": cost_daily,   "weekly": cost_weekly,   "monthly": cost_monthly}
    }

nov_baseline = get_november_baseline(df_train)





# ───────────────────────────────────────────────────────
# Chart helper functions (붙여넣기만 하면 동작)
# ───────────────────────────────────────────────────────

def make_work_type_pie(df):
    import plotly.express as px

    if df.empty or "작업유형" not in df:
        fig = px.pie(title="작업유형별 분포 (데이터 없음)")
    else:
        cnt = df["작업유형"].value_counts()
        fig = px.pie(
        names=cnt.index,
        values=cnt.values,
        color_discrete_map={
            "Light_Load": "#90ee90",    # 연파랑
            "Medium_Load": "#87cefa",   # 초록
            "Maximum_Load": "#ef4444"   # 빨강
        },
        width=600,
        height=600
)

    # 범례를 아래로 깔아서 파이 자체가 차지하는 영역을 최대화
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,      # 범례를 차트 아래로 내림
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=10, r=10, t=40, b=10)  # 여백 최소화
    )

    return fig



def make_cost_trend_chart(df, agg_unit):
    import plotly.graph_objects as go
    import pandas as pd

    if df is None or df.empty:
        return go.Figure()

    # --- 원본 렌더러의 전처리 & 집계 로직 ---
    date_col = next((c for c in df.columns if "일시" in c or "시간" in c), None)
    cost_col = next((c for c in df.columns if "요금" in c or "비용" in c), None)
    if not date_col or not cost_col:
        return go.Figure()
    d = df[[date_col, cost_col]].copy()
    d.columns = ["datetime", "cost"]
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d["cost"]     = pd.to_numeric(d["cost"], errors="coerce")
    d = d.dropna().sort_values("datetime")

    # aggregation
    if agg_unit == "hour":
        d["period"] = d["datetime"].dt.floor("H")
        d["label"]  = d["period"].dt.strftime("%m/%d %H:%M")
    elif agg_unit == "day":
        d["period"] = d["datetime"].dt.date
        d["label"]  = pd.to_datetime(d["period"]).dt.strftime("%m/%d")
    else:  # weekday
        d["weekday_num"] = d["datetime"].dt.weekday
        d["period"]      = d["weekday_num"]
        wdmap = {i: w for i, w in enumerate(
            ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
        )}
        d["label"] = d["weekday_num"].map(wdmap)

    if agg_unit == "weekday":
        agg = (
            d.groupby(["weekday_num","label"])["cost"]
             .agg(["sum","mean"])
             .reset_index()
             .sort_values("weekday_num")
        )
        agg.columns = ["weekday_num","label","total","average"]
    else:
        agg = (
            d.groupby("label")["cost"]
             .agg(["sum","mean"])
             .reset_index()
        )
        agg.columns = ["label","total","average"]

    # --- figure 생성 (원본 레이아웃 그대로) ---
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["label"], y=agg["total"],
        name="누적 요금",
        marker=dict(color="#2563eb", line=dict(color="darkblue", width=0.5)),
        opacity=0.8,
        hovertemplate="<b>%{x}</b><br>누적: %{y:,.0f}원<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=agg["label"], y=agg["average"],
        mode="lines+markers",
        name="평균 요금",
        line=dict(color="black", width=3),
        marker=dict(color="black", size=6),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>평균: %{y:,.0f}원<extra></extra>"
    ))
    fig.update_layout(
        xaxis=dict(type="category", tickangle=-45, showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="누적 요금 (원)", showgrid=True, gridcolor="lightgray"),
        yaxis2=dict(title="평균 요금 (원)", overlaying="y", side="right", showgrid=False),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        margin=dict(l=60,r=60,t=80,b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        title=dict(text="<b>[B] 전력 요금 시계열 분석</b>", x=0.5)
    )
    # annotation
    total, avg, mx, cnt = agg["total"].sum(), agg["average"].mean(), agg["total"].max(), len(d)
    fig.add_annotation(
        text=(
            f" 요약 통계<br>"
            f"총 요금: {total:,.0f}원<br>"
            f"평균: {avg:,.0f}원<br>"
            f"최대: {mx:,.0f}원<br>"
            f"데이터: {cnt}개"
        ),
        xref="paper", yref="paper", x=0.02, y=0.98,
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1,
        font=dict(size=10)
    )
    return fig






# ───────────────────────────────────────────────────────
# 헬퍼 함수: 월별 전력사용량 누적 + 평균 전기요금 차트
# ───────────────────────────────────────────────────────
def make_monthly_summary_chart(df_full, sel_month: str):
    import pandas as pd
    import plotly.graph_objects as go

    # 0) 입력 검사: None 또는 빈프레임이면 바로 빈 Figure 반환
    if df_full is None or df_full.empty:
        return go.Figure()

    # 1) 복사 & 날짜 형식 변환
    df = df_full.copy()
    if "측정일시" not in df.columns:
        # 필요한 컬럼이 없으면 빈 Figure
        return go.Figure()
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce").dropna()

    # 2) 전력·요금 컬럼 안전 탐색
    power_col = next((c for c in df.columns if "전력사용량" in c), None)
    cost_col  = next((c for c in df.columns if "전기요금"   in c), None)
    if power_col is None or cost_col is None:
        # 컬럼을 못 찾으면 빈 Figure
        return go.Figure()

    # 3) 최근 1년 필터
    latest = df["측정일시"].max()
    one_year_ago = latest - pd.DateOffset(years=1)
    df = df[(df["측정일시"] >= one_year_ago) & (df["측정일시"] <= latest)]
    if df.empty:
        return go.Figure()

    # 4) 월별 집계: '측정월' 컬럼 생성
    df["측정월"] = df["측정일시"].dt.to_period("M").dt.to_timestamp()
    agg = (
        df.groupby("측정월")[[power_col, cost_col]]
          .agg({power_col: "sum", cost_col: "mean"})
          .reset_index()
    )
    agg["측정월_라벨"] = agg["측정월"].dt.strftime("%Y-%m")
    agg["color"] = agg["측정월_라벨"].apply(lambda x: "red" if x == sel_month else "gray")

    # 5) Plotly 그리기
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["측정월_라벨"],
        y=agg[power_col],
        name="월별 전력사용량",
        marker_color=agg["color"],
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=agg["측정월_라벨"],
        y=agg[cost_col],
        name="월별 평균요금",
        mode="lines+markers",
        yaxis="y2"
    ))
    fig.update_layout(
        title="최근 1년 월별 전력사용량 및 평균요금",
        xaxis=dict(title="월"),
        yaxis=dict(title="전력사용량 (kWh)", side="left"),
        yaxis2=dict(title="평균요금 (원)", side="right", overlaying="y"),
        height=350,
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=40, l=40, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    return fig



# ✅ 컬럼명 일괄 매핑
if "전력사용량(kWh)" in test_df.columns:
    test_df["전력사용량"] = test_df["전력사용량(kWh)"]
if "전기요금(원)" in test_df.columns:
    test_df["전기요금"] = test_df["전기요금(원)"]
if "탄소배출량(tCO2)" in test_df.columns:
    test_df["탄소배출량"] = test_df["탄소배출량(tCO2)"]



# 공통: 컬럼 자동 탐색
def _find_col(df, patterns):
    """
    df 안에서 patterns 리스트 내 키워드가 들어간 첫 번째 컬럼명을 반환.
    없으면 None.
    """
    for pat in patterns:
        for col in df.columns:
            if pat in col:
                return col
    return None

# 1) 실제 데이터로부터 필요한 값을 계산
def get_peak_cost_info(df):
    cost_col = _find_col(df, ["전기요금", "cost"])
    date_col = _find_col(df, ["측정일시", "datetime"])
    if cost_col is None or date_col is None:
        return "데이터 없음"
    peak_idx  = df[cost_col].idxmax()
    peak_row  = df.loc[peak_idx]
    peak_cost = peak_row[cost_col]
    peak_date = peak_row[date_col]
    return f"₩{peak_cost:,.0f} (발생일시: {peak_date:%Y-%m-%d %H:%M})"

def get_avg_carbon_info(df):
    carbon_col = _find_col(df, ["탄소배출량", "co2"])
    if carbon_col is None:
        return "데이터 없음"
    avg_carbon = df[carbon_col].mean()
    return f"{avg_carbon:.3f} tCO₂"

def get_main_work_type_info(df):
    if "작업유형" not in df.columns or df["작업유형"].empty:
        return "데이터 없음"
    return df["작업유형"].mode().iloc[0]

def get_monthly_change_info(df):
    # 컬럼 탐색
    cost_col = _find_col(df, ["전기요금", "cost"])
    date_col = _find_col(df, ["측정일시", "datetime"])
    if cost_col is None or date_col is None or df.empty:
        return "\n데이터 없음"

    # (1) 현재 합계 (선택 월 데이터)
    cur_sum    = df[cost_col].sum()
    # (2) 기준일: 선택 월 중 최소 시각
    min_date   = df[date_col].min()
    prev_cutoff= min_date - timedelta(days=30)

    # (3) 전체 final_df 에서 과거 30일치 필터
    prev_df    = final_df[
        (final_df[date_col] >= prev_cutoff) &
        (final_df[date_col] < min_date)
    ]
    prev_sum   = prev_df[cost_col].sum() or 0.0

    # (4) 증감률 계산
    rate = (cur_sum - prev_sum) / prev_sum * 100 if prev_sum else 0.0

    # (5) 반환 포맷
    return f"{rate:+.1f}%"



def make_comparison_chart(df_full, selected_month: str, metric: str = "usage"):
    """
    전월 / 선택월 / 연간 평균 비교 차트.
    - df_full: 전체 데이터프레임 (측정일시, 전력사용량*, 전기요금* 컬럼 포함)
    - selected_month: "YYYY-MM" 형식
    - metric: "usage" or "cost"
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    # 1) 날짜 컬럼, 메트릭 컬럼 탐색
    date_col  = next((c for c in df_full.columns if "측정일시" in c or "datetime" in c), None)
    usage_col = next((c for c in df_full.columns if "전력사용량" in c), None)
    cost_col  = next((c for c in df_full.columns if "전기요금"   in c or "cost" in c.lower()), None)
    if not date_col or not usage_col or not cost_col:
        return go.Figure()

    df = df_full.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # 2) 선택월 / 전월 Period 계산
    sel_start = pd.to_datetime(f"{selected_month}-01")
    prev_start = sel_start - pd.DateOffset(months=1)
    prev_end   = sel_start - pd.DateOffset(days=1)

    # 3) 값 집계 함수
    def month_sum(df_, start):
        period = pd.Period(start, freq="M")
        sel = df_[df_[date_col].dt.to_period("M") == period]
        if metric == "usage":
            return sel[usage_col].sum()
        else:
            return sel[cost_col].sum()

    val_prev = month_sum(df, prev_start)
    val_sel  = month_sum(df, sel_start)

    # 4) 연간 평균: 1년 전부터 오늘까지 월별 합계/평균
    latest = df[date_col].max()
    one_year_ago = latest - pd.DateOffset(years=1)
    df_year = df[(df[date_col] >= one_year_ago) & (df[date_col] <= latest)]
    if metric == "usage":
        # 월별 합계 → 평균
        year_vals = df_year.set_index(date_col)[usage_col].resample("M").sum()
    else:
        year_vals = df_year.set_index(date_col)[cost_col].resample("M").sum()
    val_year_avg = year_vals.mean() if not year_vals.empty else 0

    # 5) 막대 데이터 준비
    labels = ["전월", "선택월", "연간 평균"]
    values = [val_prev, val_sel, val_year_avg]
    colors = ["lightblue", "red", "darkblue"]
    unit   = "kWh" if metric=="usage" else "원"
    title_metric = "전력사용량" if metric=="usage" else "전기요금"

    # 6) 차트 생성
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:,.0f}{unit}" for v in values],
        textposition="auto",
        hovertemplate=f"<b>%{{y}}</b><br>%{{x:,.0f}}{unit}<extra></extra>"
    ))

    # 7) 증감률 아이콘 & 텍스트
    change = (val_sel - val_prev) / val_prev * 100 if val_prev else 0
    arrow  = "🔺" if change>0 else "🔻"
    arrow_color = "red" if change>0 else "blue"
    subtitle = f"{arrow} 전월 대비: <span style='color:{arrow_color}'>{change:+.1f}%</span>"

    fig.update_layout(
        title=f"<b>전월/선택월/연간 평균 {title_metric} 비교</b><br><sub>{subtitle}</sub>",
        xaxis_title=f"{title_metric} ({unit})",
        yaxis_title="구분",
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=80, b=40),
        showlegend=False,
        font=dict(size=12),
        xaxis=dict(tickformat=",.0f")
    )

    return fig

def get_weather(lat=36.65446, lon=127.4500):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "Asia/Seoul"
        }
        response = requests.get(url, params=params, timeout=5)

        if response.status_code != 200:
            return f"🔌 오류 코드 [{response.status_code}] · 날씨 정보를 불러올 수 없습니다."

        data = response.json()
        weather = data["current_weather"]
        temp = round(weather["temperature"])
        windspeed = weather["windspeed"]
        code_map = {
            0: ("☀️", "맑음"),
            1: ("🌤️", "부분 맑음"),
            2: ("⛅", "구름 많음"),
            3: ("☁️", "흐림"),
            45: ("🌫️", "박무"),
            48: ("🌫️", "박무"),
            51: ("🌦️", "가벼운 이슬비"),
            61: ("🌧️", "비"),
            71: ("❄️", "눈"),
            95: ("⛈️", "뇌우"),
        }
        emoji, desc = code_map.get(weather["weathercode"], ("🌡️", "정보 없음"))

        return f"{emoji} {desc} | {temp}℃ | 풍속 {windspeed}km/h"
    except Exception as e:
        return f"❌ 예외 발생: {e}"

def build_summary_plain(d, sel_month, full_df):
    import pandas as pd

    # 1) 이번 달 집계
    # — 컬럼 자동 탐색
    date_col  = next((c for c in d.columns if "일시" in c or "datetime" in c), None)
    usage_col = next((c for c in d.columns if "전력사용량" in c or "power" in c.lower()), None)
    cost_col  = next((c for c in d.columns if "전기요금" in c or "cost" in c.lower()), None)
    if date_col is None or usage_col is None or cost_col is None:
        return "데이터 컬럼이 올바르지 않습니다."

    # — 현재 월 합계
    usage     = d[usage_col].sum()
    cost      = d[cost_col].sum()
    # — 피크 시간
    peak_idx  = d[cost_col].idxmax()
    peak_time = pd.to_datetime(d.loc[peak_idx, date_col])

    # — 야간 비율
    hours = d[date_col].dt.hour
    is_night = hours.ge(20) | hours.lt(6)
    night_ratio = is_night.mean()

    # 2) 이전 달 기준치 계산
    # — full_df 날짜 컬럼 변환
    ff = full_df.copy()
    ff[date_col] = pd.to_datetime(ff[date_col], errors="coerce")

    this_start = pd.to_datetime(f"{sel_month}-01")
    prev_start = (this_start - pd.offsets.MonthEnd(1)).replace(day=1)
    prev_end   = prev_start + pd.offsets.MonthEnd(0)

    prev = ff[(ff[date_col] >= prev_start) & (ff[date_col] <= prev_end)]
    prev_usage = prev[usage_col].sum() if not prev.empty else 0.0
    prev_cost  = prev[cost_col].sum()  if not prev.empty else 0.0

    # 3) 증감률 & 이상 징후 여부
    usage_rate = (usage - prev_usage) / prev_usage * 100 if prev_usage else 0.0
    cost_rate  = (cost  - prev_cost ) / prev_cost  * 100 if prev_cost  else 0.0
    is_anomaly = abs(usage_rate) > 15 or abs(cost_rate) > 20 or night_ratio > 0.6 or night_ratio < 0.2

    # 4) 포맷팅
    def fmt_rate(label, rate):
        arrow = "🔺" if rate > 0 else "🔻"
        return f"{arrow} {label} {rate:+.1f}%"

    summary = (
        "🧾 이번 달 리포트 요약\n"
        f"- 전력사용량: {usage:,.0f} kWh\n"
        f"- 전기요금: ₩{cost:,.0f}\n"
        f"- 전월 대비: {fmt_rate('사용량', usage_rate)}, {fmt_rate('요금', cost_rate)}\n"
        f"- 피크 시간: {peak_time:%Y-%m-%d %H:%M}\n"
        f"- 야간 사용 비율: {night_ratio*100:.1f}%\n"
        f"{'⚠️ 이상 징후 관측됨' if is_anomaly else '✅ 이상 징후 없음'}"
    )

    return summary



def build_summary_rich(d: pd.DataFrame, sel_month: str, full_df: pd.DataFrame) -> RichText:
    """
    d           : 이번 달 데이터 (summary_data())
    sel_month   : "YYYY-MM" 형식의 선택된 월
    full_df     : 전체 데이터프레임 (final_df)
    반환값      : DocxTemplate에 전달할 RichText 객체
    """
    # --- 1) 이번 달 집계치 계산 ---
    # 사용량, 요금, 피크 발생 시각
    usage     = d.filter(like="전력사용량").sum().iloc[0]
    cost      = d.filter(like="전기요금").sum().iloc[0]
    peak_idx  = d.filter(like="전기요금").idxmax()[0]
    peak_time = pd.to_datetime(d.loc[peak_idx, d.filter(like="측정일시").columns[0]])

    # 야간 비율
    hours = d[d.filter(like="측정일시").columns[0]].dt.hour
    night_ratio = ((hours >= 20) | (hours < 6)).mean()

    # --- 2) 이전 달 집계치 계산 ---
    this_start = pd.to_datetime(f"{sel_month}-01")
    prev_start = (this_start - pd.offsets.MonthEnd(1)).replace(day=1)
    prev_end   = prev_start + pd.offsets.MonthEnd(0)

    df_prev = full_df.copy()
    df_prev["측정일시"] = pd.to_datetime(df_prev["측정일시"], errors="coerce")
    mask = (df_prev["측정일시"] >= prev_start) & (df_prev["측정일시"] <= prev_end)
    df_prev = df_prev.loc[mask]

    usage_col = next((c for c in df_prev.columns if "전력사용량" in c), None)
    cost_col  = next((c for c in df_prev.columns if "전기요금" in c), None)
    prev_usage = df_prev[usage_col].sum() if usage_col else 0
    prev_cost  = df_prev[cost_col].sum()  if cost_col  else 0

    # --- 3) 증감률 및 이상 징후 판단 ---
    usage_rate = (usage - prev_usage) / prev_usage * 100 if prev_usage else 0
    cost_rate  = (cost  - prev_cost ) / prev_cost  * 100 if prev_cost  else 0
    is_anomaly = abs(usage_rate) > 15 or abs(cost_rate) > 20 or night_ratio > 0.6 or night_ratio < 0.2

    # --- 4) RichText 객체 생성 ---
    rt = RichText()
    rt.add("🧾 이번 달 리포트 요약\n", bold=True)
    rt.add(f"- 전력사용량: {usage:,.0f} kWh\n")
    rt.add(f"- 전기요금: ₩{cost:,.0f}\n")
    rt.add(f"- 전월 대비 사용량 {usage_rate:+.1f}% / 요금 {cost_rate:+.1f}%\n")
    rt.add(f"- 피크 시간: {peak_time:%Y-%m-%d %H:%M}\n")
    rt.add(f"- 야간 사용 비율: {night_ratio*100:.1f}%\n")
    if is_anomaly:
        rt.add("⚠️ 이상 징후 관측됨", color="red", bold=True)
    else:
        rt.add("✅ 이상 징후 없음", color="green", bold=True)

    return rt


# CSS 스타일 정의
css_style = """
<style>
/* 전체 Navbar 배경 및 글자색 */
.navbar {
    background-color: #60a5fa !important;  /* 하늘색 (Tailwind blue-400) */
    color: white !important;
}

/* 타이틀 부분 */
.navbar-brand {
    color: white !important;
    font-weight: bold;
}

/* 선택된 탭 스타일 */
.nav-link.active {
    background-color: #3b82f6 !important;  /* 더 진한 파랑 (blue-500) */
    color: white !important;
    font-weight: bold;
    border-radius: 6px;
}

/* 비활성 탭 스타일 */
.nav-link {
    color: white !important;
}

.metric-card {
    background: linear-gradient(135deg, #fcb045 0%, #fd1d1d 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 5px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

    flex: 1 1 0px;
    min-width: 180px;     /* 최소 폭 보장 */
    max-width: 250px;     /* 최대 폭 제한 (선택) */
    height: 130px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.power-card {
    background: linear-gradient(135deg, #fcd34d 0%, #f59e0b 100%);
}
.cost-card {
    background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
}
.co2-card {
    background: linear-gradient(135deg, #34d399 0%, #059669 100%);
}
.pf-card {
    background: linear-gradient(135deg, #10b981 0%, #047857 100%);
}
.type-card {
    background: linear-gradient(135deg, #6b7280 0%, #374151 100%);
}

.metric-card2 {
    background: linear-gradient(135deg, #fcb045 0%, #fd1d1d 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 5px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

    /* 👇 아래 두 줄 추가로 크기 통일 */
    width: 290px;  #인철 수정
    height: 130px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.card-container-flex {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;  /* 카드 간 간격 */
    align-items: stretch;
    gap: 10px;                        /* 카드 간 여백 */
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 5px;
}

.metric-label {
    font-size: 14px;
    opacity: 0.9;
}

.section-header {
    background-color: #60a5fa;
    color: white;
    padding: 10px 15px;
    margin: 10px 0 0 0;
    font-weight: bold;
    border-radius: 5px 5px 0 0;
}

.chart-container {
    border: 1px solid #ddd;
    border-radius: 0 0 5px 5px;
    padding: 15px;
    background-color: white;
    margin-bottom: 20px;
}

.progress-container {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}

.progress-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 15px 0;
    padding: 10px;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.progress-bar {
    height: 8px;
    background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
    border-radius: 4px;
    margin: 5px 0;
}

.sidebar-custom {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
}
.comparison-panel {
    display: flex;
    flex-direction: column;
    justify-content: stretch;
    height: 450px;        /* 왼쪽 그래프와 동일하게 */
    justify-content: center;
}
.analysis-grid {
    display: flex;
    flex-direction: column;
    gap: 15px;
    margin-top: 15px;
}

.card-row {
    display: flex;
    gap: 15px;
}

.info-card {
    flex: 1;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    border: none;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.info-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}

.card-header-custom {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
}

.card-icon {
    font-size: 24px;
    margin-right: 8px;
}

.card-title {
    color: #495057;
    font-size: 16px;
    font-weight: 600;
    margin: 0;
}

.card-content {
    font-size: 14px;
    color: #212529;
}

.cost-card { border-left: 4px solid #007bff; }
.eco-card { border-left: 4px solid #28a745; }
.workload-card { border-left: 4px solid #ffc107; }
.trend-card { border-left: 4px solid #17a2b8; }

/* 반응형 디자인 */
@media (max-width: 768px) {
    .card-row {
        flex-direction: column;
    }
}

</style>
"""
# UI 정의
app_ui = ui.page_navbar(
    
    ui.nav_panel("실시간 모니터링",
        ui.HTML(css_style),
        ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    ui.h4(" 실시간 모니터링", style="color: #2c3e50; margin-bottom: 20px;"),
                    ui.input_date_range(
                        "date_range_monitoring",
                        "기간 선택:",
                        start=test_df["측정일시"].min().strftime("%Y-%m-%d"),
                        end=test_df["측정일시"].max().strftime("%Y-%m-%d"),
                        format="yyyy-mm-dd"
                    ),
                    ui.br(),
                    ui.input_selectize(
                        "metrics_select",
                        "표시할 지표:",
                        choices={
                            "전력사용량": "전력사용량 (kWh)", 
                            "전기요금": "전기요금 (원)"
                        },
                        selected=["전력사용량", "전기요금"],
                        multiple=True
                    ),
                    ui.br(),
                    ui.br(),
                    class_="sidebar-custom"
                ),
                width=300
            ),
                # [A] 요약 카드 섹션
                ui.div(
                    ui.h4("실시간 주요 지표 요약 카드",   # 인철 수정
                          class_="section-header"),
                    ui.div(
                        ui.div(
                            ui.output_ui("card_power"),
                            ui.output_ui("card_cost"),
                            ui.output_ui("card_co2"),
                            ui.output_ui("card_pf"),
                            ui.output_ui("card_work_type"),
                            class_="card-container-flex"
                        ),
                        class_="chart-container"
                    )
                ),
                
                # [B] A번수 실시간 그래프 섹션
                ui.div(
                    ui.h4("실시간 전력 사용 및 요금 변화 추이", class_="section-header"),  ## 인철 수정
                    ui.div(
                        ui.row(
                            ui.column(12, output_widget("realtime_chart")),
                        ),
                        class_="chart-container"
                    )
                ),
                
                # [C] 전력사용량/료 슬라이더 섹션
                ui.div(
                    ui.h4("일/주/월 누적 사용량 및 요금 변화(기준 대비)",  ## 인철 수정
                          class_="section-header"),
                    ui.div(
                        ui.row(
                            ui.column(6,
                                ui.div(
                                    ui.h5("실시간 누적 전력사용량", style="color: #2c3e50;"),
                                    ui.output_ui("power_progress_bars"),
                                    style="padding: 15px;"
                                )
                            ),
                            ui.column(6,
                                ui.div(
                                    ui.h5("실시간 누적 전기요금", style="color: #2c3e50;"),
                                    ui.output_ui("cost_progress_bars"),
                                    style="padding: 15px;"
                                )
                            )
                        ),
                        class_="chart-container"
                    )
                ),
                
                # [D] 그래프/주/월/시간대별 작업 유형 분포 섹션
                ui.div(
                    ui.h4("시간대별 및 전체 작업 유형 분포(막대/파이 그래프)", class_="section-header"), ## 인철 수정
                    ui.div(
                        ui.row(
                            ui.column(8, output_widget("work_type_chart")),
                            ui.column(4, output_widget("work_type_pie"))
                        ),
                        class_="chart-container"
                    )
            )
        )
    ),
    
# ────────────────────
    # TAB 2: 전기요금 분석 보고서
    # ────────────────────

ui.nav_panel(
    "분석 보고서",

    # [A] 기간별 전력 사용 요약
ui.div(
    ui.h4("기간별 전력 사용 요약", class_="section-header"),
    ui.div(
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    id="selected_month",
                    label="분석 월 선택:",
                    choices=[f"2024-{m:02d}" for m in range(1, 13)],
                    selected="2024-05"
                ),
                # ui.input_radio_buttons(
                #         id="metric_type",
                #         label="📌 비교 항목 선택:",
                #         choices={"usage": "전력사용량", "cost": "전기요금"},
                #         selected="usage"
                # )
            ),
            ui.div(
                ui.row(
                    ui.column(3, ui.output_ui("summary_power_usage")),
                    ui.column(3, ui.output_ui("summary_power_cost")),
                    ui.column(3, ui.output_ui("summary_carbon_emission")),
                    ui.column(3, ui.output_ui("cal_power_factor_fee"))
                )
            ),
            ui.div(
                ui.row(
                    ui.column(6, output_widget("monthly_summary_chart")),
                    ui.column(6,
                        ui.div(
                            output_widget("monthly_comparison_chart"),
                            ui.div(
                                ui.tags.div(
                                    ui.tags.span("비교 항목 선택:", style="font-weight: 600; font-size: 15px; margin-right: 12px; color: #333;"),
                                    ui.input_radio_buttons(
                                        "metric_type",
                                        label=None,
                                        choices={"usage": "전력사용량", "cost": "전기요금"},
                                        selected="usage",
                                        inline=True
                                    ),
                                    style="""
                                        display: flex;
                                        align-items: center;
                                        justify-content: flex-start;
                                        gap: 18px;
                                        margin-top: 0px;
                                        margin-bottom: 5px;
                                        padding-left: 5px;
                                        flex-wrap: wrap;
                                        overflow-x: hidden;
                                    """
                                )
                            )
                        )
                    )
                ),
                class_="comparison-panel"
            )
        ),
        class_="chart-container"
    )
),


    # [B] 전력 요금 시계열 분석
    ui.div(
        ui.h4(" 전력 요금 시계열 분석", class_="section-header"),
        ui.div(
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_radio_buttons(
                        id="aggregation_unit",
                        label="집계 단위 선택:",
                        choices={
                            "hour": "시간대별",
                            "day": "일별",
                            "weekday": "요일별"
                        },
                        selected="hour"
                    )
                ),
                (output_widget("cost_trend_chart"))
            ),
        class_="chart-container"
        )
    ),


    # [C] 상세 분석 정보
ui.div(
    ui.h4("상세 분석 정보", class_="section-header"),
    ui.div(
        # 첫 번째 행
        ui.div(
            ui.div(
                ui.div(
                    ui.span("", class_="card-icon"),
                    ui.h5("최고 요금 정보", class_="card-title"),
                    class_="card-header-custom"
                ),
                ui.div(
                    ui.output_text("peak_cost_info"),
                    class_="card-content"
                ),
                class_="info-card cost-card"
            ),
            ui.div(
                ui.div(
                    ui.span("", class_="card-icon"),
                    ui.h5("평균 탄소배출량", class_="card-title"),
                    class_="card-header-custom"
                ),
                ui.div(
                    ui.output_text("avg_carbon_info"),
                    class_="card-content"
                ),
                class_="info-card eco-card"
            ),
            class_="card-row"
        ),
        # 두 번째 행
        ui.div(
            ui.div(
                ui.div(
                    ui.span("", class_="card-icon"),
                    ui.h5("주요 작업 유형", class_="card-title"),
                    class_="card-header-custom"
                ),
                ui.div(
                    ui.output_text("main_work_type_info"),
                    class_="card-content"
                ),
                class_="info-card workload-card"
            ),
            ui.div(
                ui.div(
                    ui.span("", class_="card-icon"),
                    ui.h5("전월 대비 증감률", class_="card-title"),
                    class_="card-header-custom"
                ),
                ui.div(
                    ui.output_text("monthly_change_info"),
                    class_="card-content"
                ),
                class_="info-card trend-card"
            ),
            class_="card-row"
        ),
        class_="analysis-grid"
    ),
    ui.div(
    ui.h4("이번 달 리포트 요약", class_="section-header"),
    ui.div(
        ui.output_ui("monthly_summary_text"),
        class_="chart-container",  # 여백 및 카드 스타일 유지
        style="margin-top: 20px;"
    )
    ),
    ui.div(
    ui.download_button("download_report", "📄 Word 보고서 다운로드", class_="btn-success btn-lg"),
    class_="text-center"
    )
)

),


    # ────────────────────
    # TAB 3: 부록
    # ────────────────────
    ui.nav_panel(
        "부록",
        ui.div(
            ui.h3("Appendix: 전기요금 예측 모델 개발 및 성능 향상 전처리 전략", 
                  style="color: #2c3e50; text-align: center; margin-bottom: 30px;"),
            
            # A1. 데이터 개요
            ui.div(
                ui.h4("📌 A1. 데이터 개요", class_="section-header"),
                ui.div(
                    ui.tags.ul(
                        ui.tags.li(ui.tags.strong("학습 데이터 ("), ui.tags.code("train.csv"), ui.tags.strong(") 및 테스트 데이터 ("), ui.tags.code("test.csv"), ui.tags.strong(")"), "는 15분 단위 전력 사용 이력과 환경정보 포함"),
                        ui.tags.li(ui.tags.strong("예측 타깃: "), ui.tags.code("전기요금(원)"), " 단일 목표 변수 예측")
                    ),
                    class_="chart-container",
                    style="padding: 20px;"
                )
            ),
            
            # A2. 전처리를 통한 성능 향상 전략
            ui.div(
                ui.h4("📌 A2. 전처리를 통한 성능 향상 전략", class_="section-header"),
                
                # A2-1. 시간 파생 변수 및 주기 인코딩
                ui.div(
                    ui.h5("🔷 A2-1. 시간 파생 변수 및 주기 인코딩", style="color: #34495e; margin-bottom: 15px;"),
                    ui.div(
                        ui.tags.table(
                            ui.tags.thead(
                                ui.tags.tr(
                                    ui.tags.th("변수", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;"),
                                    ui.tags.th("설명", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;"),
                                    ui.tags.th("성능 기여", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;")
                                )
                            ),
                            ui.tags.tbody(
                                ui.tags.tr(
                                    ui.tags.td(ui.tags.code("월, 일, 시간, 요일, 주말여부"), style="padding: 8px; border: 1px solid #ddd;"),
                                    ui.tags.td("시간 구조 반영", style="padding: 8px; border: 1px solid #ddd;"),
                                    ui.tags.td("요금과 계절/패턴 간 연관 반영", style="padding: 8px; border: 1px solid #ddd;")
                                ),
                                ui.tags.tr(
                                    ui.tags.td(ui.tags.code("sin_시간, cos_시간"), style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                    ui.tags.td("시간의 주기성 표현", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                    ui.tags.td("주기 구조를 부드럽게 인식 가능 (특히 LSTM에 유리)", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;")
                                )
                            ),
                            style="width: 100%; border-collapse: collapse; margin-bottom: 20px;"
                        ),
                        class_="chart-container"
                    )
                ),
                
                # A2-2. 시간대 기반 요금단가 계산
                ui.div(
                    ui.h5(" A2-2. 시간대 기반 요금단가 계산 (", ui.tags.code("요금단가"), ")", style="color: #34495e; margin-bottom: 15px;"),
                    ui.div(
                        ui.tags.ul(
                            ui.tags.li("계절, 시간대, 요금 정책 개편 시점을 반영한 실질 단가"),
                            ui.tags.li("전기요금의 구조적 요인 반영 → 예측 정밀도 향상")
                        ),
                        class_="chart-container",
                        style="padding: 15px;"
                    )
                ),
                
                # A2-3. Target Encoding 기반 통계적 인코딩
                ui.div(
                    ui.h5("A2-3. Target Encoding 기반 통계적 인코딩", style="color: #34495e; margin-bottom: 15px;"),
                    ui.div(
                        ui.tags.table(
                            ui.tags.thead(
                                ui.tags.tr(
                                    ui.tags.th("변수명", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;"),
                                    ui.tags.th("설명", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;")
                                )
                            ),
                            ui.tags.tbody(
                                ui.tags.tr(
                                    ui.tags.td(ui.tags.code("작업유형_te"), style="padding: 8px; border: 1px solid #ddd;"),
                                    ui.tags.td("각 작업유형별 평균 전기요금 반영", style="padding: 8px; border: 1px solid #ddd;")
                                ),
                                ui.tags.tr(
                                    ui.tags.td(ui.tags.code("시간_te"), style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                    ui.tags.td("시간대별 평균 요금 반영", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;")
                                ),
                                ui.tags.tr(
                                    ui.tags.td(ui.tags.code("요일_te"), style="padding: 8px; border: 1px solid #ddd;"),
                                    ui.tags.td("요일별 평균 요금 반영", style="padding: 8px; border: 1px solid #ddd;")
                                ),
                                ui.tags.tr(
                                    ui.tags.td(ui.tags.code("시간대_te"), style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                    ui.tags.td("최대/중간/경부하 구간별 평균 요금 반영", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;")
                                )
                            ),
                            style="width: 100%; border-collapse: collapse; margin-bottom: 20px;"
                        ),
                        class_="chart-container"
                    )
                ),
                
                # A2-4. 이상치 제거
                ui.div(
                    ui.h5("🔷 A2-4. 이상치 제거 (IQR 기반)", style="color: #34495e; margin-bottom: 15px;"),
                    ui.div(
                        ui.tags.ul(
                            ui.tags.li(ui.tags.code("전기요금(원)"), "의 이상치를 제거하여 학습 안정성 확보")
                        ),
                        class_="chart-container",
                        style="padding: 15px;"
                    )
                ),
                
                # A2-5. 스케일링 분리 적용
                ui.div(
                    ui.h5("🔷 A2-5. 스케일링 분리 적용", style="color: #34495e; margin-bottom: 15px;"),
                    ui.div(
                        ui.tags.table(
                            ui.tags.thead(
                                ui.tags.tr(
                                    ui.tags.th("모델군", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;"),
                                    ui.tags.th("스케일러", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;"),
                                    ui.tags.th("목적", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;")
                                )
                            ),
                            ui.tags.tbody(
                                ui.tags.tr(
                                    ui.tags.td("Tree 계열", style="padding: 8px; border: 1px solid #ddd;"),
                                    ui.tags.td(ui.tags.code("RobustScaler"), style="padding: 8px; border: 1px solid #ddd;"),
                                    ui.tags.td("이상치에 강건한 정규화", style="padding: 8px; border: 1px solid #ddd;")
                                ),
                                ui.tags.tr(
                                    ui.tags.td("LSTM", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                    ui.tags.td(ui.tags.code("MinMaxScaler"), style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                    ui.tags.td("시계열 학습 안정성 확보 (0~1 정규화 필수)", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;")
                                )
                            ),
                            style="width: 100%; border-collapse: collapse; margin-bottom: 20px;"
                        ),
                        class_="chart-container"
                    )
                )
            ),
            
            # A3. 모델별 구조 및 전략
            ui.div(
                ui.h4("📌 A3. 모델별 구조 및 전략", class_="section-header"),
                ui.div(
                    ui.tags.table(
                        ui.tags.thead(
                            ui.tags.tr(
                                ui.tags.th("모델", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;"),
                                ui.tags.th("특성", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;"),
                                ui.tags.th("역할", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;")
                            )
                        ),
                        ui.tags.tbody(
                            ui.tags.tr(
                                ui.tags.td("XGBoost / LGBM / RF", style="padding: 8px; border: 1px solid #ddd;"),
                                ui.tags.td("고차원 변수 처리", style="padding: 8px; border: 1px solid #ddd;"),
                                ui.tags.td("구조적 전처리 조합과 궁합 우수", style="padding: 8px; border: 1px solid #ddd;")
                            ),
                            ui.tags.tr(
                                ui.tags.td("LSTM", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                ui.tags.td("시계열 입력 (96×7)", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                ui.tags.td("주기/패턴 학습을 통한 정밀 예측", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;")
                            )
                        ),
                        style="width: 100%; border-collapse: collapse; margin-bottom: 15px;"
                    ),
                    ui.tags.ul(
                        ui.tags.li("각 모델의 ", ui.tags.code("R² Score"), "에 따라 ", ui.tags.strong("가중 앙상블 수행")),
                        ui.tags.li("LSTM + Tree 기반 모델의 상호 보완 구조")
                    ),
                    class_="chart-container",
                    style="padding: 20px;"
                )
            ),
            
            # A4. 하이퍼파라미터 최적화 전략
            ui.div(
                ui.h4("📌 A4. 하이퍼파라미터 최적화 전략", class_="section-header"),
                ui.div(
                    ui.tags.div(
                        ui.tags.h6("• 트리 기반 모델(XGB, LGBM, RF):", style="color: #2c3e50; font-weight: bold;"),
                        ui.tags.ul(
                            ui.tags.li("수작업 튜닝 + 경험적 값 고정 (", ui.tags.code("n_estimators, max_depth, learning_rate, subsample, colsample_bytree"), " 등)"),
                            ui.tags.li("탐색 공간을 제한하여 오버튜닝 방지 및 재현성 확보")
                        ),
                        style="margin-bottom: 20px;"
                    ),
                    ui.tags.div(
                        ui.tags.h6("• LSTM:", style="color: #2c3e50; font-weight: bold;"),
                        ui.tags.ul(
                            ui.tags.li("단층 구조 (", ui.tags.code("LSTM(64) → Dense(32) → Dense(1)"), ")"),
                            ui.tags.li(ui.tags.code("batch_size=32, epochs=20, EarlyStopping(patience=5)"), " 설정"),
                            ui.tags.li(ui.tags.code("Dropout"), "은 미사용 (모델 일반화 성능 확인 후 제외)")
                        ),
                        style="margin-bottom: 15px;"
                    ),
                    ui.tags.p("※ 추가적인 Optuna, GridSearch 등의 자동화 튜닝은 향후 확장 가능성으로 고려됨", 
                             style="color: #7f8c8d; font-style: italic;"),
                    class_="chart-container",
                    style="padding: 20px;"
                )
            ),
            
            # A5. 예측 결과 및 저장 산출물
            ui.div(
                ui.h4("📌 A5. 예측 결과 및 저장 산출물", class_="section-header"),
                ui.div(
                    ui.tags.ul(
                        ui.tags.li(ui.tags.code("submission_optimal.csv"), ": 앙상블 기반 전기요금 예측 결과 저장")
                    ),
                    class_="chart-container",
                    style="padding: 20px;"
                )
            ),
            
            # A6. 저장된 모델
            ui.div(
                ui.h4("📌 A6. 저장된 모델", class_="section-header"),
                ui.div(
                    ui.tags.table(
                        ui.tags.thead(
                            ui.tags.tr(
                                ui.tags.th("파일명", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;"),
                                ui.tags.th("설명", style="background-color: #E0E0E0; color: black; padding: 10px; border: 1px solid #ddd;")
                            )
                        ),
                        ui.tags.tbody(
                            ui.tags.tr(
                                ui.tags.td(ui.tags.code("xgb.pkl, lgb.pkl, rf.pkl"), style="padding: 8px; border: 1px solid #ddd;"),
                                ui.tags.td("트리 계열 학습 모델", style="padding: 8px; border: 1px solid #ddd;")
                            ),
                            ui.tags.tr(
                                ui.tags.td(ui.tags.code("lstm.pkl"), style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;"),
                                ui.tags.td("학습된 시계열 모델", style="padding: 8px; border: 1px solid #ddd; background-color: #f8f9fa;")
                            ),
                            ui.tags.tr(
                                ui.tags.td(ui.tags.code("scaler.pkl, seq_scaler.pkl"), style="padding: 8px; border: 1px solid #ddd;"),
                                ui.tags.td("입력 스케일러 객체 저장용", style="padding: 8px; border: 1px solid #ddd;")
                            )
                        ),
                        style="width: 100%; border-collapse: collapse;"
                    ),
                    class_="chart-container",
                    style="padding: 20px;"
                )
            ),
            
            # A7. 모델 선택을 위한 성능 평가
            ui.div(
                ui.h4("📌 A7. 모델 선택을 위한 성능 평가", class_="section-header"),
                ui.div(
                    ui.tags.div(
                        ui.tags.h6("🧪 실사용 성능을 고려한 평가 절차:", style="color: #2c3e50; font-weight: bold; margin-bottom: 10px;"),
                        ui.tags.ul(
                            ui.tags.li(ui.tags.strong("학습"), ": 1월 ~ 10월 데이터를 기반으로 모델 학습"),
                            ui.tags.li(ui.tags.strong("검증"), ": 11월 데이터를 예측하고 실제 ", ui.tags.code("전기요금(원)"), "과 비교"),
                            ui.tags.li(ui.tags.strong("지표"), ": ", ui.tags.code("Mean Absolute Error (MAE)"), "를 주요 기준으로 사용")
                        ),
                        style="margin-bottom: 20px;"
                    ),
                    ui.tags.div(
                        ui.tags.h6("🏆 모델 선정 기준:", style="color: #2c3e50; font-weight: bold; margin-bottom: 10px;"),
                        ui.tags.ul(
                            ui.tags.li("Tree 기반 모델과 LSTM, 그리고 두 모델의 앙상블 결과를 비교"),
                            ui.tags.li("앙상블 모델이 11월 전체에 대해 ", ui.tags.strong("가장 낮은 MAE를 기록"), "하여 최종 예측 모델로 선택됨")
                        )
                    ),
                    class_="chart-container",
                    style="padding: 20px;"
                )
            ),
            
            # 핵심 요약
            ui.div(
                ui.h4("✅ 핵심 요약", class_="section-header"),
                ui.div(
                    ui.tags.p(
                        "본 모델은 시간 기반 요금 단가 계산, 범주형 변수에 대한 통계적 인코딩, 적절한 이상치 제거와 스케일링 전략 분리, 하이퍼파라미터 튜닝을 통한 구조 최적화, 그리고 11월 실측 기반 성능 검증을 통해 최종적으로 Tree + LSTM 앙상블 모델을 선택하였다.",
                        style="font-size: 16px; line-height: 1.6; text-align: justify; background-color: #ecf0f1; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db;"
                    ),
                    class_="chart-container",
                    style="padding: 20px;"
                    
                )
            ),
            style="padding: 20px; max-width: 900px; margin: 0 auto;"
            
        )
    ),


    # 날씨를 가장 오른쪽에 배치하기 위해 nav_spacer와 nav_control 사용
    ui.nav_spacer(),  # 빈 공간을 만들어 오른쪽으로 밀어냄
    ui.nav_control(ui.output_ui("navbar_weather")),  # 날씨를 오른쪽 끝에 배치

    # title은 단순하게 변경
    title="LS Electric 전기요금 실시간 모니터링",
    id="main_navbar"
)

def server(input, output, session):
    # ───────────────────────────────────────────────────────
    # 0) FigureWidget 초기화 (한 번만)
    # ───────────────────────────────────────────────────────
    fig_realtime = FigureWidget()
    fig_realtime.add_scatter(name="전력사용량", mode="lines")
    fig_realtime.add_scatter(name="전기요금", mode="lines", yaxis="y2")
    
    fig_realtime.update_layout(
    yaxis=dict(title="전력사용량", side="left"),
    yaxis2=dict(
        title="전기요금",
        overlaying="y",
        side="right",
        showgrid=False
    ),
    showlegend=True,
    height=300,
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(orientation="h", y=-0.2)
)
    

    # ───────────────────────────────────────────────────────
    # 1) Reactive 데이터 준비 (분석 보고서 탭)
    # ───────────────────────────────────────────────────────
    @reactive.Calc
    def summary_data():
        try:
            base_dir = os.path.dirname(__file__)
            file_path = os.path.abspath(os.path.join(base_dir, ".", "data", "df_final.csv"))
            df_final = pd.read_csv(file_path)

            if "측정일시" not in df_final.columns:
                raise KeyError("'측정일시' 컬럼이 없습니다.")
            df_final["측정일시"] = pd.to_datetime(df_final["측정일시"], errors="coerce")

            df2 = df_final.copy()

            selected_month = input.selected_month()
            if not selected_month:
                print("선택된 월 없음. 기본값 반환")
                return df2

            start = pd.to_datetime(selected_month + "-01")
            end = start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)  
            df2 = df2[(df2["측정일시"] >= start) & (df2["측정일시"] < end)]

            return df2

        except Exception as e:
            print(f"summary_data() 실행 중 오류: {e}")
            return pd.DataFrame()
        
    
    @reactive.Calc
    def monthly_summary_data():
        base_dir = os.path.dirname(__file__)
        file_path = os.path.abspath(os.path.join(base_dir, "..", "data", "monthly_summary.csv"))

        df = pd.read_csv(SUMMARY_CSV)
        df["월"] = pd.PeriodIndex(df["월"], freq="M")
        return df



    # ───────────────────────────────────────────────────────
    # 2) Reactive 데이터 준비 (분석 보고서 탭)
    # ───────────────────────────────────────────────────────

    @reactive.Calc
    def report_summary_data():
        print("🔥 report_summary_data 시작됨")
        try:
            df_final = final_df.copy()
            df_final["측정일시"] = pd.to_datetime(df_final["측정일시"], errors="coerce")
            start_raw, end_raw = input.date_range()
            print("date_range input:", start_raw, "~", end_raw)

            start_raw, end_raw = input.date_range()
            if not start_raw or not end_raw:
                print("날짜 범위 미지정")
                return pd.DataFrame()

            start = pd.to_datetime(start_raw)
            end = pd.to_datetime(end_raw) + timedelta(days=1)
            df_final = df_final[(df_final["측정일시"] >= start) & (df_final["측정일시"] < end)]
            
            print("✅ 필터링 완료:", df_final.shape)
            return df_final

        except Exception as e:
            print("report_summary_data 오류:", e)
            return pd.DataFrame()
       



    # ─────────────────────────────
    # 💡 연결 유지를 위한 keep-alive 트리거
    # ─────────────────────────────
    @reactive.effect
    def keep_alive():
        reactive.invalidate_later(1000)  # 1초마다 반복
        # 아무것도 안 해도 되지만, 로그 남기면 디버깅에 도움
        print("🔄 Keep-alive tick.")
    # ───────────────────────────────────────────────────────
    # 2) Streamer & Accumulator 세팅 (실시간 데이터)
    # ───────────────────────────────────────────────────────
    streamer = reactive.Value(Streamer(test_df))
    accumulator = reactive.Value(Accumulator())
    is_streaming = reactive.Value(True)
    current_data = reactive.Value(pd.DataFrame())

    @reactive.effect
    def stream_data():
        try:
            if not is_streaming.get():
                return
            interval_ms = 1000
            reactive.invalidate_later(1)
            s = streamer.get()
            next_batch = s.get_next_batch(1)
            if next_batch is not None:
                accumulator.get().accumulate(next_batch)
                current_data.set(accumulator.get().get())
            else:
                is_streaming.set(False)
        except asyncio.CancelledError:
            print("⛔️ 스트리밍 중단됨 (CancelledError)")

    @reactive.Calc
    def simulated_data():
        # current_data 의 최신값을 가져옴
        current_data.get()
        d = current_data.get()
        if d.empty:
            return pd.DataFrame()
        start, end = input.date_range_monitoring()
        return d[(d["측정일시"] >= pd.to_datetime(start)) & (d["측정일시"] <= pd.to_datetime(end))]

    # ───────────────────────────────────────────────────────
    # 3) 실시간 플롯: 업데이트만, 다시 그리진 않음
    # ───────────────────────────────────────────────────────
    @reactive.effect
    def update_realtime_plot():
        interval_ms = 1000
        reactive.invalidate_later(interval_ms)

        d = simulated_data()
        x = d["측정일시"].tolist() if not d.empty else []

        # 전력사용량 trace (0)
        if "전력사용량" in input.metrics_select() and x:
            fig_realtime.data[0].x = x
            fig_realtime.data[0].y = d["전력사용량"].tolist()
        else:
            fig_realtime.data[0].x = []
            fig_realtime.data[0].y = []

        # 전기요금 trace (1)
        if "전기요금" in input.metrics_select() and x:
            fig_realtime.data[1].x = x
            fig_realtime.data[1].y = d["전기요금"].tolist()
        else:
            fig_realtime.data[1].x = []
            fig_realtime.data[1].y = []

    # ───────────────────────────────────────────────────────
    # 4) Output 정의
    # ───────────────────────────────────────────────────────
    @render.ui("navbar_weather")
    def navbar_weather():
        txt = get_weather(lat=36.65446, lon=127.4500)
        return ui.div(
            ui.span(txt, style="font-weight:400;color: white;"), # 인철 추가
                style="""
                    display: flex;
                    gap: 8px;
                    align-items: center;
                    font-size: 16px;      /* 원하시는 크기로 조정 가능 */
                """
        )
    
    @output
    @render_widget
    def realtime_chart():
        # 항상 동일한 FigureWidget 반환
        return fig_realtime
    
    @output
    @render.ui
    def card_power():
        d = simulated_data()
        val = d["전력사용량"].iloc[-1] if not d.empty else 0
        return ui.div(
            ui.div(f"{val:,.1f} kWh", class_="metric-value"),  # 숫자 + 단위 한 줄
            ui.div("누적 전력사용량", class_="metric-label"),
            class_="metric-card power-card"
    )

    @output
    @render.ui
    def card_cost():
        d = simulated_data()
        val = d["전기요금"].iloc[-1] if not d.empty else 0
        return ui.div(ui.div(f"{val:,.0f}", class_="metric-value"), ui.div("전력요금(원)", class_="metric-label"), class_="metric-card cost-card")

    @output
    @render.ui
    def card_co2():
        d = simulated_data()
        val = d["탄소배출량"].iloc[-1] if not d.empty else 0
        val = abs(val) ## 인철 수정
        return ui.div(ui.div(f"{val:,.2f}", class_="metric-value"), ui.div("CO₂", class_="metric-label"), class_="metric-card co2-card")

    @output
    @render.ui
    def card_pf():
        return ui.div(ui.div("0.95", class_="metric-value"), ui.div("PF", class_="metric-label"), class_="metric-card pf-card")

    @output
    @render.ui
    def card_work_type():
        d = simulated_data()
        typ = d["작업유형"].mode().iloc[0] if not d.empty else "N/A"
        return ui.div(
        ui.div(typ, class_="metric-value", style="font-size:18px; word-break:break-word;"),
        ui.div("작업유형", class_="metric-label"),
        class_="metric-card type-card"
    )


    # ───────────────────────────────────────────────────────
    # 4) [C] 진행률 바 공통 함수 및 렌더링
    # ───────────────────────────────────────────────────────
    def _make_bar(label, val, denom, color, start_color=None):
        pct = min(100, val / denom * 100) if denom else 0
        # start_color 지정 없으면 기본값은 color로 시작 (단색처럼 보임)
        start = start_color if start_color else color
        return ui.div(
            ui.div(f"{label}: {val:,.0f} ({pct:.1f}%) / 기준: {denom:,.0f}", style="font-weight:bold; margin-bottom:4px;"),
            ui.div(
                ui.div(style=f"""
                    width:{pct:.1f}%;
                    height:12px;
                    background: linear-gradient(to right, {start}, {color});
                    border-radius:4px;
                """),
                style="width:100%; height:12px; background:#e9ecef; border-radius:4px; overflow:hidden;"
            ),
            style="margin:12px 0; padding:4px;"
        )
    @output
    @render.ui
    def power_progress_bars():
        d = simulated_data()
        if d.empty:
            return ui.div("데이터 없음")

        now = d["측정일시"].max()
        today = now.normalize()
        week_start = today - timedelta(days=today.weekday())  # 월요일 기준
        month_start = today.replace(day=1)

        # ✅ 현재 누적값
        day_usage = d[d["측정일시"].dt.date == today.date()]["전력사용량"].sum()
        week_usage = d[(d["측정일시"] >= week_start) & (d["측정일시"] <= now)]["전력사용량"].sum()
        month_usage = d[(d["측정일시"] >= month_start) & (d["측정일시"] <= now)]["전력사용량"].sum()

        # ✅ 기준값 대비 퍼센트
        return ui.div(
            _make_bar("일일 누적", day_usage, nov_baseline["power"]["daily"], "#fef9c3"),
            _make_bar("주별 누적", week_usage, nov_baseline["power"]["weekly"], "#fcd34d"),
            _make_bar("월별 누적", month_usage, nov_baseline["power"]["monthly"], "#f59e0b"),
        )


    @output
    @render.ui
    def cost_progress_bars():
        d = simulated_data()
        if d.empty:
            return ui.div("데이터 없음")

        now = d["측정일시"].max()
        today = now.normalize()
        week_start = today - timedelta(days=today.weekday())
        month_start = today.replace(day=1)

        day_cost = d[d["측정일시"].dt.date == today.date()]["전기요금"].sum()
        week_cost = d[(d["측정일시"] >= week_start) & (d["측정일시"] <= now)]["전기요금"].sum()
        month_cost = d[(d["측정일시"] >= month_start) & (d["측정일시"] <= now)]["전기요금"].sum()

        return ui.div(
            _make_bar("일일 누적", day_cost, nov_baseline["cost"]["daily"], "#aed6f1"),
            _make_bar("주별 누적", week_cost, nov_baseline["cost"]["weekly"], "#5dade2"),
            _make_bar("월별 누적", month_cost, nov_baseline["cost"]["monthly"], "#3498db"),
        )

    # ───────────────────────────────────────────────────────
    # 5) [D] 작업 유형 분포
    # ───────────────────────────────────────────────────────
    @output
    @render_widget
    def work_type_chart():
        d = simulated_data()
        if d.empty:
            return None

        # 시간대별 작업유형 비율 계산
        hourly = (
            d.groupby([d["측정일시"].dt.hour, "작업유형"])
            .size()
            .unstack(fill_value=0)
        )
        hourly_ratio = hourly.div(hourly.sum(axis=1), axis=0)

        color_map = {
            "Light_Load": "#90ee90",   # 연초록
            "Medium_Load": "#87cefa",  # 하늘색
            "Maximum_Load": "#ef4444"  # 빨강
        }

        fig = go.Figure()
        for col in hourly_ratio.columns:
            fig.add_trace(go.Bar(x=hourly_ratio.index, y=hourly_ratio[col], name=col,marker_color=color_map.get(col, "gray") ))

        fig.update_layout(
            barmode="stack",
            title="시간대별 작업 유형 분포 (비율 기반)",
            xaxis_title="시간",
            yaxis_title="비율 (%)",
            height=300,
            yaxis=dict(tickformat=".0%"),  # ✅ 퍼센트 포맷
            uirevision="STATIC"
        )
        return fig

    @output
    @render_widget
    def work_type_pie():
        d = simulated_data()
        if d.empty:
            return None

        cnt = d["작업유형"].value_counts()

        return px.pie(
            values=cnt.values,
            names=cnt.index,
            title="작업유형별 분포",
            height=300,
            color=cnt.index, ## 인철 수정
            color_discrete_map={
                "Light_Load": "#90ee90",    # 연초록
                "Medium_Load": "#87cefa",   # 하늘색
                "Maximum_Load": "#ef4444"   # 빨강
            }
        )
    
    @reactive.Calc
    def simulated_data():
        current_data.get()
        d = current_data.get()
        if d.empty:
            return pd.DataFrame()
        start, end = input.date_range_monitoring()
        d = d[(d["측정일시"] >= pd.to_datetime(start)) & (d["측정일시"] <= pd.to_datetime(end))]
        return d


    

    # ───────────────────────────────────────────────────────
    # 6) TAB 2: 분석 보고서 출력
    # ───────────────────────────────────────────────────────

    @output
    @render.ui
    def summary_power_usage():
        d = monthly_summary_data()
        selected = pd.Period(input.selected_month(), freq="M")

        current_val = d.loc[d["월"] == selected, "전력사용량(kWh)"].sum()
        prev_val = d.loc[d["월"] == (selected - 1), "전력사용량(kWh)"].sum()
        change = ((current_val - prev_val) / prev_val * 100) if prev_val else 0

        return ui.div(
            ui.div(f"{current_val:,.1f} kWh", class_="metric-value"),
            ui.div("누적 전력사용량", class_="metric-label"),
            ui.div(f"전월 대비 {change:+.1f}%", class_="metric-subtext"),
            class_="metric-card power-card"
        )



    @output
    @render.ui
    def summary_power_cost():
        d = monthly_summary_data()
        selected = pd.Period(input.selected_month(), freq="M")

        current_val = d.loc[d["월"] == selected, "전기요금(원)"].sum()
        prev_val = d.loc[d["월"] == (selected - 1), "전기요금(원)"].sum()
        change = ((current_val - prev_val) / prev_val * 100) if prev_val else 0

        return ui.div(
            ui.div(f"₩{current_val:,.0f}", class_="metric-value"),
            ui.div("누적 전력요금", class_="metric-label"),
            ui.div(f"전월 대비 {change:+.1f}%", class_="metric-subtext"),
            class_="metric-card cost-card"
        )



    @output
    @render.ui
    def summary_carbon_emission():
        d = monthly_summary_data()
        selected = pd.Period(input.selected_month(), freq="M")

        current_val = d.loc[d["월"] == selected, "탄소배출량(tCO2)"].sum()
        prev_val = d.loc[d["월"] == (selected - 1), "탄소배출량(tCO2)"].sum()
        change = ((current_val - prev_val) / prev_val * 100) if prev_val else 0

        return ui.div(
            ui.div(f"{current_val:,.2f} tCO₂", class_="metric-value"),
            ui.div("누적 탄소배출량", class_="metric-label"),
            ui.div(f"전월 대비 {change:+.1f}%", class_="metric-subtext"),
            class_="metric-card co2-card"
        )




    @output
    @render.ui
    def cal_power_factor_fee():
        d = monthly_summary_data()
        selected = pd.Period(input.selected_month(), freq="M")

        current_val = d.loc[d["월"] == selected, "역률요금"].sum()
        prev_val = d.loc[d["월"] == (selected - 1), "역률요금"].sum()
        change = ((current_val - prev_val) / prev_val * 100) if prev_val else 0

        return ui.div(
            ui.div(f"₩{current_val:,.0f}", class_="metric-value"),
            ui.div("역률 요금", class_="metric-label"),
            ui.div(f"전월 대비 {change:+.1f}%", class_="metric-subtext"),
            class_="metric-card pf-card"
        )




    @output
    @render_widget
    def cost_trend_chart():
        """전력 요금 시계열 분석 차트 - 집계 단위 필터 적용"""
        try:
            d = summary_data()
            if d is None or len(d) == 0:
                return create_simple_error_chart("데이터가 없습니다")

            # 컬럼 설정
            date_col = next((col for col in d.columns if '일시' in col or '시간' in col), None)
            cost_col = next((col for col in d.columns if '요금' in col or '비용' in col), None)

            if not date_col or not cost_col:
                return create_simple_error_chart("날짜 또는 요금 컬럼을 찾을 수 없습니다")

            # 전처리
            df = d[[date_col, cost_col]].copy()
            df.columns = ['datetime', 'cost']
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
            df = df.dropna().sort_values('datetime')

            if len(df) == 0:
                return create_simple_error_chart("유효한 데이터가 없습니다")

            # 집계 단위 처리
            aggregation = input.aggregation_unit()

            if aggregation == "hour":
                df["period"] = df["datetime"].dt.floor("H")
                df["label"] = df["period"].dt.strftime("%m/%d %H:%M")

            elif aggregation == "day":
                df["period"] = df["datetime"].dt.date
                df["label"] = pd.to_datetime(df["period"]).dt.strftime("%m/%d")

            elif aggregation == "weekday":
                df["weekday_num"] = df["datetime"].dt.weekday  # 0~6
                df["period"] = df["weekday_num"]
                weekday_map = {
                    0: "월요일", 1: "화요일", 2: "수요일",
                    3: "목요일", 4: "금요일", 5: "토요일", 6: "일요일"
                }
                df["label"] = df["weekday_num"].map(weekday_map)

            else:
                return create_simple_error_chart("집계 단위가 올바르지 않습니다")

            # 집계
            if aggregation == "weekday":
                # 요일별 집계 시 순서 보장
                agg_df = df.groupby(["weekday_num", "label"])["cost"].agg(["sum", "mean"]).reset_index()
                agg_df.columns = ["weekday_num", "label", "total", "average"]
                # 요일 순서대로 정렬 (월요일=0 ~ 일요일=6)
                agg_df = agg_df.sort_values("weekday_num")
            else:
                agg_df = df.groupby("label")["cost"].agg(["sum", "mean"]).reset_index()
                agg_df.columns = ["label", "total", "average"]

            # 차트
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=agg_df['label'],
                y=agg_df['total'],
                name='누적 요금',
                marker=dict(color='#2563eb', line=dict(color='darkgray', width=0.5)),
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>누적: %{y:,.0f}원<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=agg_df['label'],
                y=agg_df['average'],
                mode='lines+markers',
                name='평균 요금',
                line=dict(color='black', width=3),
                marker=dict(color='black', size=6),
                yaxis='y2',
                hovertemplate='<b>%{x}</b><br>평균: %{y:,.0f}원<extra></extra>'
            ))

            fig.update_layout(
                title=dict(
                    text='<b>[B] 전력 요금 시계열 분석</b>',
                    x=0.5,
                    font=dict(size=16, color='black')
                ),
                xaxis=dict(
                    title='시간',
                    tickangle=-45,
                    type='category',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title='누적 요금 (원)',
                    side='left',
                    showgrid=True,
                    gridcolor='lightgray',
                    tickformat=',.0f'
                ),
                yaxis2=dict(
                    title='평균 요금 (원)',
                    side='right',
                    overlaying='y',
                    tickformat=',.0f',
                    showgrid=False
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=450,
                margin=dict(l=60, r=60, t=80, b=60),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                ),
                hovermode='x unified'
            )

            # 통계 요약
            total_cost = agg_df['total'].sum()
            avg_cost = agg_df['average'].mean()
            max_cost = agg_df['total'].max()
            data_points = len(df)

            fig.add_annotation(
                text=f"<b> 요약 통계</b><br>" +
                    f"총 요금: {total_cost:,.0f}원<br>" +
                    f"평균: {avg_cost:,.0f}원<br>" +
                    f"최대: {max_cost:,.0f}원<br>" +
                    f"데이터: {data_points:,}개",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10, color="black")
            )

            return fig

        except Exception as e:
            return create_simple_error_chart(f"오류: {str(e)}")


    def create_simple_error_chart(message):
        """간단한 에러 차트"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
            xref="paper", yref="paper"
        )
        fig.update_layout(
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white",
            title="차트 오류"
        )
        return fig

    @output
    @render_widget
    def monthly_summary_chart():
        """1년간 월별 전력사용량 + 평균요금 추이 (원본 데이터 전체 사용)"""
        try:
            # 1) 전역 final_df 복사
            df = final_df.copy()
            if df.empty or "측정일시" not in df.columns:
                return create_simple_error_chart("데이터가 없습니다")

            # 2) 날짜 변환
            df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")

            # 3) 최근 1년 데이터 필터링
            latest = df["측정일시"].max()
            one_year_ago = latest - pd.DateOffset(years=1)
            df = df[(df["측정일시"] >= one_year_ago) & (df["측정일시"] <= latest)]
            if df.empty:
                return create_simple_error_chart("최근 1년 데이터가 없습니다")

            # 4) 월별 집계: 기간은 Period("M") → timestamp
            df["측정월"] = df["측정일시"].dt.to_period("M")
            # 전력사용량 합, 평균요금 평균
            monthly = (
                df.groupby("측정월")
                .agg({
                    next(c for c in df.columns if "전력사용량" in c): "sum",
                    next(c for c in df.columns if "전기요금"   in c): "mean"
                })
                .reset_index()
            )
            monthly["측정월_라벨"] = monthly["측정월"].dt.strftime("%Y-%m")

            # 5) 색칠
            sel = input.selected_month()
            monthly["막대색"] = np.where(monthly["측정월_라벨"] == sel, "red", "gray")

            # 6) 그리기
            fig = go.Figure()
            usage_col = next(c for c in monthly.columns if "전력사용량" in c)
            cost_col  = next(c for c in monthly.columns if "전기요금"   in c)

            fig.add_trace(go.Bar(
                x=monthly["측정월_라벨"],
                y=monthly[usage_col],
                name="전력사용량",
                marker_color=monthly["막대색"],
                yaxis="y1",
            ))
            fig.add_trace(go.Scatter(
                x=monthly["측정월_라벨"],
                y=monthly[cost_col],
                name="평균요금",
                mode="lines+markers",
                yaxis="y2",
            ))
            # 범례용 더미
            fig.add_trace(go.Bar(x=[None], y=[None], name="현재 분석 달", marker_color="red"))

            fig.update_layout(
                title="1년간 월별 전력사용량 및 평균요금 추이",
                xaxis=dict(title="월", type="category"),
                yaxis=dict(title="전력사용량 (kWh)", side="left"),
                yaxis2=dict(title="평균요금 (원)", side="right", overlaying="y", showgrid=False),
                height=450,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(t=60, b=60, l=60, r=60),
            )
            return fig

        except Exception as e:
            return create_simple_error_chart(f"오류 발생: {e}")

        
    @output
    @render_widget
    def monthly_comparison_chart():
        return make_comparison_chart(final_df, input.selected_month(), input.metric_type())



            
    @output
    @render.text
    def peak_cost_info():
        d = summary_data()
        if d.empty:
            return "데이터 없음"
        idx = d["전기요금(원)"].idxmax()
        cost = d.loc[idx, "전기요금(원)"]
        dt   = d.loc[idx, "측정일시"]
        return f"최고요금: ₩{cost:,.0f}\n발생일시: {dt:%Y-%m-%d %H:%M}\n요일: {dt:%A}"

    @output
    @render.text
    def avg_carbon_info():
        d = summary_data()
        print(d["탄소배출량(tCO2)"].describe())
        print(d["측정일시"].min(), d["측정일시"].max())

        if d.empty:
            return "데이터 없음"
        avg, tot = d["탄소배출량(tCO2)"].mean(), d["탄소배출량(tCO2)"].sum()
        return f"평균: {avg:.3f} tCO₂\n총 배출량: {tot:.1f} tCO₂"
    
    @output
    @render.ui
    def monthly_summary_text():
        d = summary_data()
        if d.empty:
            return ui.p("데이터 없음")

        usage = d["전력사용량(kWh)"].sum()
        cost = d["전기요금(원)"].sum()

        peak_idx = d["전기요금(원)"].idxmax()
        peak_time = pd.to_datetime(d.loc[peak_idx, "측정일시"])

        d["hour"] = d["측정일시"].dt.hour
        d["is_night"] = d["hour"].apply(lambda x: x >= 20 or x < 6)
        night_ratio = d["is_night"].mean()

        selected_month = input.selected_month()
        cur_start = pd.to_datetime(selected_month + "-01")
        prev_start = cur_start - pd.DateOffset(months=1)
        prev_end = cur_start - pd.Timedelta(days=1)

        df_full = final_df
        df_full["측정일시"] = pd.to_datetime(df_full["측정일시"])
        prev_df = df_full[(df_full["측정일시"] >= prev_start) & (df_full["측정일시"] <= prev_end)]

        summary_df = monthly_summary_data()
        selected = pd.Period(input.selected_month(), freq="M")
        prev_period = selected - 1

        prev_usage = summary_df.loc[summary_df["월"] == prev_period, "전력사용량(kWh)"].sum()
        prev_cost = summary_df.loc[summary_df["월"] == prev_period, "전기요금(원)"].sum()

        summary_df = monthly_summary_data()
        selected = pd.Period(input.selected_month(), freq="M")

        cur_usage = summary_df.loc[summary_df["월"] == selected, "전력사용량(kWh)"].sum()
        prev_usage = summary_df.loc[summary_df["월"] == (selected - 1), "전력사용량(kWh)"].sum()
        usage_rate = (cur_usage - prev_usage) / prev_usage * 100 if prev_usage else 0

        cur_cost = summary_df.loc[summary_df["월"] == selected, "전기요금(원)"].sum()
        prev_cost = summary_df.loc[summary_df["월"] == (selected - 1), "전기요금(원)"].sum()
        cost_rate = (cur_cost - prev_cost) / prev_cost * 100 if prev_cost else 0

        # ✅ 이상 징후 판단
        is_anomaly = (
            abs(usage_rate) > 15
            or abs(cost_rate) > 20
            or night_ratio > 0.6
            or night_ratio < 0.2
            or not (8 <= peak_time.hour <= 10 or 18 <= peak_time.hour <= 21)
        )

        anomaly_msg = (
            "⚠️ 이번 달에는 일부 항목에서 주의가 필요한 <b style='color:#d9534f'>이상 징후</b>가 관측되었습니다."
            if is_anomaly else
            "이번 달에는 이상 징후가 관측되지 않았으며, 에너지는 전반적으로 안정적으로 사용된 것으로 판단됩니다."
        )

        def color_text(label, rate):
            color = "gray"
            if rate > 0:
                color = "red"
            elif rate < 0:
                color = "blue"
            return f"<span style='color:{color}; font-weight:bold'>{label} {rate:+.1f}%</span>"

        usage_html = color_text("사용량", usage_rate)
        cost_html = color_text("요금", cost_rate)

        return ui.HTML(
            f"""
            <div style='padding: 15px; background-color: #f9f9f9; border-radius: 10px; font-size: 14px;'>
                <h5 style='margin-bottom: 8px; color: #2c3e50;'>🧾 이번 달 리포트 요약</h5>
                <p>이번 달 전력사용량은 총 <b>{usage:,.0f} kWh</b>, 전기요금은 약 <b>₩{cost:,.0f}</b>으로 집계되었습니다.</p>
                <p>전월 대비 {usage_html}, {cost_html}의 변화가 있었으며,<br>피크 요금은 <b>{peak_time:%Y-%m-%d %H:%M}</b>에 발생해 시간대 관리 필요성을 시사합니다.</p>
                <p>야간 시간대(20시~6시) 전력 사용 비율은 <b>{night_ratio*100:.1f}%</b>로 확인되었습니다.</p>
                <p>{anomaly_msg}</p>
            </div>
            """
        )

    
    @output
    @render.text
    def main_work_type_info():
        d = summary_data()
        if d.empty or "작업유형" not in d:
            return "데이터 없음"
        vc = d["작업유형"].value_counts()
        top, cnt, tot = vc.idxmax(), vc.max(), vc.sum()
        return f"최다 작업유형: {top}\n비중: {cnt/tot*100:.1f}% ({cnt}건)"

    @output
    @render.text
    def monthly_change_info():
        d = summary_data()
        if d.empty:
            return "데이터 없음" ###

        # 📅 현재 월 범위
        selected_month = input.selected_month()
        cur_start = pd.to_datetime(selected_month + "-01")
        cur_end = cur_start + pd.offsets.MonthEnd(0)

        # 📅 전월 범위
        prev_start = cur_start - pd.DateOffset(months=1)
        prev_end = cur_start - pd.Timedelta(days=1)

        # 🔄 전체 데이터 로드
        df_full = final_df
        df_full["측정일시"] = pd.to_datetime(df_full["측정일시"], errors="coerce")

        # 🔎 컬럼 확인
        usage_col = next((col for col in df_full.columns if '전력사용량' in col), None)
        cost_col = next((col for col in df_full.columns if '전기요금' in col), None)

        if not usage_col or not cost_col:
            return "전력사용량/요금 컬럼 없음"

        #  집계
        cur = df_full[(df_full["측정일시"] >= cur_start) & (df_full["측정일시"] <= cur_end)]
        prev = df_full[(df_full["측정일시"] >= prev_start) & (df_full["측정일시"] <= prev_end)]

        cur_usage = cur[usage_col].sum() if not cur.empty else 0
        cur_cost = cur[cost_col].sum() if not cur.empty else 0
        prev_usage = prev[usage_col].sum() if not prev.empty else cur_usage
        prev_cost = prev[cost_col].sum() if not prev.empty else cur_cost

        # 📈 증감률 계산
        usage_rate = (cur_usage - prev_usage) / prev_usage * 100 if prev_usage else 0
        cost_rate = (cur_cost - prev_cost) / prev_cost * 100 if prev_cost else 0

        # 🎨 화살표
        def format_rate(rate):
            arrow = "🔺" if rate > 0 else "🔻"
            return f"{arrow} {rate:+.1f}%"

        return (
            f" 전력사용량: {format_rate(usage_rate)}\n"
            f" 전기요금: {format_rate(cost_rate)}"
    )


    @output
    @render.download(filename="LS_Electric_보고서.docx")
    def download_report():
        import pandas as pd
        import tempfile
        from datetime import timedelta

        # 1) 이번 달 데이터 불러오기
        d = summary_data()
        if d.empty:
            raise ValueError("📂 데이터 없음")

        # 2) 차트 저장
        fig1 = make_work_type_pie(d)
        fig2 = make_cost_trend_chart(d, input.aggregation_unit())
        fig3 = make_monthly_summary_chart(final_df, input.selected_month())
        fig4 = make_comparison_chart(final_df, input.selected_month(), "usage")

        img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img4 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

        fig1.write_image(img1, width=600, height=300)
        fig2.write_image(img2, width=600, height=300)
        fig3.write_image(img3, width=600, height=300)
        fig4.write_image(img4, width=600, height=300)

        # 3) 이번 달/이전 달 기간 계산
        sel_month  = input.selected_month()
        this_start = pd.to_datetime(f"{sel_month}-01")
        prev_start = (this_start - pd.offsets.MonthEnd(1)).replace(day=1)

        # 4) 이전 달 데이터 합계
        df_full = final_df.copy()
        df_full["측정일시"] = pd.to_datetime(df_full["측정일시"], errors="coerce")
        prev_df = df_full[
            (df_full["측정일시"] >= prev_start) &
            (df_full["측정일시"] <  this_start)
        ]
        prev_usage = prev_df["전력사용량(kWh)"].sum()
        prev_cost  = prev_df["전기요금(원)"].sum()

        # 5) 주요 지표
        peak_cost_info      = get_peak_cost_info(d)
        avg_carbon_info     = get_avg_carbon_info(d)
        main_work_type_info = get_main_work_type_info(d)
        monthly_change_info = get_monthly_change_info(d)

        # 6) summary_plain 생성 (dashboard와 동일 로직)
        usage       = d["전력사용량(kWh)"].sum()
        cost        = d["전기요금(원)"].sum()
        peak_time   = pd.to_datetime(d.loc[d["전기요금(원)"].idxmax(), "측정일시"])
        d["hour"]   = d["측정일시"].dt.hour
        d["is_night"] = d["hour"].between(20,23) | d["hour"].between(0,5)
        night_ratio = d["is_night"].mean()
        usage_rate  = (usage - prev_usage) / prev_usage * 100 if prev_usage else 0
        cost_rate   = (cost  - prev_cost ) / prev_cost  * 100 if prev_cost  else 0
        anomaly_flag = abs(usage_rate)>15 or abs(cost_rate)>20 or night_ratio>0.6 or night_ratio<0.2

        summary_plain = f"""🧾 이번 달 리포트 요약
        - 전력사용량: {usage:,.0f} kWh
        - 전기요금: ₩{cost:,.0f}
        - 전월 대비 사용량 {usage_rate:+.1f}% / 요금 {cost_rate:+.1f}%
        - 피크 요금 시간: {peak_time:%Y-%m-%d %H:%M}
        - 야간 사용 비율: {night_ratio*100:.1f}%
        {"⚠️ 이상 징후 관측됨" if anomaly_flag else "✅ 이상 징후 없음"}"""

        # 7) context에 합치기
        context = {
            "customer_name":        "홍길동",
            "billing_month":        this_start.strftime("%m"),
            "customer_id":          "LS202405-01",
            "total_cost":           f"₩{cost:,.0f}",
            "usage_period":         f"{d['측정일시'].min():%Y-%m-%d} ~ {d['측정일시'].max():%Y-%m-%d}",
            "main_work_type":       d["작업유형"].mode().iloc[0],
            "previous_month":       prev_start.strftime("%m"),
            "current_usage":        f"{usage:,.1f} kWh",
            "previous_usage":       f"{prev_usage:,.1f} kWh",
            "address":              "서울시 강남구 역삼동…",
            "previous_total_cost":  f"₩{prev_cost:,.0f}",
            "contract_type":        "일반용 저압",
            "peak_cost_info":       peak_cost_info,
            "avg_carbon_info":      avg_carbon_info,
            "main_work_type_info":  main_work_type_info,
            "monthly_change_info":  monthly_change_info,
            "report_summary_text":  summary_plain,
            "graph1_path":          img1,
            "graph2_path":          img2,
            "graph3_path":          img3,
            "graph4_path":          img4,
        }

        # 8) 보고서 생성
        report_path = generate_report(context)
        return open(report_path, "rb")

# 앱 실
app = App(app_ui, server)
