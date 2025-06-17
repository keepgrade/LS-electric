from shiny import App, render, ui, reactive
from shinywidgets import render_widget, output_widget
from plotly.graph_objects import FigureWidget
from pandas.tseries.offsets import Week
from pathlib import Path
import os
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import plotly.graph_objects as go
from datetime import datetime, timedelta
from generate_report import generate_report
import warnings
warnings.filterwarnings("ignore")
import asyncio
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
            # 여기서 출력될 기본 크기를 크게 잡습니다
            width=600,    # 차트 내부 크기: 600px
            height=600    # 차트 내부 크기: 600px
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
        marker=dict(color="gray", line=dict(color="darkgray", width=0.5)),
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
            f"📊 요약 통계<br>"
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
        end = start + pd.offsets.MonthEnd(0)
        sel = df_[(df_[date_col] >= start) & (df_[date_col] <= end)]
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


# CSS 스타일 정의
css_style = """
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 5px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

    /* 👇 아래 두 줄 추가로 크기 통일 */
    width: 190px;
    height: 130px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.metric-card2 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 5px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

    /* 👇 아래 두 줄 추가로 크기 통일 */
    width: 240px;
    height: 130px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
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
    background-color: #2c3e50;
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
</style>
"""

# UI 정의
app_ui = ui.page_navbar(
    ui.nav_panel("실시간 모니터링",
        ui.HTML(css_style),
        ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    ui.h4("📊 실시간 모니터링", style="color: #2c3e50; margin-bottom: 20px;"),
                    ui.input_date_range(
                        "date_range_monitoring",
                        "📅 기간 선택:",
                        start=test_df["측정일시"].min().strftime("%Y-%m-%d"),
                        end=test_df["측정일시"].max().strftime("%Y-%m-%d"),
                        format="yyyy-mm-dd"
                    ),
                    ui.br(),
                    ui.input_selectize(
                        "metrics_select",
                        "📈 표시할 지표:",
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
                    ui.h4("[A] 요약 카드 : 실시간 전력사용량, 이용료, 탄소배출량, 평균 PF, 작업 유형", 
                          class_="section-header"),
                    ui.div(
                        ui.row(
                            ui.column(2, ui.output_ui("card_power")),
                            ui.column(2, ui.output_ui("card_cost")),
                            ui.column(2, ui.output_ui("card_co2")),
                            ui.column(2, ui.output_ui("card_pf")),
                            ui.column(2, ui.output_ui("card_work_type")),
                            ui.column(2, ui.output_ui("card_weather"))
                        ),
                        class_="chart-container"
                    )
                ),
                
                # [B] A번수 실시간 그래프 섹션
                ui.div(
                    ui.h4("[B] A번수 실시간 그래프", class_="section-header"),
                    ui.div(
                        ui.row(
                            ui.column(12, output_widget("realtime_chart")),
                        ),
                        class_="chart-container"
                    )
                ),
                
                # [C] 전력사용량/료 슬라이더 섹션
                ui.div(
                    ui.h4("[C] 전력사용량/료 슬라이더 : 전력 실시간 및 누적 (일/주/월)", 
                          class_="section-header"),
                    ui.div(
                        ui.row(
                            ui.column(6,
                                ui.div(
                                    ui.h5("🔋 실시간 누적 전력사용량", style="color: #2c3e50;"),
                                    ui.output_ui("power_progress_bars"),
                                    style="padding: 15px;"
                                )
                            ),
                            ui.column(6,
                                ui.div(
                                    ui.h5("💰 실시간 누적 전기요금", style="color: #2c3e50;"),
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
                    ui.h4("[D] 그래프/주/월/시간대별 작업 유형 분포", class_="section-header"),
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
    ui.h4("[A]📋 기간별 전력 사용 요약", class_="section-header"),
    ui.div(
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    id="selected_month",
                    label="📅 분석 월 선택:",
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
        ui.h4("[B]📈 전력 요금 시계열 분석", class_="section-header"),
        ui.div(
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_radio_buttons(
                        id="aggregation_unit",
                        label="🕒 집계 단위 선택:",
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
        ui.h4("[C]📊 상세 분석 정보", class_="section-header"),
        ui.div(
            ui.layout_column_wrap(
                ui.card(ui.card_header("💰 최고 요금 정보"), ui.output_text("peak_cost_info")),
                ui.card(ui.card_header("🌿 평균 탄소배출량"), ui.output_text("avg_carbon_info")),
                ui.card(ui.card_header("⚙️ 주요 작업 유형"), ui.output_text("main_work_type_info")),
                ui.card(ui.card_header("📊 전월 대비 증감률"), ui.output_text("monthly_change_info")),
                width=1/2
            ),
            class_="chart-container"
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
        ui.h3("📚 부록")
    ),

    title="⚡ LS Electric 전기요금 실시간 모니터링",
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
                raise KeyError("❌ '측정일시' 컬럼이 없습니다.")
            df_final["측정일시"] = pd.to_datetime(df_final["측정일시"], errors="coerce")

            df2 = df_final.copy()

            selected_month = input.selected_month()
            if not selected_month:
                print("⛔ 선택된 월 없음. 기본값 반환")
                return df2

            start = pd.to_datetime(selected_month + "-01")
            end = start + pd.offsets.MonthEnd(0)
            df2 = df2[(df2["측정일시"] >= start) & (df2["측정일시"] <= end)]

            return df2

        except Exception as e:
            print(f"❌ summary_data() 실행 중 오류: {e}")
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
            print("📅 date_range input:", start_raw, "~", end_raw)

            start_raw, end_raw = input.date_range()
            if not start_raw or not end_raw:
                print("⚠️ 날짜 범위 미지정")
                return pd.DataFrame()

            start = pd.to_datetime(start_raw)
            end = pd.to_datetime(end_raw) + timedelta(days=1)
            df_final = df_final[(df_final["측정일시"] >= start) & (df_final["측정일시"] < end)]
            
            print("✅ 필터링 완료:", df_final.shape)
            return df_final

        except Exception as e:
            print("❌ report_summary_data 오류:", e)
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
            class_="metric-card"
    )

    @output
    @render.ui
    def card_cost():
        d = simulated_data()
        val = d["전기요금"].iloc[-1] if not d.empty else 0
        return ui.div(ui.div(f"{val:,.0f}", class_="metric-value"), ui.div("원", class_="metric-label"), class_="metric-card")

    @output
    @render.ui
    def card_co2():
        d = simulated_data()
        val = d["탄소배출량"].iloc[-1] if not d.empty else 0
        return ui.div(ui.div(f"{val:,.0f}", class_="metric-value"), ui.div("CO₂", class_="metric-label"), class_="metric-card")

    @output
    @render.ui
    def card_pf():
        return ui.div(ui.div("0.95", class_="metric-value"), ui.div("PF", class_="metric-label"), class_="metric-card")

    @output
    @render.ui
    def card_work_type():
        d = simulated_data()
        typ = d["작업유형"].mode().iloc[0] if not d.empty else "N/A"
        return ui.div(
        ui.div(typ, class_="metric-value", style="font-size:18px; word-break:break-word;"),
        ui.div("작업유형", class_="metric-label"),
        class_="metric-card"
    )

    @output
    @render.ui
    def card_weather():
        return ui.div(ui.div("31°C", class_="metric-value"), ui.div("날씨", class_="metric-label"), class_="metric-card")


    # ───────────────────────────────────────────────────────
    # 4) [C] 진행률 바 공통 함수 및 렌더링
    # ───────────────────────────────────────────────────────
    def _make_bar(label, val, denom, color):
        pct = min(100, val / denom * 100) if denom else 0
        return ui.div(
            # 상단 텍스트
            ui.div(f"{label}: {val:,.0f} ({pct:.1f}%) / 기준: {denom:,.0f}", style="font-weight:bold; margin-bottom:4px;"),
            
            # ✅ 배경 바 (100%)
            ui.div(
                # ✅ 채워지는 부분 (겹쳐진 div)
                ui.div(style=f"width:{pct:.1f}%; height:12px; background:{color}; border-radius:4px;"),
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
            _make_bar("일일 누적", day_usage, nov_baseline["power"]["daily"], "#3498db"),
            _make_bar("주별 누적", week_usage, nov_baseline["power"]["weekly"], "#9b59b6"),
            _make_bar("월별 누적", month_usage, nov_baseline["power"]["monthly"], "#e67e22"),
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
            _make_bar("일일 누적", day_cost, nov_baseline["cost"]["daily"], "#27ae60"),
            _make_bar("주별 누적", week_cost, nov_baseline["cost"]["weekly"], "#f39c12"),
            _make_bar("월별 누적", month_cost, nov_baseline["cost"]["monthly"], "#c0392b"),
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

        fig = go.Figure()
        for col in hourly_ratio.columns:
            fig.add_trace(go.Bar(x=hourly_ratio.index, y=hourly_ratio[col], name=col))

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
        return px.pie(values=cnt.values, names=cnt.index,
                      title="작업유형별 분포", height=300)
    
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
            class_="metric-card2"
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
            class_="metric-card2"
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
            class_="metric-card2"
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
            class_="metric-card2"
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
                marker=dict(color='gray', line=dict(color='darkgray', width=0.5)),
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
                text=f"<b>📊 요약 통계</b><br>" +
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
        try:
            # ✅ 현재 선택된 월의 데이터
            d = summary_data()
            if d.empty or "측정일시" not in d.columns or "전기요금(원)" not in d.columns:
                return "📭 데이터 없음"

            cur_sum = d["전기요금(원)"].sum()
            min_date = d["측정일시"].min()

            if pd.isna(min_date):
                return "⛔ 날짜 정보 없음"

            prev_cutoff = min_date - timedelta(days=30)

            # ✅ 전체 데이터 재로딩
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, "data", "df_final.csv")

            if not os.path.exists(file_path):
                return "❌ 전체 데이터 파일을 찾을 수 없습니다."

            df_full = pd.read_csv(file_path)
            if "측정일시" not in df_full.columns or "전기요금(원)" not in df_full.columns:
                return "❌ 필요한 컬럼이 없습니다."

            df_full["측정일시"] = pd.to_datetime(df_full["측정일시"], errors="coerce")
            df_prev = df_full[(df_full["측정일시"] >= prev_cutoff) & (df_full["측정일시"] < min_date)]

            prev_sum = df_prev["전기요금(원)"].sum() if not df_prev.empty else cur_sum
            rate = (cur_sum - prev_sum) / prev_sum * 100 if prev_sum != 0 else 0

            return f"{rate:+.1f}%"

        except Exception as e:
            print(f"❌ monthly_change_info() 오류: {e}")
            return "⚠️ 분석 중 오류 발생"


    @output
    @render.download(filename="LS_Electric_보고서.docx")
    def download_report():
        # 1) summary_data() 를 사용해 현재 선택 월 데이터 가져오기
        d = summary_data()
        if d.empty:
            raise ValueError("📂 데이터 없음")
        

        # 2) 차트 생성용 원본·파라미터
        current_df = d.copy()                  # 실 데이터를 쓰는 df
        sel_month  = input.selected_month()    # "2024-05" 형식
        agg_unit   = input.aggregation_unit()  # "hour"/"day"/"weekday"

        # 3) 각 차트 함수 호출
        fig1 = make_work_type_pie(summary_data())  
        fig2 = make_cost_trend_chart(summary_data(), input.aggregation_unit())
        fig3 = make_monthly_summary_chart(final_df, input.selected_month())
        fig4 = make_comparison_chart(final_df, sel_month, "usage")




        # 4) 임시 파일 경로 생성
        import tempfile
        img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img4 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name  # 예시로 동일한 이미지 사용

        # 5) 차트 PNG 로 저장
        fig1.write_image(img1, width=600, height=300)
        fig2.write_image(img2, width=600, height=300)
        fig3.write_image(img3, width=600, height=300)
        fig4.write_image(img4, width=600, height=300)  # 예시로 동일한 이미지 사용

        # 6) 동적으로 값 계산 (예: 최고 요금, 평균 탄소배출량 등)
        peak_cost_info      = get_peak_cost_info(d)
        avg_carbon_info     = get_avg_carbon_info(d)
        main_work_type_info = get_main_work_type_info(d)
        monthly_change_info = get_monthly_change_info(d)
        
        # 7) 워드 템플릿에 넘길 context 구성
        context = {
            "customer_name":      "홍길동",
            "billing_month":      sel_month.split("-")[1],
            "customer_id":        "LS202405-01",
            "total_cost":         f"{d['전기요금(원)'].sum():,.0f} 원",
            "usage_period":       f"{d['측정일시'].min():%Y-%m-%d} ~ {d['측정일시'].max():%Y-%m-%d}",
            "main_work_type":     d["작업유형"].mode().iloc[0],
            "previous_month":     f"{(pd.to_datetime(sel_month + '-01') - pd.DateOffset(months=1)):%m}",
            "current_usage":      f"{d['전력사용량(kWh)'].sum():,.1f} kWh",
            "previous_usage":     "…",  # 필요 시 계산
            "address":            "서울시 강남구 역삼동…",
            "previous_total_cost":"…",  # 필요 시 계산
            "contract_type":      "일반용 저압",
            "peak_cost_info": peak_cost_info,
            "avg_carbon_info": avg_carbon_info,
            "main_work_type_info": main_work_type_info,
            "monthly_change_info": monthly_change_info,
            # 차트 경로
            "graph1_path": img1,
            "graph2_path": img2,
            "graph3_path": img3,
            "graph4_path": img4  # 예시로 동일한 이미지 사용
        }

        # 7) 보고서 생성
        report_path = generate_report(context)
        return open(report_path, "rb")


# 앱 실행
app = App(app_ui, server)
