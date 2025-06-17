# app.py

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

# ───────────────────────────────────────────────────────
# 1) 경로 설정
# ───────────────────────────────────────────────────────
# app.py가 위치한 폴더를 기준으로 상대 경로 설정
BASE_DIR = Path(__file__).resolve().parent       # 👉 dashboard/
DATA_DIR = BASE_DIR / "data"                     # 👉 dashboard/data/

DF_FINAL = DATA_DIR / "df_final.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test_predicted_december_data.csv"


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
        return px.pie(title="작업유형별 분포 (데이터 없음)")
    cnt = df["작업유형"].value_counts()
    fig = px.pie(
        names=cnt.index,
        values=cnt.values,
        title="작업유형별 분포",
        height=300
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


def make_monthly_summary_chart(df_full, sel_month):
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    if df_full is None or df_full.empty:
        return go.Figure()

    df = df_full.copy()
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    latest = df["측정일시"].max()
    one_year_ago = latest - pd.DateOffset(years=1)
    df = df[(df["측정일시"] >= one_year_ago) & (df["측정일시"] <= latest)]

    grp = (
        df.groupby(df["측정일시"].dt.to_period("M"))
          .agg({"전력사용량":"sum","전기요금":"mean"})
          .reset_index()
    )
    grp["month_ts"] = grp["측정일시"].dt.to_timestamp()
    grp["label"]    = grp["month_ts"].dt.strftime("%Y-%m")
    grp["color"]    = np.where(grp["label"]==sel_month, "red", "gray")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["label"], y=grp["전력사용량"],
        name="전력사용량", marker_color=grp["color"], yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=grp["label"], y=grp["전기요금"],
        name="평균요금", mode="lines+markers",
        line=dict(color="black"), yaxis="y2"
    ))
    fig.add_trace(go.Bar(x=[None], y=[None], name="현재 분석 달", marker_color="red"))

    fig.update_layout(
        title="1년간 월별 전력사용량 및 평균요금 추이",
        xaxis=dict(type="category"),
        yaxis=dict(title="전력사용량 (kWh)", side="left"),
        yaxis2=dict(title="평균요금 (원)", overlaying="y", side="right", showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        height=450, margin=dict(t=60,b=60,l=60,r=60)
    )
    return fig




# ───────────────────────────────────────────────────────
# 헬퍼 함수: 월별 전력사용량 누적 + 평균 전기요금 차트
# ───────────────────────────────────────────────────────
def make_monthly_summary_chart(df_full, sel_month: str):
    import pandas as pd
    import plotly.graph_objects as go
    # 복사 & 날짜타입
    df = df_full.copy()
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    # 전력·요금 컬럼 자동 감지
    power_col = next((c for c in df.columns if "전력사용량" in c), None)
    cost_col  = next((c for c in df.columns if "전기요금" in c), None)
    if df.empty or power_col is None or cost_col is None:
        return go.Figure()

    # 최근 1년 데이터로 필터링
    latest = df["측정일시"].max()
    one_year_ago = latest - pd.DateOffset(years=1)
    df = df[(df["측정일시"] >= one_year_ago) & (df["측정일시"] <= latest)]

    # 월(period)단위 집계
    df["측정월"] = df["측정일시"].dt.to_period("M").dt.to_timestamp()
    agg = (
        df.groupby("측정월")
          .agg({power_col: "sum", cost_col: "mean"})
          .reset_index()
    )
    agg["측정월_라벨"] = agg["측정월"].dt.strftime("%Y-%m")
    # 현재 선택달만 빨간, 나머지 회색
    agg["color"] = ["red" if lab == sel_month else "gray" for lab in agg["측정월_라벨"]]

    # Plotly 그리기
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["측정월_라벨"],
        y=agg[power_col],
        name="월별 전력사용량",
        marker_color=agg["color"],
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=agg["측정월_라벨"],
        y=agg[cost_col],
        name="월별 평균요금",
        mode="lines+markers",
        yaxis="y2",
    ))
    fig.update_layout(
        title="최근 1년 월별 전력사용량 및 평균요금",
        xaxis=dict(title="월"),
        yaxis=dict(title="전력사용량", side="left"),
        yaxis2=dict(title="평균요금", side="right", overlaying="y"),
        height=350,
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig


# 1) 실제 데이터로부터 필요한 값을 계산
def get_peak_cost_info(df):
    # 최고 요금 정보: 전기요금이 가장 높은 행
    peak_row = df.loc[df["전기요금"].idxmax()]
    peak_cost = peak_row["전기요금"]
    peak_date = peak_row["측정일시"]
    return f"₩{peak_cost:,.0f} (발생일시: {peak_date:%Y-%m-%d %H:%M})"

def get_avg_carbon_info(df):
    # 평균 탄소배출량
    avg_carbon = df["탄소배출량"].mean()
    return f"{avg_carbon:.3f} tCO₂"

def get_main_work_type_info(df):
    # 가장 많은 작업유형
    main_work_type = df["작업유형"].mode().iloc[0]
    return main_work_type

def get_monthly_change_info(df, selected_month):
    # 전월 대비 증감률 계산
    current_sum = df["전기요금"].sum()
    
    # 전월 데이터 로드
    prev_month_start = pd.to_datetime(f"{selected_month}-01") - timedelta(days=1)
    prev_month_end = prev_month_start.replace(day=1)
    prev_month_data = df[(df["측정일시"] >= prev_month_start) & (df["측정일시"] < prev_month_end)]
    prev_sum = prev_month_data["전기요금"].sum()

    # 증감률 계산
    change_rate = (current_sum - prev_sum) / prev_sum * 100 if prev_sum else 0
    return f"{change_rate:+.1f}%"

# 2) 템플릿에 넣을 값 계산
def generate_report_with_dynamic_data(df, selected_month):
    # 데이터에서 동적으로 값 계산
    peak_cost_info = get_peak_cost_info(df)
    avg_carbon_info = get_avg_carbon_info(df)
    main_work_type_info = get_main_work_type_info(df)
    monthly_change_info = get_monthly_change_info(df, selected_month)

# ✅ 컬럼명 일괄 매핑
if "전력사용량(kWh)" in test_df.columns:
    test_df["전력사용량"] = test_df["전력사용량(kWh)"]
if "전기요금(원)" in test_df.columns:
    test_df["전기요금"] = test_df["전기요금(원)"]
if "탄소배출량(tCO2)" in test_df.columns:
    test_df["탄소배출량"] = test_df["탄소배출량(tCO2)"]


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
                    ui.input_radio_buttons(
                        "chart_type",
                        "📊 차트 유형:",
                        choices={
                            "line": "선형 그래프",
                            "bar": "막대 그래프"
                        },
                        selected="line"
                    ),
                    ui.br(),
                    ui.input_slider(
                        "update_interval",
                        "🔄 업데이트 간격 (초):",
                        min=0.1, max=5, value=1, step=0.1
                    ),
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
                            ui.column(8, output_widget("realtime_chart")),
                            ui.column(4, 
                                ui.div(
                                    ui.input_date_range(
                                        "chart_date_range",
                                        "기간 선택:",
                                        start=test_df["측정일시"].min().strftime("%Y-%m-%d"),
                                        end=test_df["측정일시"].max().strftime("%Y-%m-%d")
                                    ),
                                    ui.br(),
                                    ui.input_action_button("update_chart", "예측 시작", class_="btn-primary"),
                                    style="padding: 20px;"
                                )
                            )
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
                    )
                ),
                ui.layout_column_wrap(
                    ui.value_box(
                        title="누적 전력사용량",
                        value=ui.output_text("summary_power_usage"),
                        showcase="⚡",
                    ),
                    ui.value_box(
                        title="누적 전력요금",
                        value=ui.output_text("summary_power_cost"),
                        showcase="💰"
                    ),
                    ui.value_box(
                        title="누적 탄소배출량",
                        value=ui.output_text("summary_carbon_emission"),
                        showcase="🌱"
                    ),
                    ui.value_box(
                        title="평균 역률",
                        value=ui.output_text("summary_power_factor"),
                        showcase="⚙️",
                    ),
                    width=1/4
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
        ui.div(output_widget("monthly_summary_chart")),
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
            interval_ms = input.update_interval() * 1000
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
        interval_ms = int(input.update_interval() * 1000)
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
        return ui.div(ui.div(f"{val:,.0f}", class_="metric-value"), ui.div("kWh", class_="metric-label"), class_="metric-card")

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
            ui.div(f"{label}: {val:,.0f} ({pct:.1f}%) / 기준: {denom:,.0f}", style="font-weight:bold;"),
            ui.div(style=f"width:{pct:.1f}%; height:8px; background:{color}; border-radius:4px;"),
            style="margin:10px 0; padding:10px; background:white; border-radius:8px;"
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
    @render.text
    def summary_power_cost():
        d = summary_data()
        return f"₩{d['전기요금(원)'].sum():,.0f}" if not d.empty else "₩0"

    @output
    @render.text
    def summary_power_usage():
        d = summary_data()
        return f"{d['전력사용량(kWh)'].sum():,.1f} kWh" if not d.empty else "0.0 kWh"

    @output
    @render.text
    def summary_carbon_emission():
        d = summary_data()
        return f"{d['탄소배출량(tCO2)'].sum():,.1f} tCO₂" if not d.empty else "0.0 tCO₂"


    @output
    @render.text
    def summary_power_factor():
        d = summary_data()
        if d.empty:
            return "데이터 없음"
        
        pf1 = d["지상역률(%)"].mean() if "지상역률(%)" in d else None
        pf2 = d["진상역률(%)"].mean() if "진상역률(%)" in d else None
        
        txt = ""
        if pf1 is not None:
            txt += f"지상역률 평균: {pf1:.2f}%\n"
        if pf2 is not None:
            txt += f"진상역률 평균: {pf2:.2f}%"
        return txt.strip() or "역률 정보 없음"

    @output 
    @render_widget
    def cost_trend_chart():
        return make_cost_trend_chart(summary_data(), input.aggregation_unit())


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
        return make_monthly_summary_chart(final_df, input.selected_month())
            
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


        # 4) 임시 파일 경로 생성
        import tempfile
        img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

        # 5) 차트 PNG 로 저장
        fig1.write_image(img1, width=600, height=300)
        fig2.write_image(img2, width=600, height=300)
        fig3.write_image(img3, width=600, height=300)

        # 6) 동적으로 값 계산 (예: 최고 요금, 평균 탄소배출량 등)
        peak_cost_info = get_peak_cost_info(d)  # 최고 요금 정보 계산 함수
        avg_carbon_info = get_avg_carbon_info(d)  # 평균 탄소배출량 계산 함수
        main_work_type_info = get_main_work_type_info(d)  # 주요 작업 유형 계산 함수
        monthly_change_info = get_monthly_change_info(d, sel_month)  # 전월 대비 증감률 계산 함수


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
        }

        # 7) 보고서 생성
        report_path = generate_report(context)
        return open(report_path, "rb")





# 앱 실행
app = App(app_ui, server)
