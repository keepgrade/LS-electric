from shiny import App, render, ui, reactive
from shinywidgets import render_widget, output_widget
from plotly.graph_objects import FigureWidget
from pandas.tseries.offsets import Week
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings("ignore")


# 데이터 로드
def load_data():
    try:
        df = pd.read_csv('../data/train.csv')
        # 날짜 컬럼이 있다면 datetime으로 변환
        if '측정일시' in df.columns:
            df['측정일시'] = pd.to_datetime(df['측정일시'])
        elif 'datetime' in df.columns:
            df['측정일시'] = pd.to_datetime(df['datetime'])
            
        # 필요한 컬럼들이 없으면 생성
        if '전력사용량' not in df.columns and 'power_usage' in df.columns:
            df['전력사용량'] = df['power_usage']
        if '전기요금' not in df.columns and 'cost' in df.columns:
            df['전기요금'] = df['cost']
        if '탄소배출량' not in df.columns and 'co2' in df.columns:
            df['탄소배출량'] = df['co2']
        if '작업유형' not in df.columns:
            df['작업유형'] = np.random.choice(['Light_Load', 'Heavy_Load'], len(df))
            
        return df
    except FileNotFoundError:
        # 테스트용 더미 데이터 생성
        dates = pd.date_range(start='2025-05-01', end='2025-06-30', freq='H')
        df = pd.DataFrame({
            '측정일시': dates,
            '전력사용량': np.random.normal(341203, 50000, len(dates)),
            '전기요금': np.random.normal(120327, 20000, len(dates)),
            '탄소배출량': np.random.normal(328, 30, len(dates)),
            '작업유형': np.random.choice(['Light_Load', 'Heavy_Load'], len(dates))
        })
        return df

df = load_data()

<<<<<<< HEAD
=======


>>>>>>> b289ecf (Merge pull request #8 from P-fe/main)
class Streamer:
    def __init__(self, df):
        self.df = df.sort_values("측정일시").reset_index(drop=True)
        self.index = 0

    def get_next_batch(self, n=1):
        if self.index >= len(self.df):
            return None
        batch = self.df.iloc[self.index:self.index + n]
        self.index += n
        return batch

    def get_current_data(self):
        return self.df.iloc[:self.index].copy()


class Accumulator:
    def __init__(self):
        self.df = pd.DataFrame()

    def accumulate(self, batch):
        self.df = pd.concat([self.df, batch], ignore_index=True)

    def get(self):
        return self.df.copy()


<<<<<<< HEAD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_predicted_december_data.csv'))
test_df["측정일시"] = pd.to_datetime(test_df["측정일시"])  # 반드시 datetime으로 변환



print("✅ test_df info")
print(test_df.info())      # dtype, 결측치, 행수 확인
print(test_df.head())      # 샘플 확인

=======
# 기준값 계산 함수
def get_november_baseline(train_df):
    nov_df = train_df[
        (train_df["측정일시"] >= "2024-11-01") &
        (train_df["측정일시"] < "2024-12-01")
    ].copy()

    # 누적 기준값 (평균 아님!)
    daily_total = nov_df.groupby(nov_df["측정일시"].dt.date)["전력사용량"].sum().mean()
    weekly_total = nov_df.groupby(nov_df["측정일시"].dt.to_period("W"))["전력사용량"].sum().mean()
    monthly_total = nov_df["전력사용량"].sum()


    cost_daily_total = nov_df.groupby(nov_df["측정일시"].dt.date)["전기요금"].sum().mean()
    cost_weekly_total = nov_df.groupby(nov_df["측정일시"].dt.to_period("W"))["전기요금"].sum().mean()
    cost_monthly_total = nov_df["전기요금"].sum()

    return {
        "power": {
            "daily": daily_total,
            "weekly": weekly_total,
            "monthly": monthly_total,
        },
        "cost": {
            "daily": cost_daily_total,
            "weekly": cost_weekly_total,
            "monthly": cost_monthly_total,
        }
    }


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

final_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'df_final.csv'))
test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'test_predicted_december_data.csv'))
test_df["측정일시"] = pd.to_datetime(test_df["측정일시"])  # 반드시 datetime으로 변환
train_df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'train.csv'))
train_df["측정일시"] = pd.to_datetime(train_df["측정일시"])
if "전력사용량(kWh)" in train_df.columns:
    train_df["전력사용량"] = train_df["전력사용량(kWh)"]
if "전기요금(원)" in train_df.columns:
    train_df["전기요금"] = train_df["전기요금(원)"]
if "탄소배출량(tCO2)" in train_df.columns:
    train_df["탄소배출량"] = train_df["탄소배출량(tCO2)"]

nov_baseline = get_november_baseline(train_df)


>>>>>>> b289ecf (Merge pull request #8 from P-fe/main)
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
<<<<<<< HEAD
=======
                    ui.br(),
                    ui.input_slider(
                        "update_interval",
                        "🔄 업데이트 간격 (초):",
                        min=0.1, max=5, value=1, step=0.1
                    ),
>>>>>>> b289ecf (Merge pull request #8 from P-fe/main)
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
<<<<<<< HEAD
                                    ui.input_action_button("toggle_streaming", "⏯️ 스트리밍 시작 / 중지", class_="btn btn-primary"),
                                    ui.input_action_button("update_chart", "예측시작", class_="btn-primary"),
=======
                                    ui.input_action_button("update_chart", "예측 시작", class_="btn-primary"),
>>>>>>> b289ecf (Merge pull request #8 from P-fe/main)
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
            ui.input_action_button("download_pdf", "📄 PDF 보고서 다운로드", class_="btn-success btn-lg"),
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
<<<<<<< HEAD
    @reactive.effect
    def toggle_streaming_state():
        if input.toggle_streaming():
            current = is_streaming.get()
            is_streaming.set(not current)
            print(f"🚦 스트리밍 {'시작' if not current else '중지'}됨")

=======
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
    
>>>>>>> b289ecf (Merge pull request #8 from P-fe/main)

    # ───────────────────────────────────────────────────────
    # 1) Reactive 데이터 준비 (분석 보고서 탭)
    # ───────────────────────────────────────────────────────
<<<<<<< HEAD

=======
>>>>>>> b289ecf (Merge pull request #8 from P-fe/main)
    @reactive.Calc
    def summary_data():
        # 📂 CSV 로드
        base_dir = os.path.dirname(__file__)
        file_path = os.path.abspath(os.path.join(base_dir, "..", "data", "df_final.csv"))
        df_final = pd.read_csv(file_path)

        # ✅ datetime 형변환 강제 수행
        if "측정일시" not in df_final.columns:
            raise KeyError("❌ '측정일시' 컬럼이 없습니다.")
        df_final["측정일시"] = pd.to_datetime(df_final["측정일시"], errors="coerce")

        df2 = df_final.copy()

        # ✅ 날짜 필터 안전하게 적용
        try:
            selected_month = input.selected_month()  # 예: "2024-05"
            start = pd.to_datetime(selected_month + "-01")
            end = start + pd.offsets.MonthEnd(0)  # 말일 계산

            df2 = df2[(df2["측정일시"] >= start) & (df2["측정일시"] <= end)]
        except Exception as e:
            print("⛔ 날짜 필터 적용 중 오류:", e)

        return df2

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
<<<<<<< HEAD
            cutoff = now - timedelta(days=1)
        return df[df["측정일시"] >= cutoff].copy()
    
    streamer = reactive.Value(Streamer(test_df))
    accumulator = reactive.Value(Accumulator())
    is_streaming = reactive.Value(True)
    current_data = reactive.Value(pd.DataFrame())



    @reactive.effect
    def stream_data():
        try:
            if not is_streaming.get():
                return

            # ⏱️ 업데이트 간격 (초 단위)
            interval_sec = input.update_interval() if hasattr(input, "update_interval") else 1
            reactive.invalidate_later(interval_sec)

            s = streamer.get()
            next_batch = s.get_next_batch(1)

            if next_batch is not None:
                accumulator.get().accumulate(next_batch)
                current_data.set(accumulator.get().get())
                print(f"📡 Streaming: index={s.index}, batch={len(next_batch)}")
            else:
                print("✅ 스트리밍 완료")
                is_streaming.set(False)

        except Exception as e:
            print("⛔ 오류 발생:", e)
            is_streaming.set(False)

    # ───────────────────────────────────────────────────────
    # 2) [A] 요약 카드 (실시간 탭)
    # ───────────────────────────────────────────────────────
    @output
    @render.ui
    def card_power():
        d = simulated_data()
        val = d["전력사용량"].iloc[-1] if not d.empty else 0
        return ui.div(
            ui.div(f"{val:,.0f}", class_="metric-value"),
            ui.div("kWh", class_="metric-label"),
            class_="metric-card",
        )

    @output
    @render.ui
    def card_cost():
        d = simulated_data()
        val = d["전기요금"].iloc[-1] if not d.empty else 0
        return ui.div(
            ui.div(f"{val:,.0f}", class_="metric-value"),
            ui.div("원", class_="metric-label"),
            class_="metric-card",
        )

    @output
    @render.ui
    def card_co2():
        d = simulated_data()
        val = d["탄소배출량"].iloc[-1] if not d.empty else 0
        return ui.div(
            ui.div(f"{val:,.0f}", class_="metric-value"),
            ui.div("CO₂", class_="metric-label"),
            class_="metric-card",
        )

    @output
    @render.ui
    def card_pf():
        return ui.div(
            ui.div("0.95", class_="metric-value"),
            ui.div("PF", class_="metric-label"),
            class_="metric-card",
        )

    @output
    @render.ui
    def card_work_type():
        d = simulated_data()
        typ = d["작업유형"].mode().iloc[0] if not d.empty else "N/A"
        return ui.div(
            ui.div(typ, class_="metric-value", style="font-size:18px;"),
            ui.div("작업유형", class_="metric-label"),
            class_="metric-card",
        )

    @output
    @render.ui
    def card_weather():
        return ui.div(
            ui.div("31°C", class_="metric-value"),
            ui.div("날씨", class_="metric-label"),
            class_="metric-card",
        )

    # ───────────────────────────────────────────────────────
    # 3) [B] 실시간 그래프
=======
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
>>>>>>> b289ecf (Merge pull request #8 from P-fe/main)
    # ───────────────────────────────────────────────────────
    @output
    @render_widget
    def realtime_chart():
<<<<<<< HEAD
        d = simulated_data()
        
        if d.empty or len(d) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="📭 표시할 데이터가 없습니다",
                x=0.5, y=0.5, showarrow=False, font=dict(size=20),
                xref="paper", yref="paper"
            )
            fig.update_layout(height=400)
            return fig

        # 샘플링: 최대 100개로 제한
        sample = d.iloc[:: max(1, len(d)//100)]
        
        # 차트 타입 선택
        chart_type = input.chart_type()
        Trace = go.Scatter if chart_type == "line" else go.Bar

        # 시각화 시작
        fig = go.Figure()

        if "전력사용량" in input.metrics_select():
            fig.add_trace(Trace(
                x=sample["측정일시"],
                y=sample["전력사용량"],
                name="전력사용량",
                yaxis="y",
                marker_color="#3498db"
            ))

        if "전기요금" in input.metrics_select():
            fig.add_trace(Trace(
                x=sample["측정일시"],
                y=sample["전기요금"],
                name="전기요금",
                yaxis="y2",
                marker_color="#e74c3c"
            ))

        # 레이아웃 업데이트
        fig.update_layout(
            title="📡 실시간 전력사용량 & 전기요금",
            xaxis=dict(title="시간", tickformat="%m-%d %H:%M"),
            yaxis=dict(title="전력사용량 (kWh)", side="left"),
            yaxis2=dict(title="전기요금 (원)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(t=60, b=40, l=50, r=50),
            height=400,
        )

        return fig
=======
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

>>>>>>> b289ecf (Merge pull request #8 from P-fe/main)


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
        """1년간 월별 전력사용량 + 평균요금 추이 (날짜 필터 무시)"""
        try:
            # 🔹 summary_data는 날짜 필터가 적용되므로 load_data로 전체 불러옴
            base_dir = os.path.dirname(__file__)
            file_path = os.path.abspath(os.path.join(base_dir, "..", "data", "df_final.csv"))
            df = pd.read_csv(file_path)
            df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")

            # 🔹 최근 1년 필터링
            latest = df["측정일시"].max()
            one_year_ago = latest - pd.DateOffset(years=1)
            df = df[(df["측정일시"] >= one_year_ago) & (df["측정일시"] <= latest)]

            # 🔹 월별 집계
            monthly = (
                df.groupby(df["측정일시"].dt.to_period("M"))
                .agg({
                    "전력사용량(kWh)": "sum",
                    "전기요금(원)": "mean"
                })
                .reset_index()
            )
            monthly["측정월"] = monthly["측정일시"].dt.to_timestamp()
            monthly["측정월_라벨"] = monthly["측정월"].dt.strftime("%Y-%m")

            # 🔴 선택한 월에만 빨간색
            selected = input.selected_month()  # 예: "2024-05"
            monthly["막대색"] = np.where(monthly["측정월_라벨"] == selected, "red", "gray")

            # 🔹 Plotly 그리기
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=monthly["측정월_라벨"],
                y=monthly["전력사용량(kWh)"],
                name="전력사용량(kWh)",
                marker_color=monthly["막대색"],
                yaxis="y1"
            ))

            fig.add_trace(go.Scatter(
                x=monthly["측정월_라벨"],
                y=monthly["전기요금(원)"],
                name="평균요금(원)",
                mode="lines+markers",
                line=dict(color="black"),
                yaxis="y2"
            ))

            # ✅ 범례용 빨간 막대 더미 추가
            fig.add_trace(go.Bar(
                x=[None],
                y=[None],
                name="현재 분석 달",
                marker_color="red",
                showlegend=True
            ))


            fig.update_layout(
                title="1년간 월별 전력사용량 및 평균요금 추이",
                xaxis=dict(title="월", type="category"),
                yaxis=dict(title="전력사용량 (kWh)", side="left"),
                yaxis2=dict(title="평균요금 (원)", side="right", overlaying="y", showgrid=False),
                height=450,
                plot_bgcolor="white",
                margin=dict(t=60, b=60, l=60, r=60)
            )

            return fig

        except Exception as e:
            return create_simple_error_chart(f"오류 발생: {e}")
        
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
        d = summary_data()
        if d.empty:
            return "데이터 없음" ###

        cur_sum = d["전기요금(원)"].sum()
        min_date = d["측정일시"].min()
        prev_cutoff = min_date - timedelta(days=30)

        # ✅ 같은 소스로부터 전체 데이터 재로딩 (df 대신)
        base_dir = os.path.dirname(__file__)
        file_path = os.path.abspath(os.path.join(base_dir, "..", "data", "df_final.csv"))
        df_full = pd.read_csv(file_path)
        df_full["측정일시"] = pd.to_datetime(df_full["측정일시"], errors="coerce")

        prev = df_full[(df_full["측정일시"] >= prev_cutoff) & (df_full["측정일시"] < min_date)]
        prev_sum = prev["전기요금(원)"].sum() if not prev.empty else cur_sum

        rate = (cur_sum - prev_sum) / prev_sum * 100 if prev_sum else 0
        return f"{rate:+.1f}%"
    


# 앱 실행
app = App(app_ui, server)

if __name__ == "__main__":
    app.run( port=8000)