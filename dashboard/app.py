from shiny import App, render, ui, reactive
from shinywidgets import render_widget, output_widget

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
                        start="2025-05-12",
                        end="2025-06-12",
                        format="yyyy-mm-dd"
                    ),
                    ui.br(),
                    ui.input_selectize(
                        "metrics_select",
                        "📈 표시할 지표:",
                        choices={
                            "전력사용량": "전력사용량 (kWh)", 
                            "전기요금": "전기요금 (원)",
                            "탄소배출량": "탄소배출량 (CO2)",
                            "작업유형": "작업유형"
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
                        min=1,
                        max=60,
                        value=15,
                        step=1
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
                                        start="2025-05-12",
                                        end="2025-06-12"
                                    ),
                                    ui.br(),
                                    ui.input_action_button("update_chart", "업데이트", class_="btn-primary"),
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
        ui.layout_column_wrap(
            ui.card(
                ui.card_header("📋 기간별 전력 사용 요약"),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_radio_buttons(
                            "summary_period",
                            "요약 기간:",
                            choices={
                                "15min": "최근 15분",
                                "30min": "최근 30분",
                                "1hour": "최근 1시간",
                                "today": "오늘",
                                "week": "이번주",
                                "month": "이번달"
                            },
                            selected="today"
                        )
                    ),
                    ui.layout_column_wrap(
                        ui.value_box(
                            title="누적 전력사용량",
                            value=ui.output_text("summary_power_usage"),
                            showcase="⚡"
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
                            showcase="⚙️"
                        ),
                        width=1/2
                    )
                )
            ),
            ui.card(
                ui.card_header("📈 전력 요금 시계열 분석"),
                output_widget("cost_trend_chart")
            ),
            ui.card(
                ui.card_header("📊 상세 분석 정보"),
                ui.layout_column_wrap(
                    ui.card(ui.card_header("최고 요금 정보"), ui.output_text("peak_cost_info")),
                    ui.card(ui.card_header("평균 탄소배출량"), ui.output_text("avg_carbon_info")),
                    ui.card(ui.card_header("주요 작업 유형"), ui.output_text("main_work_type_info")),
                    ui.card(ui.card_header("전월 대비 증감률"), ui.output_text("monthly_change_info")),
                    width=1/2
                ),
                ui.br(),
                ui.div(
                    ui.input_action_button("download_pdf", "📄 PDF 보고서 다운로드", class_="btn-success btn-lg"),
                    class_="text-center"
                )
            ),
            width=1
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

# 서버 로직
def server(input, output, session):

    # ───────────────────────────────────────────────────────
    # 1) Reactive 데이터 준비
    # ───────────────────────────────────────────────────────
    @reactive.Calc
    def filtered_data():
        """실시간 탭용: date_range_monitoring 필터 적용."""
        df2 = df.copy()
        dr = input.date_range_monitoring()
        if dr:
            start, end = pd.to_datetime(dr[0]).date(), pd.to_datetime(dr[1]).date()
            df2 = df2[(df2["측정일시"].dt.date >= start)
                    & (df2["측정일시"].dt.date <= end)]
        return df2

    @reactive.Calc
    def summary_data():
        """분석 보고서 탭용: summary_period 에 따라 원본 df 필터링."""
        period = input.summary_period()
        now = datetime.now()
        if period == "15min":
            cutoff = now - timedelta(minutes=15)
        elif period == "30min":
            cutoff = now - timedelta(minutes=30)
        elif period == "1hour":
            cutoff = now - timedelta(hours=1)
        elif period == "today":
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            cutoff = now - timedelta(days=7)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = now - timedelta(days=1)
        return df[df["측정일시"] >= cutoff].copy()

    # ───────────────────────────────────────────────────────
    # 2) [A] 요약 카드 (실시간 탭)
    # ───────────────────────────────────────────────────────
    @output
    @render.ui
    def card_power():
        d = filtered_data()
        val = d["전력사용량"].iloc[-1] if not d.empty else 0
        return ui.div(
            ui.div(f"{val:,.0f}", class_="metric-value"),
            ui.div("kWh", class_="metric-label"),
            class_="metric-card",
        )

    @output
    @render.ui
    def card_cost():
        d = filtered_data()
        val = d["전기요금"].iloc[-1] if not d.empty else 0
        return ui.div(
            ui.div(f"{val:,.0f}", class_="metric-value"),
            ui.div("원", class_="metric-label"),
            class_="metric-card",
        )

    @output
    @render.ui
    def card_co2():
        d = filtered_data()
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
        d = filtered_data()
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
    # ───────────────────────────────────────────────────────
    @output
    @render_widget
    def realtime_chart():
        d = filtered_data()
        if d.empty:
            return None
        sample = d.iloc[:: max(1, len(d)//100)]
        fig = go.Figure()
        if "전력사용량" in input.metrics_select():
            fig.add_trace(go.Scatter(x=sample["측정일시"], y=sample["전력사용량"],
                                     mode="lines", name="전력사용량"))
        if "전기요금" in input.metrics_select():
            fig.add_trace(go.Scatter(x=sample["측정일시"], y=sample["전기요금"],
                                     mode="lines", name="전기요금", yaxis="y2"))
        fig.update_layout(
            title="실시간 전력사용량 & 전기요금",
            xaxis_title="시간",
            yaxis=dict(title="kWh", side="left"),
            yaxis2=dict(title="원", overlaying="y", side="right"),
            hovermode="x unified", height=400,
        )
        return fig

    # ───────────────────────────────────────────────────────
    # 4) [C] 진행률 바 공통 함수 및 렌더링
    # ───────────────────────────────────────────────────────
    def _make_bar(label, val, denom, color):
        pct = min(100, val/denom*100) if denom else 0
        return ui.div(
            ui.div(f"{label}: {val:,.0f}", style="font-weight:bold;"),
            ui.div(style=f"width:{pct}%;height:8px;background:{color};border-radius:4px;"),
            style="margin:10px 0; padding:10px; background:white; border-radius:8px;",
        )

    @output
    @render.ui
    def power_progress_bars():
        d = filtered_data()
        if d.empty:
            return ui.div("데이터 없음")
        daily = d.groupby(d["측정일시"].dt.date)["전력사용량"].sum().tail(7)
        weekly = daily.sum()
        monthly = d.groupby(d["측정일시"].dt.to_period("M"))["전력사용량"].sum().iloc[-1]
        return ui.div(
            _make_bar("일일 누적", daily.iloc[-1], 1000, "#3498db"),
            _make_bar("주별 누적", weekly,         5000, "#9b59b6"),
            _make_bar("월별 누적", monthly,      20000, "#e67e22"),
        )

    @output
    @render.ui
    def cost_progress_bars():
        d = filtered_data()
        if d.empty:
            return ui.div("데이터 없음")
        daily = d.groupby(d["측정일시"].dt.date)["전기요금"].sum().tail(7)
        weekly = daily.sum()
        monthly = d.groupby(d["측정일시"].dt.to_period("M"))["전기요금"].sum().iloc[-1]
        return ui.div(
            _make_bar("일일 누적", daily.iloc[-1],  500000, "#27ae60"),
            _make_bar("주별 누적", weekly,         2000000, "#f39c12"),
            _make_bar("월별 누적", monthly,       8000000, "#c0392b"),
        )

    # ───────────────────────────────────────────────────────
    # 5) [D] 작업 유형 분포
    # ───────────────────────────────────────────────────────
    @output
    @render_widget
    def work_type_chart():
        d = filtered_data()
        if d.empty:
            return None
        hourly = d.groupby([d["측정일시"].dt.hour, "작업유형"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for t in hourly.columns:
            fig.add_trace(go.Bar(x=hourly.index, y=hourly[t], name=t))
        fig.update_layout(barmode="stack", title="시간대별 작업 유형 분포",
                          xaxis_title="시간", yaxis_title="건수", height=300)
        return fig

    @output
    @render_widget
    def work_type_pie():
        d = filtered_data()
        if d.empty:
            return None
        cnt = d["작업유형"].value_counts()
        return px.pie(values=cnt.values, names=cnt.index,
                      title="작업유형별 분포", height=300)

    # ───────────────────────────────────────────────────────
    # 6) TAB 2: 분석 보고서 출력
    # ───────────────────────────────────────────────────────
    @output
    @render.text
    def summary_power_usage():
        d = summary_data()
        return f"{d['전력사용량'].sum():,.1f} kWh" if not d.empty else "0.0 kWh"

    @output
    @render.text
    def summary_power_cost():
        d = summary_data()
        return f"₩{d['전기요금'].sum():,.0f}" if not d.empty else "₩0"

    @output
    @render.text
    def summary_carbon_emission():
        d = summary_data()
        return f"{d['탄소배출량'].sum():,.1f} tCO₂" if not d.empty else "0.0 tCO₂"

    @output
    @render.text
    def summary_power_factor():
        d = summary_data()
        avg = d['역률'].mean() if '역률' in d and not d.empty else 0
        return f"{avg:.2f}"

    @output
    @render_widget
    def cost_trend_chart():
        d = summary_data()
        if d.empty:
            fig = go.Figure()
            fig.add_annotation(text="데이터 없음", x=0.5, y=0.5, showarrow=False)
            return fig
        hourly = (
            d.groupby(d["측정일시"].dt.floor("H"))["전기요금"]
             .agg(["sum","mean"])
             .reset_index()
        )
        hourly.columns = ["시간","누적요금","평균요금"]
        fig = go.Figure()
        fig.add_trace(go.Bar(    x=hourly["시간"], y=hourly["누적요금"], name="누적요금", opacity=0.7))
        fig.add_trace(go.Scatter(x=hourly["시간"], y=hourly["평균요금"],
                                 mode="lines+markers", name="평균요금", line=dict(color="red")))
        fig.update_layout(title="전력요금 시계열 분석", xaxis_title="시간", yaxis_title="원",
                          hovermode="x unified")
        return fig

    @output
    @render.text
    def peak_cost_info():
        d = summary_data()
        if d.empty:
            return "데이터 없음"
        idx = d["전기요금"].idxmax()
        cost = d.loc[idx, "전기요금"]
        dt   = d.loc[idx, "측정일시"]
        return f"최고요금: ₩{cost:,.0f}\n발생일시: {dt:%Y-%m-%d %H:%M}\n요일: {dt:%A}"

    @output
    @render.text
    def avg_carbon_info():
        d = summary_data()
        if d.empty:
            return "데이터 없음"
        avg, tot = d["탄소배출량"].mean(), d["탄소배출량"].sum()
        return f"평균: {avg:.1f} tCO₂\n총 배출량: {tot:.1f} tCO₂"

    @output
    @render.text
    def main_work_type_info():
        d = summary_data()
        if d.empty or "작업유형" not in d:
            return "데이터 없음"
        vc = d["작업유 유형"].value_counts()
        top, cnt, tot = vc.idxmax(), vc.max(), vc.sum()
        return f"최다 작업유형: {top}\n비중: {cnt/tot*100:.1f}% ({cnt}건)"

    @output
    @render.text
    def monthly_change_info():
        d = summary_data()
        if d.empty:
            return "데이터 없음"
        cur_sum = d["전기요금"].sum()
        prev_cutoff = d["측정일시"].min() - timedelta(days=30)
        prev = df[(df["측정일시"] >= prev_cutoff)
                 & (df["측정일시"] < d["측정일시"].min())]
        prev_sum = prev["전기요금"].sum() if not prev.empty else cur_sum
        rate = (cur_sum - prev_sum) / prev_sum * 100 if prev_sum else 0
        return f"{rate:+.1f}%"



# 앱 실행
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(debug=True, port=8000)