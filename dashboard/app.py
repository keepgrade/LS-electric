from shiny import App, render, ui, reactive
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
from shinywidgets import render_widget, output_widget

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
    
    ui.nav_panel("요금분석 보고서", 
        ui.div(
            ui.h3("📊 요금분석 보고서"),
            ui.p("여기에 분석 보고서 내용이 들어갑니다.")
        )
    ),
    
    ui.nav_panel("부록",
        ui.div(
            ui.h3("📚 부록"),
            ui.p("여기에 부록 내용이 들어갑니다.")
        )
    ),
    
    title="전기요금 실시간 모니터링 대시보드",
    id="main_navbar"
)

# 서버 로직
def server(input, output, session):
    
    # 필터링된 데이터
    @reactive.Calc
    def filtered_data():
        filtered_df = df.copy()
        
        if hasattr(input, 'date_range_monitoring') and input.date_range_monitoring():
            start_date = pd.to_datetime(input.date_range_monitoring()[0])
            end_date = pd.to_datetime(input.date_range_monitoring()[1])
            filtered_df = filtered_df[
                (filtered_df['측정일시'].dt.date >= start_date.date()) &
                (filtered_df['측정일시'].dt.date <= end_date.date())
            ]
        
        return filtered_df
    
    # [A] 요약 카드들
    @output
    @render.ui
    def card_power():
        data = filtered_data()
        current_power = data['전력사용량'].iloc[-1] if len(data) > 0 else 0
        return ui.div(
            ui.div(f"{current_power:,.0f}", class_="metric-value"),
            ui.div("kWh", class_="metric-label"),
            class_="metric-card"
        )
    
    @output
    @render.ui
    def card_cost():
        data = filtered_data()
        current_cost = data['전기요금'].iloc[-1] if len(data) > 0 else 0
        return ui.div(
            ui.div(f"{current_cost:,.0f}", class_="metric-value"),
            ui.div("원", class_="metric-label"),
            class_="metric-card"
        )
    
    @output
    @render.ui
    def card_co2():
        data = filtered_data()
        current_co2 = data['탄소배출량'].iloc[-1] if len(data) > 0 else 0
        return ui.div(
            ui.div(f"{current_co2:,.0f}", class_="metric-value"),
            ui.div("CO2", class_="metric-label"),
            class_="metric-card"
        )
    
    @output
    @render.ui
    def card_pf():
        return ui.div(
            ui.div("0.95", class_="metric-value"),
            ui.div("PF", class_="metric-label"),
            class_="metric-card"
        )
    
    @output
    @render.ui
    def card_work_type():
        data = filtered_data()
        dominant_type = data['작업유형'].mode().iloc[0] if len(data) > 0 else "N/A"
        return ui.div(
            ui.div(dominant_type, class_="metric-value", style="font-size: 18px;"),
            ui.div("작업유형", class_="metric-label"),
            class_="metric-card"
        )
    
    @output
    @render.ui
    def card_weather():
        return ui.div(
            ui.div("31°C", class_="metric-value"),
            ui.div("날씨", class_="metric-label"),
            class_="metric-card"
        )
    
    # [B] 실시간 그래프
    @output
    @render_widget
    def realtime_chart():
        data = filtered_data()
        if len(data) == 0:
            return None
        
        # 시간별 데이터 샘플링 (너무 많은 데이터 포인트 방지)
        data_sampled = data.iloc[::max(1, len(data)//100)]
        
        fig = go.Figure()
        
        if input.chart_type() == "line":
            if "전력사용량" in input.metrics_select():
                fig.add_trace(go.Scatter(
                    x=data_sampled['측정일시'],
                    y=data_sampled['전력사용량'],
                    mode='lines',
                    name='전력사용량 (kWh)',
                    line=dict(color='#3498db', width=2)
                ))
            
            if "전기요금" in input.metrics_select():
                fig.add_trace(go.Scatter(
                    x=data_sampled['측정일시'],
                    y=data_sampled['전기요금'],
                    mode='lines',
                    name='전기요금 (원)',
                    yaxis='y2',
                    line=dict(color='#e74c3c', width=2)
                ))
        
        fig.update_layout(
            title="실시간 전력사용량 및 전기요금 추이",
            xaxis_title="시간",
            yaxis=dict(title="전력사용량 (kWh)", side="left"),
            yaxis2=dict(title="전기요금 (원)", side="right", overlaying="y"),
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    # [C] 진행률 바들
    @output
    @render.ui
    def power_progress_bars():
        data = filtered_data()
        
        # 일/주/월 누적 계산
        daily_power = data.groupby(data['측정일시'].dt.date)['전력사용량'].sum().tail(7)
        weekly_power = daily_power.sum()
        monthly_power = data.groupby(data['측정일시'].dt.to_period('M'))['전력사용량'].sum().iloc[-1] if len(data) > 0 else 0
        
        return ui.div(
            ui.div(
                ui.div(
                    ui.div(f"일일 누적: {daily_power.iloc[-1]:,.0f} kWh", style="font-weight: bold;"),
                    ui.div(style=f"width: {min(100, daily_power.iloc[-1]/1000)}%; height: 8px; background: linear-gradient(90deg, #3498db, #2ecc71); border-radius: 4px; margin: 5px 0;"),
                    style="margin: 10px 0; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
                ),
                ui.div(
                    ui.div(f"주별 누적: {weekly_power:,.0f} kWh", style="font-weight: bold;"),
                    ui.div(style=f"width: {min(100, weekly_power/5000)}%; height: 8px; background: linear-gradient(90deg, #9b59b6, #8e44ad); border-radius: 4px; margin: 5px 0;"),
                    style="margin: 10px 0; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
                ),
                ui.div(
                    ui.div(f"월별 누적: {monthly_power:,.0f} kWh", style="font-weight: bold;"),
                    ui.div(style=f"width: {min(100, monthly_power/20000)}%; height: 8px; background: linear-gradient(90deg, #e67e22, #d35400); border-radius: 4px; margin: 5px 0;"),
                    style="margin: 10px 0; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
                )
            )
        )
    
    @output
    @render.ui
    def cost_progress_bars():
        data = filtered_data()
        
        # 일/주/월 누적 계산
        daily_cost = data.groupby(data['측정일시'].dt.date)['전기요금'].sum().tail(7)
        weekly_cost = daily_cost.sum()
        monthly_cost = data.groupby(data['측정일시'].dt.to_period('M'))['전기요금'].sum().iloc[-1] if len(data) > 0 else 0
        
        return ui.div(
            ui.div(
                ui.div(
                    ui.div(f"일일 누적: ₩{daily_cost.iloc[-1]:,.0f}", style="font-weight: bold;"),
                    ui.div(style=f"width: {min(100, daily_cost.iloc[-1]/500000)}%; height: 8px; background: linear-gradient(90deg, #27ae60, #229954); border-radius: 4px; margin: 5px 0;"),
                    style="margin: 10px 0; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
                    ),
                ui.div(
                    ui.div(f"주별 누적: ₩{weekly_cost:,.0f}", style="font-weight: bold;"),
                    ui.div(style=f"width: {min(100, weekly_cost/2000000)}%; height: 8px; background: linear-gradient(90deg, #f39c12, #e67e22); border-radius: 4px; margin: 5px 0;"),
                    style="margin: 10px 0; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
                ),
                ui.div(
                    ui.div(f"월별 누적: ₩{monthly_cost:,.0f}", style="font-weight: bold;"),
                    ui.div(style=f"width: {min(100, monthly_cost/8000000)}%; height: 8px; background: linear-gradient(90deg, #c0392b, #a93226); border-radius: 4px; margin: 5px 0;"),
                    style="margin: 10px 0; padding: 10px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
                )
            )
        )
    
    # [D] 작업 유형 분포 차트
    @output
    @render.plot
    def work_type_chart():
        data = filtered_data()
        if len(data) == 0:
            return None
        
        # 시간대별 작업 유형 분포
        hourly_work = data.groupby([data['측정일시'].dt.hour, '작업유형']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        
        for work_type in hourly_work.columns:
            fig.add_trace(go.Bar(
                x=hourly_work.index,
                y=hourly_work[work_type],
                name=work_type,
                text=hourly_work[work_type],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="시간대별 작업 유형 분포",
            xaxis_title="시간",
            yaxis_title="빈도",
            barmode='stack',
            height=300
        )
        
        return fig
    
    @output
    @render.plot
    def work_type_pie():
        data = filtered_data()
        if len(data) == 0:
            return None
        
        work_type_counts = data['작업유형'].value_counts()
        
        fig = px.pie(
            values=work_type_counts.values,
            names=work_type_counts.index,
            title="작업유형별 분포"
        )
        
        fig.update_layout(height=300)
        return fig

# 앱 실행
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(debug=True, port=8000)