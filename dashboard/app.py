from shiny import App, render, ui, reactive
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
from shinywidgets import render_widget, output_widget

# 데이터 로드
def load_data():
    try:
        df = pd.read_csv('./data/train.csv')
        # 날짜 컬럼이 있다면 datetime으로 변환
        if '측정일시' in df.columns:
            df['측정일시'] = pd.to_datetime(df['측정일시'])
        return df
    except FileNotFoundError:
        # 테스트용 더미 데이터 생성
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='H')
        df = pd.DataFrame({
            '측정일시': dates,
            '전기요금(원)': np.random.normal(5000, 1000, len(dates)),
            '전력사용량(kWh)': np.random.normal(100, 20, len(dates)),
            '탄소배출량(CO2)': np.random.normal(50, 10, len(dates)),
            '작업유형': np.random.choice(['Light_Load', 'Heavy_Load'], len(dates))
        })
        return df

df = load_data()

# UI 정의
app_ui = ui.page_navbar(
    ui.nav_panel("실시간 모니터링",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("실시간 그래프"),
                ui.input_select(
                    "metric_select",
                    "지표 선택:",
                    choices={
                        "전기요금(원)": "전기요금(원)",
                        "전력사용량(kWh)": "전력사용량(kWh)",
                        "탄소배출량(CO2)": "탄소배출량(CO2)"
                    }
                ),
                ui.input_select(
                    "time_range",
                    "시간 범위:",
                    choices={
                        "1": "최근 1시간",
                        "3": "최근 3시간",
                        "24": "최근 24시간"
                    }
                ),
                ui.input_radio_buttons(
                    "chart_type",
                    "차트 유형:",
                    choices={
                        "line": "선 그래프",
                        "bar": "막대 그래프"
                    }
                ),
                ui.br(),
                ui.h5("시각화 구성"),
                ui.p("1. 현재 시점 누적요금 카드"),
                ui.p("2. 현재 시점 요금 선그래프"),
                ui.p("3. 전어 시간 요금 예측 선그래프")
            ),
            
                # 상단 카드들
                ui.row(
                    ui.column(4,
                        ui.div(
                            ui.h4("누적 요금"),
                            ui.output_text("cumulative_cost"),
                            class_="card p-3 bg-primary text-white"
                        )
                    ),
                    ui.column(4,
                        ui.div(
                            ui.h4("현재 요금"),
                            ui.output_text("current_cost"),
                            class_="card p-3 bg-success text-white"
                        )
                    ),
                    ui.column(4,
                        ui.div(
                            ui.h4("예측 요금"),
                            ui.output_text("predicted_cost"),
                            class_="card p-3 bg-warning text-white"
                        )
                    )
                ),
                ui.br(),
                # 차트들
                ui.row(
                    ui.column(12,
                        output_widget("main_chart")
                    )
                ),
                ui.br(),
                ui.row(
                    ui.column(12,
                        output_widget("prediction_chart")
                    )
                
            )
        )
    ),
    
    ui.nav_panel("분석 보고서",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("날짜 필터링"),
                ui.input_date_range(
                    "date_range",
                    "날짜 선택:",
                    start="2023-01-01",
                    end="2023-12-31"
                ),
                ui.input_select(
                    "work_type_filter",
                    "작업유형:",
                    choices=["전체", "Light_Load", "Heavy_Load"]
                ),
                ui.br(),
                ui.h5("시각화 구성"),
                ui.p("1. 주간 날짜 필터"),
                ui.p("2. 요일별 평균 요금 막대 그래프"),
                ui.p("3. 가장 높은 요금 날짜 카드"),
                ui.p("4. 상세 정보 카드")
            ),
            
                # 상단 정보 카드들
                ui.row(
                    ui.column(6,
                        ui.div(
                            ui.h4("요일별 평균 전기요금"),
                            output_widget("daily_avg_chart"),
                            class_="card p-3"
                        )
                    ),
                    ui.column(6,
                        ui.div(
                            ui.h4("가장 높은 요금"),
                            ui.output_text("highest_cost_info"),
                            class_="card p-3 bg-info text-white"
                        )
                    )
                ),
                ui.br(),
                # 상세 정보 카드들
                ui.row(
                    ui.column(4,
                        ui.div(
                            ui.h5("요일/시간대"),
                            ui.output_text("time_usage_info"),
                            class_="card p-3"
                        )
                    ),
                    ui.column(4,
                        ui.div(
                            ui.h5("사용 유형"),
                            ui.output_text("usage_type_info"),
                            class_="card p-3"
                        )
                    ),
                    ui.column(4,
                        ui.div(
                            ui.h5("전력계량 값들"),
                            ui.output_text("power_metrics_info"),
                            class_="card p-3"
                        )
                    )
                ),
                ui.br(),
                # 추가 분석 차트
                ui.row(
                    ui.column(12,
                        output_widget("detailed_analysis_chart")
                    )
                
            )
        )
    ),
    
    title="전기요금 실시간 모니터링",
    id="main_navbar"
)

# 서버 로직 정의
def server(input, output, session):
    
    # 필터링된 데이터 반응형 함수
    @reactive.Calc
    def filtered_data():
        filtered_df = df.copy()
        
        # 시간 범위 필터링 (첫 번째 탭용)
        if hasattr(input, 'time_range') and input.time_range():
            hours = int(input.time_range())
            end_time = df['측정일시'].max()
            start_time = end_time - timedelta(hours=hours)
            filtered_df = filtered_df[filtered_df['측정일시'] >= start_time]
        
        return filtered_df
    
    # 날짜 범위 필터링된 데이터 (두 번째 탭용)
    @reactive.Calc
    def date_filtered_data():
        filtered_df = df.copy()
        
        if hasattr(input, 'date_range') and input.date_range():
            start_date = pd.to_datetime(input.date_range()[0])
            end_date = pd.to_datetime(input.date_range()[1])
            filtered_df = filtered_df[
                (filtered_df['측정일시'].dt.date >= start_date.date()) &
                (filtered_df['측정일시'].dt.date <= end_date.date())
            ]
        
        if hasattr(input, 'work_type_filter') and input.work_type_filter() != "전체":
            filtered_df = filtered_df[filtered_df['작업유형'] == input.work_type_filter()]
        
        return filtered_df
    
    # 첫 번째 탭 - 실시간 모니터링
    @output
    @render.text
    def cumulative_cost():
        data = filtered_data()
        total = data['전기요금(원)'].sum()
        return f"₩{total:,.0f}"
    
    @output
    @render.text
    def current_cost():
        data = filtered_data()
        current = data['전기요금(원)'].iloc[-1] if len(data) > 0 else 0
        return f"₩{current:,.0f}"
    
    @output
    @render.text
    def predicted_cost():
        # 간단한 예측 로직
        data = filtered_data()
        if len(data) > 1:
            recent_avg = data['전기요금(원)'].tail(5).mean()
            return f"₩{recent_avg:,.0f}"
        return "₩0"
    
    @output
    @render_widget
    def main_chart():
        data = filtered_data()
        if len(data) == 0:
            return None
        
        metric = input.metric_select() if hasattr(input, 'metric_select') else '전기요금(원)'
        
        if input.chart_type() == "line":
            fig = px.line(data, x='측정일시', y=metric, 
                         title=f'{metric} 시계열 차트')
        else:
            fig = px.bar(data, x='측정일시', y=metric,
                        title=f'{metric} 막대 차트')
        
        fig.update_layout(
            title_font_size=16,
            xaxis_title="시간",
            yaxis_title=metric
        )
        return fig
    
    @output
    @render_widget
    def prediction_chart():
        data = filtered_data()
        if len(data) < 10:
            return None
        
        # 간단한 선형 회귀 예측
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['전기요금(원)'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 미래 시점 예측
        future_X = np.arange(len(data), len(data) + 24).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # 시간 인덱스 생성
        future_times = pd.date_range(
            start=data['측정일시'].max() + timedelta(hours=1),
            periods=24,
            freq='H'
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['측정일시'],
            y=data['전기요금(원)'],
            mode='lines',
            name='실제 데이터',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=future_times,
            y=predictions,
            mode='lines',
            name='예측 데이터',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="전기요금 예측 차트",
            xaxis_title="시간",
            yaxis_title="전기요금(원)"
        )
        return fig
    
    # 두 번째 탭 - 분석 보고서
    @output
    @render_widget
    def daily_avg_chart():
        data = date_filtered_data()
        if len(data) == 0:
            return None
        
        # 요일별 평균 계산
        data['요일'] = data['측정일시'].dt.day_name()
        daily_avg = data.groupby('요일')['전기요금(원)'].mean().reset_index()
        
        fig = px.bar(daily_avg, x='요일', y='전기요금(원)',
                    title='요일별 평균 전기요금')
        fig.update_layout(
            xaxis_title="요일",
            yaxis_title="평균 전기요금(원)"
        )
        return fig
    
    @output
    @render.text
    def highest_cost_info():
        data = date_filtered_data()
        if len(data) == 0:
            return "데이터 없음"
        
        max_cost = data['전기요금(원)'].max()
        max_date = data.loc[data['전기요금(원)'].idxmax(), '측정일시']
        return f"최고 요금: ₩{max_cost:,.0f}\n날짜: {max_date.strftime('%Y-%m-%d %H:%M')}"
    
    @output
    @render.text
    def time_usage_info():
        data = date_filtered_data()
        if len(data) == 0:
            return "데이터 없음"
        
        peak_hour = data.groupby(data['측정일시'].dt.hour)['전기요금(원)'].mean().idxmax()
        return f"피크 시간대: {peak_hour}시\n평균 사용량이 가장 높은 시간"
    
    @output
    @render.text
    def usage_type_info():
        data = date_filtered_data()
        if len(data) == 0:
            return "데이터 없음"
        
        type_avg = data.groupby('작업유형')['전기요금(원)'].mean()
        dominant_type = type_avg.idxmax()
        return f"주요 작업유형: {dominant_type}\n평균 요금: ₩{type_avg[dominant_type]:,.0f}"
    
    @output
    @render.text
    def power_metrics_info():
        data = date_filtered_data()
        if len(data) == 0:
            return "데이터 없음"
        
        avg_power = data['전력사용량(kWh)'].mean()
        avg_co2 = data['탄소배출량(CO2)'].mean()
        return f"평균 전력사용량: {avg_power:.1f}kWh\n평균 탄소배출량: {avg_co2:.1f}CO2"
    
    @output
    @render_widget
    def detailed_analysis_chart():
        data = date_filtered_data()
        if len(data) == 0:
            return None
        
        # 시간별 요금 히트맵 스타일의 차트
        data['시간'] = data['측정일시'].dt.hour
        data['날짜'] = data['측정일시'].dt.date
        
        hourly_avg = data.groupby('시간')['전기요금(원)'].mean().reset_index()
        
        fig = px.line(hourly_avg, x='시간', y='전기요금(원)',
                     title='시간대별 평균 전기요금 패턴')
        fig.update_layout(
            xaxis_title="시간",
            yaxis_title="평균 전기요금(원)"
        )
        return fig

# 앱 실행
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(debug=True)