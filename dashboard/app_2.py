"""
전기요금 실시간 모니터링 대시보드
LS Electric 전력 사용량 및 요금 분석 시스템

주요 기능:
1. TAP 1: 실시간 전력 모니터링
2. TAP 2: 전기요금 분석 보고서
3. 예측 모델을 통한 요금 예측

작성자: Assistant
"""

from shiny import App, render, ui, reactive
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from shinywidgets import render_widget, output_widget
import asyncio
import time

# ====================================
# 데이터 로드 및 모델 초기화 함수
# ====================================

def load_data():
    """
    CSV 데이터 파일을 로드하는 함수
    train.csv와 test.csv를 읽어와서 전처리 수행
    """
    try:
        # 학습 데이터 로드
        train_df = pd.read_csv('./data/train.csv')
        
        # 테스트 데이터 로드 (실시간 데이터 시뮬레이션용)
        try:
            test_df = pd.read_csv('./data/test.csv')
        except FileNotFoundError:
            test_df = None
            
        # 날짜 컬럼 전처리
        if '측정일시' in train_df.columns:
            train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])
            
        if test_df is not None and '측정일시' in test_df.columns:
            test_df['측정일시'] = pd.to_datetime(test_df['측정일시'])
            
        return train_df, test_df
    
    except FileNotFoundError:
        # 테스트용 더미 데이터 생성
        print("데이터 파일을 찾을 수 없어 더미 데이터를 생성합니다.")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='15min')
        
        # 더미 데이터 생성 - 실제 전력 사용 패턴 반영
        np.random.seed(42)
        n_samples = len(dates)
        
        # 시간대별 패턴을 반영한 전력 사용량 생성
        hour_factor = np.sin(2 * np.pi * dates.hour / 24) * 0.3 + 1
        
        train_df = pd.DataFrame({
            '측정일시': dates,
            '전력사용량': np.random.normal(100, 20, n_samples) * hour_factor,
            '전력요금': np.random.normal(15000, 3000, n_samples) * hour_factor,
            '탄소배출량': np.random.normal(45, 8, n_samples) * hour_factor,
            '역률': np.random.normal(0.85, 0.05, n_samples),
            '작업유형': np.random.choice(['작업A', '작업B', '작업C'], n_samples, p=[0.4, 0.35, 0.25])
        })
        
        # 12월 데이터를 test로 분리
        test_df = train_df[train_df['측정일시'].dt.month == 12].copy()
        train_df = train_df[train_df['측정일시'].dt.month != 12].copy()
        
        return train_df, test_df

def load_model():
    """
    사전 훈련된 XGBoost 모델을 로드하는 함수
    """
    try:
        with open('./www/xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("XGBoost 모델이 성공적으로 로드되었습니다.")
        return model
    except FileNotFoundError:
        print("모델 파일을 찾을 수 없습니다. 더미 모델을 사용합니다.")
        # 더미 모델 클래스
        class DummyModel:
            def predict(self, X):
                return np.random.normal(15000, 2000, len(X))
        return DummyModel()

# ====================================
# 데이터 및 모델 초기화
# ====================================

train_df, test_df = load_data()
prediction_model = load_model()

# ====================================
# 유틸리티 함수들
# ====================================

def calculate_change_rate(current_value, previous_value):
    """
    증감률 계산 함수
    Args:
        current_value: 현재 값
        previous_value: 이전 값
    Returns:
        증감률 (%)
    """
    if previous_value == 0 or pd.isna(previous_value):
        return 0
    return ((current_value - previous_value) / abs(previous_value)) * 100

def format_currency(amount):
    """
    통화 포맷팅 함수
    Args:
        amount: 금액
    Returns:
        포맷된 문자열
    """
    return f"₩{amount:,.0f}"

def format_number_with_unit(value, unit):
    """
    숫자와 단위를 포함한 포맷팅 함수
    """
    return f"{value:,.1f} {unit}"

# ====================================
# 실시간 데이터 시뮬레이션 클래스
# ====================================

class RealTimeDataSimulator:
    """
    실시간 데이터 스트리밍을 시뮬레이션하는 클래스
    test_df의 데이터를 1초마다 순차적으로 반환
    """
    def __init__(self, test_data):
        self.test_data = test_data.copy() if test_data is not None else pd.DataFrame()
        self.current_index = 0
        self.start_time = time.time()
        
    def get_current_data(self):
        """
        현재 시점의 데이터를 반환
        """
        if len(self.test_data) == 0:
            return None
            
        if self.current_index >= len(self.test_data):
            # 데이터 끝에 도달하면 처음부터 다시 시작
            self.current_index = 0
            
        current_row = self.test_data.iloc[self.current_index]
        self.current_index += 1
        
        return current_row
    
    def get_cumulative_data(self):
        """
        현재까지의 누적 데이터를 반환
        """
        if len(self.test_data) == 0 or self.current_index == 0:
            return pd.DataFrame()
            
        return self.test_data.iloc[:self.current_index]

# 실시간 시뮬레이터 초기화
real_time_simulator = RealTimeDataSimulator(test_df)

# ====================================
# UI 정의
# ====================================

app_ui = ui.page_navbar(
    # TAB 1: 실시간 모니터링
    ui.nav_panel("실시간 모니터링",
        ui.layout_column_wrap(
            # 상단 요약 카드 섹션
            ui.card(
                ui.card_header("📊 실시간 전력 현황"),
                ui.layout_column_wrap(
                    # 전력사용량 카드
                    ui.value_box(
                        title="전력사용량",
                        value=ui.output_text("rt_power_usage"),
                        showcase="⚡",
                        theme="bg-primary"
                    ),
                    # 전력요금 카드
                    ui.value_box(
                        title="전력요금",
                        value=ui.output_text("rt_power_cost"),
                        showcase="💰",
                        theme="bg-success"
                    ),
                    # 탄소배출량 카드
                    ui.value_box(
                        title="탄소배출량",
                        value=ui.output_text("rt_carbon_emission"),
                        showcase="🌱",
                        theme="bg-warning"
                    ),
                    # 평균 역률 카드
                    ui.value_box(
                        title="평균 역률",
                        value=ui.output_text("rt_power_factor"),
                        showcase="⚙️",
                        theme="bg-info"
                    ),
                    width=1/4
                )
            ),
            
            # 실시간 그래프 섹션
            ui.card(
                ui.card_header("📈 실시간 추이 그래프"),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_select(
                            "rt_metric_select",
                            "표시할 지표:",
                            choices={
                                "전력사용량": "전력사용량 (kWh)",
                                "전력요금": "전력요금 (원)",
                                "탄소배출량": "탄소배출량 (tCO₂)",
                                "역률": "역률"
                            },
                            selected="전력사용량"
                        ),
                        ui.input_date_range(
                            "rt_date_range",
                            "날짜 범위:",
                            start="2024-12-01",
                            end="2024-12-31"
                        ),
                        ui.input_action_button(
                            "rt_play_pause",
                            "▶️ 재생/일시정지",
                            class_="btn-primary"
                        )
                    ),
                    output_widget("rt_trend_chart")
                )
            ),
            
            # 누적 사용량 슬라이더 섹션
            ui.card(
                ui.card_header("📊 12월 누적 사용량 진행률"),
                ui.div(
                    ui.h5("전력사용량 진행률", class_="text-center"),
                    ui.output_ui("power_usage_progress"),
                    ui.br(),
                    ui.h5("전력요금 진행률", class_="text-center"), 
                    ui.output_ui("power_cost_progress")
                )
            ),
            
            # 작업 유형 분포 섹션
            ui.card(
                ui.card_header("📋 작업 유형별 전력 사용 분포"),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_radio_buttons(
                            "work_type_period",
                            "집계 기간:",
                            choices={
                                "day": "일별",
                                "week": "주별", 
                                "month": "월별",
                                "hour": "시간대별"
                            },
                            selected="day"
                        )
                    ),
                    ui.layout_column_wrap(
                        output_widget("work_type_bar_chart"),
                        output_widget("work_type_donut_chart"),
                        width=1/2
                    )
                )
            ),
            width=1
        )
    ),
    
    # TAB 2: 분석 보고서
    ui.nav_panel("분석 보고서",
        ui.layout_column_wrap(
            # 기간별 요약 카드
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
            
            # 전력 요금 그래프
            ui.card(
                ui.card_header("📈 전력 요금 시계열 분석"),
                output_widget("cost_trend_chart")
            ),
            
            # 분석 보고서 정보 카드
            ui.card(
                ui.card_header("📊 상세 분석 정보"),
                ui.layout_column_wrap(
                    ui.card(
                        ui.card_header("최고 요금 정보"),
                        ui.output_text("peak_cost_info")
                    ),
                    ui.card(
                        ui.card_header("평균 탄소배출량"),
                        ui.output_text("avg_carbon_info")
                    ),
                    ui.card(
                        ui.card_header("주요 작업 유형"),
                        ui.output_text("main_work_type_info")
                    ),
                    ui.card(
                        ui.card_header("전월 대비 증감률"),
                        ui.output_text("monthly_change_info")
                    ),
                    width=1/2
                ),
                ui.br(),
                ui.div(
                    ui.input_action_button(
                        "download_pdf",
                        "📄 PDF 보고서 다운로드",
                        class_="btn-success btn-lg"
                    ),
                    class_="text-center"
                )
            ),
            width=1
        )
    ),
    
    title="⚡ LS Electric 전기요금 실시간 모니터링",
    id="main_navbar"
)

# ====================================
# 서버 로직 정의
# ====================================

def server(input, output, session):
    """
    Shiny 서버 함수
    모든 반응형 로직과 출력 렌더링을 담당
    """
    
    # ====================================
    # 반응형 데이터 함수들
    # ====================================
    
    @reactive.Calc
    def get_realtime_data():
        """
        실시간 데이터를 반환하는 반응형 함수
        """
        # 실시간 시뮬레이션을 위해 현재 데이터 반환
        current_data = real_time_simulator.get_current_data()
        return current_data
    
    @reactive.Calc
    def get_cumulative_realtime_data():
        """
        누적 실시간 데이터를 반환하는 반응형 함수
        """
        return real_time_simulator.get_cumulative_data()
    
    @reactive.Calc
    def get_filtered_data_by_period():
        """
        선택된 기간에 따라 필터링된 데이터를 반환
        """
        period = input.summary_period()
        now = datetime.now()
        
        # 기간별 시작 시간 계산
        if period == "15min":
            start_time = now - timedelta(minutes=15)
        elif period == "30min":
            start_time = now - timedelta(minutes=30)
        elif period == "1hour":
            start_time = now - timedelta(hours=1)
        elif period == "today":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start_time = now - timedelta(days=7)
        elif period == "month":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)
            
        # 데이터 필터링
        filtered_data = train_df[train_df['측정일시'] >= start_time].copy()
        return filtered_data
    
    # ====================================
    # TAB 1: 실시간 모니터링 출력들
    # ====================================
    
    @output
    @render.text
    def rt_power_usage():
        """실시간 전력사용량 출력"""
        current_data = get_realtime_data()
        if current_data is not None:
            value = current_data.get('전력사용량', 0)
            return format_number_with_unit(value, 'kWh')
        return "0.0 kWh"
    
    @output
    @render.text
    def rt_power_cost():
        """실시간 전력요금 출력"""
        current_data = get_realtime_data()
        if current_data is not None:
            value = current_data.get('전력요금', 0)
            return format_currency(value)
        return "₩0"
    
    @output
    @render.text
    def rt_carbon_emission():
        """실시간 탄소배출량 출력"""
        current_data = get_realtime_data()
        if current_data is not None:
            value = current_data.get('탄소배출량', 0)
            return format_number_with_unit(value, 'tCO₂')
        return "0.0 tCO₂"
    
    @output
    @render.text
    def rt_power_factor():
        """실시간 평균 역률 출력"""
        current_data = get_realtime_data()
        if current_data is not None:
            value = current_data.get('역률', 0)
            return f"{value:.2f}"
        return "0.00"
    
    @output
    @render_widget
    def rt_trend_chart():
        """실시간 추이 그래프"""
        cumulative_data = get_cumulative_realtime_data()
        
        if len(cumulative_data) == 0:
            # 빈 차트 반환
            fig = go.Figure()
            fig.add_annotation(
                text="데이터를 불러오는 중...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        metric = input.rt_metric_select()
        
        # 선택된 지표에 따른 차트 생성
        fig = px.line(
            cumulative_data, 
            x='측정일시', 
            y=metric,
            title=f'실시간 {metric} 추이',
            line_shape='spline'
        )
        
        # 차트 스타일링
        fig.update_layout(
            title_font_size=16,
            xaxis_title="시간",
            yaxis_title=metric,
            hovermode='x unified'
        )
        
        return fig
    
    @output
    @render.ui
    def power_usage_progress():
        """전력사용량 진행률 프로그레스 바"""
        cumulative_data = get_cumulative_realtime_data()
        
        if len(cumulative_data) == 0:
            current_usage = 0
            predicted_total = 100000  # 예측값 (더미)
        else:
            current_usage = cumulative_data['전력사용량'].sum()
            predicted_total = current_usage * 2  # 간단한 예측 로직
        
        progress_percent = min((current_usage / predicted_total) * 100, 100)
        
        return ui.div(
            ui.p(f"현재: {format_number_with_unit(current_usage, 'kWh')} / 예상: {format_number_with_unit(predicted_total, 'kWh')}"),
            ui.tags.div(
                ui.tags.div(
                    style=f"width: {progress_percent}%; background-color: #007bff; height: 25px; border-radius: 12px; transition: width 0.3s ease;"
                ),
                style="width: 100%; background-color: #e9ecef; border-radius: 12px; overflow: hidden;"
            ),
            ui.p(f"진행률: {progress_percent:.1f}%", class_="text-center mt-2")
        )
    
    @output
    @render.ui
    def power_cost_progress():
        """전력요금 진행률 프로그레스 바"""
        cumulative_data = get_cumulative_realtime_data()
        
        if len(cumulative_data) == 0:
            current_cost = 0
            predicted_total = 5000000  # 예측값 (더미)
        else:
            current_cost = cumulative_data['전력요금'].sum()
            predicted_total = current_cost * 2  # 간단한 예측 로직
        
        progress_percent = min((current_cost / predicted_total) * 100, 100)
        
        return ui.div(
            ui.p(f"현재: {format_currency(current_cost)} / 예상: {format_currency(predicted_total)}"),
            ui.tags.div(
                ui.tags.div(
                    style=f"width: {progress_percent}%; background-color: #28a745; height: 25px; border-radius: 12px; transition: width 0.3s ease;"
                ),
                style="width: 100%; background-color: #e9ecef; border-radius: 12px; overflow: hidden;"
            ),
            ui.p(f"진행률: {progress_percent:.1f}%", class_="text-center mt-2")
        )
    
    @output
    @render_widget
    def work_type_bar_chart():
        """작업 유형별 막대 그래프"""
        period = input.work_type_period()
        data = train_df.copy()
        
        # 기간별 그룹핑
        if period == "hour":
            data['period'] = data['측정일시'].dt.hour
            xlabel = "시간"
        elif period == "day":
            data['period'] = data['측정일시'].dt.date
            xlabel = "날짜"
        elif period == "week":
            data['period'] = data['측정일시'].dt.isocalendar().week
            xlabel = "주"
        elif period == "month":
            data['period'] = data['측정일시'].dt.month
            xlabel = "월"
        
        # 작업 유형별 집계
        grouped_data = data.groupby(['period', '작업유형'])['전력사용량'].sum().reset_index()
        
        fig = px.bar(
            grouped_data,
            x='period',
            y='전력사용량',
            color='작업유형',
            title=f'{period.title()}별 작업유형 전력사용량',
            barmode='stack'
        )
        
        fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title="전력사용량 (kWh)"
        )
        
        return fig
    
    @output
    @render_widget
    def work_type_donut_chart():
        """작업 유형별 도넛 차트"""
        data = train_df.copy()
        
        # 작업 유형별 총 사용량 계산
        work_type_total = data.groupby('작업유형')['전력사용량'].sum().reset_index()
        
        fig = px.pie(
            work_type_total,
            values='전력사용량',
            names='작업유형',
            title='전체 기간 작업유형별 비율',
            hole=0.4
        )
        
        return fig
    
    # ====================================
    # TAB 2: 분석 보고서 출력들
    # ====================================
    
    @output
    @render.text
    def summary_power_usage():
        """기간별 누적 전력사용량"""
        data = get_filtered_data_by_period()
        if len(data) > 0:
            total = data['전력사용량'].sum()
            return format_number_with_unit(total, 'kWh')
        return "0.0 kWh"
    
    @output
    @render.text
    def summary_power_cost():
        """기간별 누적 전력요금"""
        data = get_filtered_data_by_period()
        if len(data) > 0:
            total = data['전력요금'].sum()
            return format_currency(total)
        return "₩0"
    
    @output
    @render.text
    def summary_carbon_emission():
        """기간별 누적 탄소배출량"""
        data = get_filtered_data_by_period()
        if len(data) > 0:
            total = data['탄소배출량'].sum()
            return format_number_with_unit(total, 'tCO₂')
        return "0.0 tCO₂"
    
    @output
    @render.text
    def summary_power_factor():
        """기간별 평균 역률"""
        data = get_filtered_data_by_period()
        if len(data) > 0:
            avg = data['역률'].mean()
            return f"{avg:.2f}"
        return "0.00"
    
    @output
    @render_widget
    def cost_trend_chart():
        """전력 요금 시계열 분석 차트"""
        data = get_filtered_data_by_period()
        
        if len(data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="선택된 기간에 데이터가 없습니다.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # 시간별 집계
        hourly_data = data.groupby(data['측정일시'].dt.floor('H')).agg({
            '전력요금': ['sum', 'mean']
        }).reset_index()
        
        hourly_data.columns = ['시간', '누적요금', '평균요금']
        
        fig = go.Figure()
        
        # 누적 요금 막대 그래프
        fig.add_trace(go.Bar(
            x=hourly_data['시간'],
            y=hourly_data['누적요금'],
            name='시간별 누적요금',
            yaxis='y1',
            opacity=0.7
        ))
        
        # 평균 요금 선 그래프
        fig.add_trace(go.Scatter(
            x=hourly_data['시간'],
            y=hourly_data['평균요금'],
            mode='lines+markers',
            name='시간별 평균요금',
            yaxis='y2',
            line=dict(color='red', width=2)
        ))
        
        # 이중 Y축 설정
        fig.update_layout(
            title="전력요금 시계열 분석",
            xaxis_title="시간",
            yaxis=dict(title="누적요금 (원)", side="left"),
            yaxis2=dict(title="평균요금 (원)", side="right", overlaying="y"),
            hovermode='x unified'
        )
        
        return fig
    
    @output
    @render.text
    def peak_cost_info():
        """최고 요금 정보"""
        data = get_filtered_data_by_period()
        if len(data) > 0:
            max_cost = data['전력요금'].max()
            max_date = data.loc[data['전력요금'].idxmax(), '측정일시']
            max_day = max_date.strftime('%A')  # 요일
            return f"최고요금: {format_currency(max_cost)}\n발생일시: {max_date.strftime('%m/%d %H:%M')}\n요일: {max_day}"
        return "데이터 없음"
    
    @output
    @render.text
    def avg_carbon_info():
        """평균 탄소배출량 정보"""
        data = get_filtered_data_by_period()
        if len(data) > 0:
            avg_carbon = data['탄소배출량'].mean()
            total_carbon = data['탄소배출량'].sum()
            return f"평균: {avg_carbon:.1f} tCO₂\n총 배출량: {total_carbon:.1f} tCO₂"
        return "데이터 없음"
    
    @output
    @render.text
    def main_work_type_info():
        """최다 작업유형 정보"""
        data = get_filtered_data_by_period()
        if len(data) > 0 and '작업유형' in data.columns:
            work_type_counts = data['작업유형'].value_counts()
            top_type = work_type_counts.idxmax()
            top_count = work_type_counts.max()
            total = work_type_counts.sum()
            ratio = (top_count / total) * 100
            return f"최다 작업유형: {top_type}\n비중: {ratio:.1f}% ({top_count}건)"
        return "데이터 없음"
    
    @output
    @render_widget
    def weekday_usage_bar():
        """요일별 평균 전력 사용량 시각화"""
        data = get_filtered_data_by_period()
        if len(data) == 0:
            return go.Figure()

        data['요일'] = data['측정일시'].dt.day_name()
        weekday_avg = data.groupby('요일')['전력사용량'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=weekday_avg.index,
            y=weekday_avg.values,
            marker_color='lightskyblue',
            name='평균 전력사용량'
        ))
        fig.update_layout(title='요일별 평균 전력사용량', yaxis_title='kWh')
        return fig

    @output
    @render_widget
    def worktype_cost_pie():
        """작업유형별 전력요금 분포 (파이 차트)"""
        data = get_filtered_data_by_period()
        if len(data) == 0 or '작업유형' not in data.columns:
            return go.Figure()

        cost_by_type = data.groupby('작업유형')['전력요금'].sum()
        fig = go.Figure(data=[go.Pie(
            labels=cost_by_type.index,
            values=cost_by_type.values,
            hole=0.3
        )])
        fig.update_layout(title='작업유형별 전력요금 비중')
        return fig

# =============================
# 🔧 Shiny 앱 객체 생성 및 실행
# =============================
# ================================
# 🚀 4. 앱 실행
# ================================
app = App(app_ui, server)
