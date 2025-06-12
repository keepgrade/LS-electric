##############
# 📦 라이브러리 및 데이터 임포트
##############

import seaborn as sns
from faicons import icon_svg
from shared import app_dir, df

from shiny import reactive
from shiny.express import input, render, ui

##############
# 🖼️ 페이지 전역 설정
##############

ui.page_opts(title="Penguins dashboard", fillable=True)

##############
# 🎛️ 사이드바 필터 구성
##############

with ui.sidebar(title="Filter controls"):
    ui.input_slider("mass", "Mass", 2000, 6000, 6000)
    ui.input_checkbox_group(
        "species",
        "Species",
        ["Adelie", "Gentoo", "Chinstrap"],
        selected=["Adelie", "Gentoo", "Chinstrap"],
    )

##############
# 📊 요약 지표 카드 (value_box)
##############

with ui.layout_column_wrap(fill=False):
    with ui.value_box(showcase=icon_svg("earlybirds")):
        "Number of penguins"

        @render.text
        def count():
            return filtered_df().shape[0]

    with ui.value_box(showcase=icon_svg("ruler-horizontal")):
        "Average bill length"

        @render.text
        def bill_length():
            return f"{filtered_df()['bill_length_mm'].mean():.1f} mm"

    with ui.value_box(showcase=icon_svg("ruler-vertical")):
        "Average bill depth"

        @render.text
        def bill_depth():
            return f"{filtered_df()['bill_depth_mm'].mean():.1f} mm"

##############
# 📈 시각화 및 📋 데이터 테이블 카드
##############

with ui.layout_columns():
    with ui.card(full_screen=True):
        ui.card_header("Bill length and depth")

        @render.plot
        def length_depth():
            return sns.scatterplot(
                data=filtered_df(),
                x="bill_length_mm",
                y="bill_depth_mm",
                hue="species",
            )

    with ui.card(full_screen=True):
        ui.card_header("Penguin data")

        @render.data_frame
        def summary_statistics():
            cols = [
                "species",
                "island",
                "bill_length_mm",
                "bill_depth_mm",
                "body_mass_g",
            ]
            return render.DataGrid(filtered_df()[cols], filters=True)

##############
# 🎨 사용자 정의 스타일 시트 연결
##############

ui.include_css(app_dir / "styles.css")

##############
# 🔁 리액티브 필터링 함수 정의
##############

@reactive.calc
def filtered_df():
    filt_df = df[df["species"].isin(input.species())]
    filt_df = filt_df.loc[filt_df["body_mass_g"] < input.mass()]
    return filt_df
