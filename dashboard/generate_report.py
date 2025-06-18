# generate_report.py

from docxtpl import DocxTemplate, InlineImage
from pathlib import Path
from datetime import datetime
from docx.shared import Mm
import tempfile

# ───────────────────────────────────────────────────────
# 템플릿과 출력 디렉토리 설정
# ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

TEMPLATE_PATH = BASE_DIR / "templates" / "report_template.docx"  # ✅ 올바른 경로
OUTPUTS_DIR = Path(tempfile.gettempdir()) / "outputs"            # ✅ 안전한 임시 폴더
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_report(context: dict) -> Path:

    """
    context에 아래 키들이 포함되어야 합니다:
      customer_name, billing_month (예: "07"), customer_id,
      total_cost, usage_period, main_work_type,
      previous_month, current_usage, previous_usage,
      address, previous_total_cost, contract_type,
      graph1_path, graph2_path, graph3_path
    """
    # 1) 템플릿 로드
    doc = DocxTemplate(str(TEMPLATE_PATH))

    # 2) 생성일 추가
    context["generation_date"] = datetime.now().strftime("%Y-%m-%d")

    if "graph1_path" in context:
        context["graph1"] = InlineImage(
            doc,
            str(context["graph1_path"]),
            width=Mm(82.6),  # 너비 8.26 cm (82.6 mm)
            height=Mm(42.2),  # 높이 4.72 cm (47.2 mm)
        )
        del context["graph1_path"]


    if "graph2_path" in context:
        context["graph2"] = InlineImage(
            doc,
            str(context["graph2_path"]),
            width=Mm(154.9),
            height=Mm(77),
        )
        del context["graph2_path"]

    if "graph3_path" in context:
        context["graph3"] = InlineImage(
            doc,
            str(context["graph3_path"]),
            width=Mm(154.9),
            height=Mm(77),
        )
        del context["graph3_path"]

    if "graph4_path" in context:
        context["graph4"] = InlineImage(
            doc,
            str(context["graph4_path"]),
            width=Mm(154.9),
            height=Mm(77),
        )
        del context["graph4_path"]

    # 4) 템플릿 렌더링
    doc.render(context)

    # 5) 결과 파일명에 청구월 포함
    out_filename = f"electricity_report_{context['billing_month']}.docx"
    output_path = OUTPUTS_DIR / out_filename

    # 6) 저장 후 경로 반환
    doc.save(str(output_path))
    return output_path
