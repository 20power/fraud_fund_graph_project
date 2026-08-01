from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_default_page_renders():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) >= 5
    assert len(app.dataframe) >= 2


@pytest.mark.parametrize("page", ["风险总览", "风险账户", "资金图谱", "可疑路径", "典型案例", "模型与口径"])
def test_every_dashboard_page_renders(page):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception


def test_dynamic_graph_uses_explicit_refresh_button():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.sidebar.radio[0].set_value("资金图谱").run()
    assert any(button.label == "生成 / 刷新资金图谱" for button in app.button)
