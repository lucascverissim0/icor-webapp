from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_login_page_renders_without_a_local_secrets_file(socket_enabled: None) -> None:
    app = AppTest.from_file(PROJECT_ROOT / "ui" / "app.py", default_timeout=10)

    app.run()

    assert not app.exception
    assert app.title[0].value == "Strategic Opportunities"
    assert [field.label for field in app.text_input] == ["Username", "Password"]
