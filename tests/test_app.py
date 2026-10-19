import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_clear_button_resets_issue_input(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()
        app.text_area[0].set_value("Pod is stuck in Pending")
        app.run()

        app.button[2].click().run()

        self.assertEqual(app.text_area[0].value, "")
        self.assertEqual(app.text_input[0].value, "")
        self.assertFalse(app.exception)

    def test_api_key_is_password_field_and_model_dropdown_is_visible(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()

        self.assertEqual(app.text_input[0].label, "Gemini API key")
        self.assertEqual(app.text_input[0].proto.type, app.text_input[0].proto.PASSWORD)
        self.assertEqual(app.selectbox[0].label, "Gemini model")
        self.assertFalse(app.exception)

    def test_gemini_key_is_cleared_after_form_submission(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()
        app.text_input[0].set_value("temporary-test-key")
        app.run()

        app.button[1].click().run()

        self.assertEqual(app.text_input[0].value, "")
        self.assertEqual(app.session_state["gemini_api_key"], "")
        self.assertFalse(app.exception)

    def test_issue_text_area_accepts_double_quotes(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()
        issue = 'Error: "failed to create pod"'

        app.text_area[0].set_value(issue)
        app.run()

        self.assertEqual(app.text_area[0].value, issue)
        self.assertFalse(app.exception)

    def test_search_errors_are_not_shown_in_output(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()
        app.session_state["investigation"] = {
            "answer": "Summary",
            "results": [],
            "search_errors": ["search timed out"],
        }

        app.run()

        self.assertEqual(app.get("warning"), [])
        self.assertEqual(app.expander, [])
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
