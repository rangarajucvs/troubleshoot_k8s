import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_clear_button_resets_issue_input(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()
        app.text_area[0].set_value("Pod is stuck in Pending")
        app.run()

        app.button[0].click().run()

        self.assertEqual(app.text_area[0].value, "")
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
