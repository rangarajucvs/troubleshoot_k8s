import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai.errors import APIError

from k8s_research import (
    MAX_ANSWER_WORDS,
    MAX_QUERY_WORDS,
    SOURCES,
    SearchResult,
    _DuckDuckGoResultsParser,
    _PageTextParser,
    _resolve_latest_flash_lite_model,
    research_issue,
    search_sources,
    synthesize_answer,
)


class SearchParserTests(unittest.TestCase):
    def test_extracts_result_title_url_and_snippet(self):
        parser = _DuckDuckGoResultsParser()
        parser.feed(
            '<div class="result"><div class="result__body">'
            '<a class="result__a" href="https://example.com">Pod scheduling</a>'
            "<div><a class='result__snippet'>No nodes available</a></div>"
            "</div></div>"
        )

        self.assertEqual(
            parser.results,
            [
                {
                    "url": "https://example.com",
                    "title": "Pod scheduling",
                    "snippet": "No nodes available",
                }
            ],
        )

    def test_page_parser_omits_scripts_and_normalizes_page_text(self):
        parser = _PageTextParser()
        parser.feed(
            "<html><body><nav>Navigation</nav><main>"
            "<p>Pod is <strong>Pending</strong>.</p><script>ignore()</script>"
            "</main></body></html>"
        )

        self.assertEqual(" ".join("".join(parser.parts).split()), "Navigation Pod is Pending.")


class ResearchTests(unittest.TestCase):
    def test_rejects_issue_over_word_limit_before_search(self):
        with patch("k8s_research.requests.get") as get:
            with self.assertRaisesRegex(ValueError, str(MAX_QUERY_WORDS)):
                search_sources("error " * (MAX_QUERY_WORDS + 1))
        get.assert_not_called()

    @patch("k8s_research._search_source")
    def test_deduplicates_results_and_collects_source_errors(self, search_source):
        result = SearchResult(
            SOURCES[0].name, "Troubleshooting", "https://example.com", ""
        )
        search_source.side_effect = [([result], None), ([result], "source unavailable")] + [
            ([], None)
        ] * 11

        with patch(
            "k8s_research._fetch_page_content", return_value=(result, None)
        ) as fetch_page:
            results, errors = search_sources("pod pending")

        self.assertEqual(results, [result])
        self.assertEqual(errors, ["source unavailable"])
        fetch_page.assert_called_once_with(result)

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    @patch("k8s_research._resolve_latest_flash_lite_model", return_value="gemini-3.1-flash-lite")
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_synthesis_uses_latest_flash_lite_and_caps_response(self, chat_model, resolve_model):
        chat_model.return_value.invoke.return_value = Mock(
            content="word " * (MAX_ANSWER_WORDS + 20)
        )

        answer = synthesize_answer(
            "pod pending",
            [SearchResult(SOURCES[0].name, "Scheduling", "https://example.com", "excerpt")],
        )

        self.assertEqual(len(answer.split()), MAX_ANSWER_WORDS)
        kwargs = chat_model.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.1-flash-lite")
        self.assertEqual(kwargs["google_api_key"], "test-key")
        self.assertEqual(kwargs["max_output_tokens"], 1800)
        resolve_model.assert_called_once_with("test-key")

    @patch("k8s_research.genai.Client")
    def test_resolver_selects_newest_stable_supported_flash_lite(self, client_class):
        client_class.return_value.close.return_value = None
        client_class.return_value.models.list.return_value = [
            SimpleNamespace(
                name="models/gemini-2.5-flash-lite",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-3.1-flash-lite",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-3.8-flash-lite-preview",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-4.0-flash-lite",
                supported_actions=["embedContent"],
            ),
        ]

        selected_model = _resolve_latest_flash_lite_model("test-key")

        self.assertEqual(selected_model, "gemini-3.1-flash-lite")
        client_class.assert_called_once_with(api_key="test-key")
        client_class.return_value.close.assert_called_once_with()

    @patch.dict(
        "os.environ",
        {"GOOGLE_API_KEY": "test-key", "GEMINI_MODEL": "gemini-custom"},
        clear=True,
    )
    @patch("k8s_research._resolve_latest_flash_lite_model")
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_configured_model_overrides_dynamic_discovery(
        self, chat_model, resolve_model
    ):
        chat_model.return_value.invoke.return_value = Mock(content="Summary")

        synthesize_answer("pod pending", [])

        self.assertEqual(chat_model.call_args.kwargs["model"], "gemini-custom")
        resolve_model.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)
    def test_synthesis_reports_missing_api_key(self):
        with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY or GEMINI_API_KEY"):
            synthesize_answer("pod pending", [])

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True)
    @patch(
        "k8s_research._resolve_latest_flash_lite_model",
        return_value="gemini-3.1-flash-lite",
    )
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_synthesis_surfaces_gemini_api_errors(self, chat_model, _resolve_model):
        chat_model.return_value.invoke.side_effect = APIError(
            503, {"error": {"message": "service timed out"}}
        )

        with self.assertRaisesRegex(RuntimeError, "service timed out"):
            synthesize_answer("pod pending", [])

    @patch("k8s_research.search_sources", return_value=([], ["search unavailable"]))
    @patch("k8s_research.synthesize_answer")
    def test_graph_skips_synthesis_without_search_results(self, synthesize, _search):
        answer, results, errors = research_issue("pod pending")

        self.assertEqual(answer, "")
        self.assertEqual(results, [])
        self.assertEqual(errors, ["search unavailable"])
        synthesize.assert_not_called()

    @patch("k8s_research.search_sources")
    @patch("k8s_research.synthesize_answer", return_value="Summary")
    def test_graph_searches_then_summarizes(self, synthesize, search):
        result = SearchResult(
            SOURCES[0].name, "Troubleshooting", "https://example.com", "excerpt"
        )
        search.return_value = ([result], [])

        answer, results, errors = research_issue("pod pending")

        self.assertEqual(answer, "Summary")
        self.assertEqual(results, [result])
        self.assertEqual(errors, [])
        search.assert_called_once_with("pod pending")
        synthesize.assert_called_once_with("pod pending", [result])


if __name__ == "__main__":
    unittest.main()
