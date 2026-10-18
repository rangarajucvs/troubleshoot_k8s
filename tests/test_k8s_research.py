import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from google.genai.errors import APIError

from k8s_research import (
    MAX_ANSWER_WORDS,
    MAX_QUERY_WORDS,
    SOURCES,
    SearchResult,
    _PageTextParser,
    _fetch_page_content,
    _search_source,
    _resolve_latest_flash_lite_model,
    list_gemini_models,
    research_issue,
    search_sources,
    synthesize_answer,
)


class SearchParserTests(unittest.TestCase):
    def test_page_parser_omits_scripts_and_normalizes_page_text(self):
        parser = _PageTextParser()
        parser.feed(
            "<html><body><nav>Navigation</nav><main>"
            "<p>Pod is <strong>Pending</strong>.</p><script>ignore()</script>"
            "</main></body></html>"
        )

        self.assertEqual(" ".join("".join(parser.parts).split()), "Navigation Pod is Pending.")


class ResearchTests(unittest.TestCase):
    @patch("k8s_research.requests.get")
    def test_forbidden_page_falls_back_to_search_excerpt(self, get):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError(
            "403 Client Error: Forbidden",
            response=Mock(status_code=403),
        )
        get.return_value = response
        result = SearchResult(
            "Stack Overflow Kubernetes questions",
            "Pods stuck in ContainerCreating",
            "https://stackoverflow.com/questions/123",
            "CNI configuration is missing",
        )

        page_result, error = _fetch_page_content(result)

        self.assertIsNone(error)
        self.assertEqual(page_result.snippet, result.snippet)
        self.assertEqual(page_result.page_excerpt, "")
        self.assertIn("HTTP 403", page_result.page_note)

    def test_rejects_issue_over_word_limit_before_search(self):
        with patch("k8s_research.DDGS") as ddgs:
            with self.assertRaisesRegex(ValueError, str(MAX_QUERY_WORDS)):
                search_sources("error " * (MAX_QUERY_WORDS + 1))
        ddgs.assert_not_called()

    @patch("k8s_research.DDGS")
    def test_search_uses_ddgs_and_converts_results(self, ddgs_class):
        ddgs_class.return_value.__enter__.return_value.text.return_value = [
            {
                "title": "Pod scheduling",
                "href": "https://example.com/pods",
                "body": "No nodes available",
            },
            {"title": "Missing URL"},
        ]

        results, error = _search_source(SOURCES[0], "pod pending", result_limit=3)

        self.assertIsNone(error)
        self.assertEqual(
            results,
            [
                SearchResult(
                    SOURCES[0].name,
                    "Pod scheduling",
                    "https://example.com/pods",
                    "No nodes available",
                )
            ],
        )
        ddgs_class.assert_called_once_with(timeout=30)
        ddgs_class.return_value.__enter__.return_value.text.assert_called_once_with(
            "site:kubernetes.io/docs Kubernetes pod pending",
            region="us-en",
            safesearch="moderate",
            backend="auto",
            max_results=3,
        )

    @patch("k8s_research.DDGS")
    def test_search_query_preserves_double_quoted_error_text(self, ddgs_class):
        ddgs_class.return_value.__enter__.return_value.text.return_value = []

        _search_source(SOURCES[0], 'Error: "failed to create pod"', result_limit=2)

        query = ddgs_class.return_value.__enter__.return_value.text.call_args.args[0]
        self.assertIn('"failed to create pod"', query)

    @patch("k8s_research.DDGS")
    def test_search_reports_timeout_with_recovery_hint(self, ddgs_class):
        from ddgs.exceptions import TimeoutException

        ddgs_class.return_value.__enter__.return_value.text.side_effect = TimeoutException(
            "connection timeout"
        )

        results, error = _search_source(SOURCES[0], "pod pending", result_limit=2)

        self.assertEqual(results, [])
        self.assertIn("timed out after 30s", error)
        self.assertIn("proxy settings", error)

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

    @patch("k8s_research._resolve_latest_flash_lite_model", return_value="gemini-3.1-flash-lite")
    @patch("k8s_research.list_gemini_models", return_value=["gemini-3.1-flash-lite"])
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_synthesis_uses_latest_flash_lite_and_caps_response(
        self, chat_model, _list_models, resolve_model
    ):
        chat_model.return_value.invoke.return_value = Mock(
            content="word " * (MAX_ANSWER_WORDS + 20)
        )

        answer = synthesize_answer(
            "pod pending",
            [SearchResult(SOURCES[0].name, "Scheduling", "https://example.com", "excerpt")],
            api_key="test-key",
        )

        self.assertEqual(len(answer.split()), MAX_ANSWER_WORDS)
        kwargs = chat_model.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.1-flash-lite")
        self.assertEqual(kwargs["google_api_key"], "test-key")
        self.assertEqual(kwargs["max_output_tokens"], 1800)
        resolve_model.assert_called_once_with(["gemini-3.1-flash-lite"])

    def test_resolver_selects_newest_stable_flash_lite(self):
        selected_model = _resolve_latest_flash_lite_model(
            [
                "gemini-2.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-3.8-flash-lite-preview",
            ]
        )

        self.assertEqual(selected_model, "gemini-3.1-flash-lite")

    @patch("k8s_research.genai.Client")
    def test_lists_only_models_supporting_generation(self, client_class):
        client_class.return_value.close.return_value = None
        client_class.return_value.models.list.return_value = [
            SimpleNamespace(
                name="models/gemini-3.1-flash-lite",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(name="models/embedding-001", supported_actions=["embedContent"]),
        ]

        models = list_gemini_models("user-key")

        self.assertEqual(models, ["gemini-3.1-flash-lite"])
        client_class.assert_called_once_with(api_key="user-key")
        client_class.return_value.close.assert_called_once_with()

    @patch("k8s_research.genai.Client")
    def test_model_listing_reports_invalid_api_key(self, client_class):
        client_class.return_value.close.return_value = None
        client_class.return_value.models.list.side_effect = APIError(
            400, {"error": {"message": "API key not valid"}}
        )

        with self.assertRaisesRegex(RuntimeError, "Could not load Gemini models"):
            list_gemini_models("invalid-key")
        client_class.return_value.close.assert_called_once_with()

    @patch.dict("os.environ", {}, clear=True)
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_explicit_model_skips_model_discovery(self, chat_model):
        chat_model.return_value.invoke.return_value = Mock(content="Summary")

        synthesize_answer("pod pending", [], api_key="test-key", model="gemini-custom")

        self.assertEqual(chat_model.call_args.kwargs["model"], "gemini-custom")

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_synthesis_extracts_text_from_gemini_3_content_blocks(self, chat_model):
        chat_model.return_value.invoke.return_value = Mock(
            content=[
                {"type": "thinking", "thinking": "internal reasoning"},
                {"type": "text", "text": "  Summary of problem.  "},
                {"type": "text", "text": "Next steps."},
            ]
        )

        answer = synthesize_answer(
            "pod pending", [], api_key="test-key", model="gemini-3.1-flash-lite"
        )

        self.assertEqual(answer, "Summary of problem.\nNext steps.")

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_empty_gemini_response_has_actionable_diagnostic(self, chat_model):
        chat_model.return_value.invoke.return_value = Mock(
            content="",
            response_metadata={"finish_reason": "SAFETY"},
        )

        with self.assertRaisesRegex(RuntimeError, "SAFETY") as error:
            synthesize_answer(
                "pod pending", [], api_key="test-key", model="gemini-3.1-flash-lite"
            )
        self.assertIn("Try rephrasing", str(error.exception))

    @patch.dict("os.environ", {}, clear=True)
    def test_synthesis_reports_missing_api_key(self):
        with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY or GEMINI_API_KEY"):
            synthesize_answer("pod pending", [])

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "k8s_research._resolve_latest_flash_lite_model",
        return_value="gemini-3.1-flash-lite",
    )
    @patch("k8s_research.list_gemini_models", return_value=["gemini-3.1-flash-lite"])
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_synthesis_surfaces_gemini_api_errors(
        self, chat_model, _list_models, _resolve_model
    ):
        chat_model.return_value.invoke.side_effect = APIError(
            503, {"error": {"message": "service timed out"}}
        )

        with self.assertRaisesRegex(RuntimeError, "service timed out"):
            synthesize_answer("pod pending", [], api_key="test-key")

    @patch("k8s_research.search_sources", return_value=([], ["search unavailable"]))
    @patch("k8s_research.synthesize_answer")
    def test_graph_skips_synthesis_without_search_results(self, synthesize, _search):
        answer, results, errors = research_issue(
            "pod pending",
            api_key="user-key",
            model="gemini-3.1-flash",
            available_models=["gemini-3.1-flash", "gemini-3.1-flash-lite"],
        )

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

        answer, results, errors = research_issue(
            "pod pending",
            api_key="user-key",
            model="gemini-3.1-flash",
            available_models=["gemini-3.1-flash", "gemini-3.1-flash-lite"],
        )

        self.assertEqual(answer, "Summary")
        self.assertEqual(results, [result])
        self.assertEqual(errors, [])
        search.assert_called_once_with("pod pending")
        synthesize.assert_called_once_with(
            "pod pending",
            [result],
            "user-key",
            "gemini-3.1-flash",
            ["gemini-3.1-flash", "gemini-3.1-flash-lite"],
        )


if __name__ == "__main__":
    unittest.main()
