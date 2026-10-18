import os

import streamlit as st

from k8s_research import MAX_ANSWER_WORDS, MAX_QUERY_WORDS, research_issue


st.set_page_config(page_title="Kubernetes Issue Researcher", page_icon="☸️", layout="wide")
st.title("Kubernetes Issue Researcher")
st.write(
    "Enter a Kubernetes error or issue. The agent searches upstream and cloud-provider "
    "documentation, relevant GitHub issues, Reddit, Stack Overflow, and Server Fault, "
    "then provides best-effort next steps. No conversation history is kept."
)
st.caption(
    "Best-effort guidance only. Verify recommendations against your cluster, "
    "environment, and current vendor documentation."
)
st.caption(
    "Issue text is sent to DuckDuckGo for search and to Google Gemini for synthesis. "
    "Do not include credentials, secrets, or sensitive cluster data."
)


def clear_investigation() -> None:
    st.session_state.clear()


with st.form("investigation_form", clear_on_submit=False):
    issue_text = st.text_area(
        "Kubernetes issue or error",
        key="issue_input",
        height=180,
        placeholder="Paste the error message and relevant context here...",
        help=f"Maximum {MAX_QUERY_WORDS} words.",
    )
    submitted = st.form_submit_button("Research issue", type="primary")

st.button("Clear and start a new investigation", on_click=clear_investigation)

if submitted:
    st.session_state.pop("investigation", None)
    issue = issue_text.strip()
    word_count = len(issue.split())
    if not issue:
        st.error("Enter an issue or error message to investigate.")
    elif word_count > MAX_QUERY_WORDS:
        st.error(f"The issue cannot exceed {MAX_QUERY_WORDS} words (currently {word_count}).")
    elif not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        st.error("Set GOOGLE_API_KEY or GEMINI_API_KEY before researching an issue.")
    else:
        try:
            with st.spinner("Searching Kubernetes sources and preparing a response..."):
                answer, results, search_errors = research_issue(issue)
                if not results:
                    if search_errors:
                        with st.expander("Some source searches or pages could not be retrieved"):
                            for error in search_errors:
                                st.warning(error)
                    st.error(
                        "No search results were retrieved, so a source-grounded response "
                        "could not be prepared."
                    )
                else:
                    st.session_state["investigation"] = {
                        "answer": answer,
                        "results": results,
                        "search_errors": search_errors,
                    }
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))

investigation = st.session_state.get("investigation")
if investigation:
    st.subheader("Best-effort assessment")
    answer = investigation["answer"]
    if len(answer.split()) > MAX_ANSWER_WORDS:
        answer = " ".join(answer.split()[:MAX_ANSWER_WORDS])
    st.markdown(answer)
    st.caption(f"Response length: {len(answer.split())} / {MAX_ANSWER_WORDS} words")

    if investigation["search_errors"]:
        with st.expander("Some source searches or pages could not be retrieved"):
            for error in investigation["search_errors"]:
                st.warning(error)

    st.subheader("Sources found")
    for index, result in enumerate(investigation["results"], start=1):
        st.markdown(f"{index}. [{result.title}]({result.url}) — *{result.source}*")
        excerpt = result.page_excerpt or result.snippet
        if excerpt:
            st.caption(excerpt[:600])
