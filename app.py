import streamlit as st

from k8s_research import (
    AUTO_MODEL_OPTION,
    MAX_ANSWER_WORDS,
    MAX_QUERY_WORDS,
    list_gemini_models,
    research_issue,
)


st.set_page_config(page_title="Kubernetes Issue Researcher", page_icon="☸️", layout="wide")
st.title("Kubernetes Issue Researcher")
st.write(
    "Enter a Kubernetes error or issue. The agent searches upstream, cloud-providers documentations [EKS, GKE, AKS, OKE],"
    "Red Hat OpenShift documentation, Istio, relevant GitHub issues, Reddit, "
    "Stack Overflow, and Server Fault, then provides best-effort next steps. "
    "No conversation history is kept."
)
st.caption(
    "Best-effort guidance only. Verify recommendations against your cluster, "
    "environment, and current vendor documentation."
)
st.caption(
    "Issue text is sent to DuckDuckGo for search and to Google Gemini for synthesis. "
    "Do not include credentials, secrets, or sensitive cluster data."
)


available_models = st.session_state.get("available_gemini_models", [])
model_options = [AUTO_MODEL_OPTION, *available_models]
if st.session_state.get("gemini_model") not in model_options:
    st.session_state["gemini_model"] = AUTO_MODEL_OPTION
st.selectbox("Gemini model", model_options, key="gemini_model")

issue_text = st.text_area(
    "Kubernetes issue or error",
    key="issue_input",
    height=180,
    placeholder="Paste the error message and relevant context here...",
    help=(
        f"Maximum {MAX_QUERY_WORDS} words. Double quotes are allowed, "
        'for example: "Failed to create pod".'
    ),
)
with st.form("gemini_key_form", clear_on_submit=True):
    st.text_input(
        "Gemini API key",
        type="password",
        key="gemini_api_key",
        help=(
            "Your key is used for this submission and cleared from the session "
            "when you load models or research an issue."
        ),
    )
    load_models = st.form_submit_button("Load available Gemini models")
    submitted = st.form_submit_button("Research issue", type="primary")

if load_models:
    api_key = st.session_state.get("gemini_api_key", "").strip()
    if not api_key:
        st.error("Enter your Gemini API key before loading models.")
        st.session_state.pop("available_gemini_models", None)
    else:
        try:
            with st.spinner("Loading models available to this API key..."):
                st.session_state["available_gemini_models"] = list_gemini_models(api_key)
            st.session_state["gemini_model"] = AUTO_MODEL_OPTION
        except RuntimeError as exc:
            st.session_state.pop("available_gemini_models", None)
            st.error(str(exc))

def clear_investigation() -> None:
    st.session_state.clear()


st.button("Clear and start a new investigation", on_click=clear_investigation)

if submitted:
    st.session_state.pop("investigation", None)
    issue = issue_text.strip()
    word_count = len(issue.split())
    if not issue:
        st.error("Enter an issue or error message to investigate.")
    elif word_count > MAX_QUERY_WORDS:
        st.error(f"The issue cannot exceed {MAX_QUERY_WORDS} words (currently {word_count}).")
    elif not st.session_state.get("gemini_api_key", "").strip():
        st.error("Enter your Gemini API key before researching an issue.")
    elif not available_models:
        st.error("Load the models available to your API key before researching an issue.")
    else:
        try:
            with st.spinner("Searching Kubernetes sources and preparing a response..."):
                selected_model = st.session_state["gemini_model"]
                answer, results, _ = research_issue(
                    issue,
                    api_key=st.session_state["gemini_api_key"].strip(),
                    model=None if selected_model == AUTO_MODEL_OPTION else selected_model,
                    available_models=available_models,
                )
                if not results:
                    st.error(
                        "No search results were retrieved, so a source-grounded response "
                        "could not be prepared."
                    )
                else:
                    st.session_state["investigation"] = {
                        "answer": answer,
                        "results": results,
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

    st.subheader("Sources found")
    for index, result in enumerate(investigation["results"], start=1):
        st.markdown(f"{index}. [{result.title}]({result.url}) — *{result.source}*")
        excerpt = result.page_excerpt or result.snippet
        if excerpt:
            st.caption(excerpt[:600])
        if result.page_note:
            st.caption(result.page_note)
