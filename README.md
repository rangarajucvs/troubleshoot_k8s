# Kubernetes Issue Researcher

A non-conversational Streamlit app that searches online Kubernetes documentation,
GitHub issue pages, and community Q&A for a supplied issue or error. It presents
an evidence-informed, best-effort assessment with linked sources. It is not a
substitute for vendor guidance or testing changes in a safe environment.

## Run locally

1. Use Python 3.10 or newer.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

4. Paste your Gemini API key in the password-masked field, select **Load
   available Gemini models**, then choose a model from the dropdown. The list
   includes only Gemini 3 and Gemini 4 models enabled for your key that support
   generation. The default option uses the newest available version, preferring
   stable Pro models over Flash models of the same version for troubleshooting.
   The key is used only for the current submission and is cleared from the
   Streamlit session after either action; it is not written to source files or
   investigation results. Re-enter it for each action, and reload the model
   list if you use a different key. The issue text remains available after a
   submission.

The app searches Kubernetes, EKS, AWS re:Post Knowledge Center, GKE, AKS, OKE,
and Red Hat OpenShift documentation; selected Kubernetes/cloud-provider GitHub
issue pages; Reddit; Stack Overflow; and Server Fault. OpenShift documentation
searches include Red Hat's OpenShift Container Platform docs and the legacy
docs.openshift.com site. GitHub searches include repositories under the
`openshift` organization, AWS- and Istio-owned repositories, and Kubernetes
cloud-provider repositories.
The app uses the maintained `ddgs` package for discovery (the
`duckduckgo-search` project has been renamed to `ddgs`), fetches top matching
pages when available, and uses Gemini to summarize retrieved page text and
search excerpts. DDGS uses automatic metasearch with a 30-second timeout; the
app does not spoof TLS/JA3 fingerprints. If searches still time out, check
outbound access or proxy settings for the app's host.
Some sites (including community Q&A pages) may deny direct page access with
HTTP 403. In that case, the app keeps the search result and uses its excerpt
instead of retrying or attempting to bypass the site's access controls.
LangGraph runs the stateless search-then-summarize workflow, and LangChain's
Google Generative AI integration calls Gemini. The app does not use Google ADK:
its single fixed workflow does not need a second agent/orchestration runtime.
Search results and model output depend on external services and can be incomplete
or incorrect. The issue text is sent to DuckDuckGo and Google Gemini; do not
submit secrets, credentials, or sensitive cluster data. The Gemini key is sent
to Google's model API to list models and generate the response.
Model availability and pricing can change, so check
[Google's current Gemini API model documentation](https://ai.google.dev/gemini-api/docs/models)
and [pricing](https://ai.google.dev/gemini-api/docs/pricing).
The input is limited to 500 words, and the response is capped at 1,000 words.
The app does not maintain a conversation history; use **Clear and start a new
investigation** to remove the current result and input.

## Tests

Run the focused unit tests with:

```bash
python -m unittest discover -s tests
```
# troubleshoot_k8s
