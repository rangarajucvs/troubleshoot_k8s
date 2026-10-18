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

3. Set a Google AI Studio API key. By default, the app checks the models
   available to that key and selects the newest stable Gemini Flash-Lite model
   that supports text generation. To pin a particular available model:

   ```bash
   export GOOGLE_API_KEY="your-api-key"
   export GEMINI_MODEL="gemini-3.1-flash-lite"
   ```

   `GEMINI_API_KEY` is also accepted in place of `GOOGLE_API_KEY`.

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

The app searches Kubernetes, EKS, GKE, AKS, and OKE documentation; selected
Kubernetes/cloud-provider GitHub issue pages; Reddit; Stack Overflow; and
Server Fault. It uses DuckDuckGo's HTML search endpoint for discovery, fetches
the top matching page in each source group when available, and uses Gemini to
summarize retrieved page text and search excerpts.
LangGraph runs the stateless search-then-summarize workflow, and LangChain's
Google Generative AI integration calls Gemini. The app does not use Google ADK:
its single fixed workflow does not need a second agent/orchestration runtime.
Search results and model output depend on external services and can be incomplete
or incorrect. The issue text is sent to DuckDuckGo and Google Gemini; do not
submit secrets, credentials, or sensitive cluster data.
Flash-Lite is Google's cost-focused Gemini family; its
[Gemini 3.1 Flash-Lite model](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
is described as cost-effective. Model pricing can change, so check
[Google's current Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing).
The input is limited to 500 words, and the response is capped at 1,000 words.
The app does not maintain a conversation history; use **Clear and start a new
investigation** to remove the current result and input.

## Tests

Run the focused unit tests with:

```bash
python -m unittest discover -s tests
```
# troubleshoot_k8s
