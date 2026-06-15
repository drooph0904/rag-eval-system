# Deploying the click-to-try demo

The hosted demo is **read-only**: it shows a finished evaluation (the **Load sample
demo** button) on a protein-design paper, with zero API keys. The live "Run
evaluation" path needs a local clone + keys (see the main README) and does **not**
run on Streamlit Cloud, because `pipeline_runner.py` shells out to a local `.venv`
and calls paid APIs.

## Streamlit Community Cloud (free, ~3 min)

1. Push this branch to GitHub (e.g. `demo-hosting`).
2. Go to **https://share.streamlit.io** → sign in with GitHub → **Create app**.
3. Point it at:
   - **Repository:** `drooph0904/rag-eval-system`
   - **Branch:** `demo-hosting`
   - **Main file path:** `ui/app.py`
4. **Advanced settings → Python dependencies file:** set to `requirements-demo.txt`
   (the light, fast set). If your Cloud version can't pick the file, just rename
   `requirements-demo.txt` → `requirements.txt` on this branch before deploying.
5. **Deploy.** You get a public URL like
   `https://rag-eval-system.streamlit.app` — that's the link for the DM.

The landing page tells the visitor to click **Load sample demo** in the sidebar;
the full Stage 1 → Stage 2 → Winner → per-question drill-down renders instantly.

## Make the numbers real before sharing (recommended)

The bundled sample (`results/sample_protein_design.json`) is **illustrative**. For authentic numbers, run the engine once locally on a real
protein-design PDF, then commit the output as the sample:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt
cp .env.example .env          # add OPENAI_API_KEY and COHERE_API_KEY
# generate golden Q&A + evaluate end to end via the UI:
.venv/bin/streamlit run ui/app.py
# upload the paper, click "Run full evaluation", let it finish, then:
cp results/<your_paper>_eval_results.json results/sample_protein_design.json
git commit -am "Use a real evaluation run as the demo sample" && git push
```

Now the hosted demo shows a real run — strongest possible artifact for the DM.
