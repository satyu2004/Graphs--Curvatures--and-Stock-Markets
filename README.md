Graph Structures in Markets

Interactive Streamlit app for visualizing stock correlation networks and graph curvatures.

Run locally

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (create a new repo).

```bash
git init
git add .
git commit -m "Add Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app" → select the repository, branch `main`, and set the main file path to `app.py`.
4. Streamlit will install packages from `requirements.txt` and deploy the app.

Notes

- If your `src/data/india/data_clean_interpolated.csv` file is large, consider removing it from the repo and loading from cloud storage to avoid slow deploys or exceeding platform limits.
- Use Streamlit Secrets (Settings → Secrets) for any credentials.
