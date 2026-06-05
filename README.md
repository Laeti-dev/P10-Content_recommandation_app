# Content Recommandation App

OpenClassrooms • [Ai Engineer track](https://openclassrooms.com/fr/paths/795-ai-engineer) • [Laetitia Ikusawa](https://www.linkedin.com/in/laetitia-ikusawa/)

---

## Business Objetives

- **Deliver personalized content recommendations** to improve user engagement and satisfaction.
- **Facilitate scalable onboarding** of new users and articles to support company growth.
- **Enhance the quality and relevance of content suggestions** using data-driven AI models.
- **Enable seamless integration with cloud-based infrastructure** for efficient deployment and maintenance.
- **Provide a user-friendly application** that aligns with business goals and customer needs.
- **Optimize recommendation accuracy** to increase retention and content consumption.
- **Support the start-up’s strategic vision** by developing a minimum viable product (MVP) quickly and iteratively.

## Learning objectives

- Develop skills in data manipulation and preprocessing to **prepare datasets for recommendation systems**.
- Acquire knowledge of **statistical modeling and machine learning techniques** for content recommendation.
- Gain experience in **designing and implementing scalable, serverless architectures using cloud services**.
- Improve software development practices by **structuring and documenting code** effectively.
- Build abilities in **deploying web applications** using frameworks like Flask or Streamlit.
- Enhance **communication skills** through preparing presentations and demonstrating project outcomes.

*---

## 🏗️ Architecture

```
Streamlit App
     ↓ HTTP GET /recommend/{user_id}?beta=0.8&topk=5
Azure Function (serverless · Python 3.12)
     ↓ cold start: loads artifacts once from Blob Storage
Azure Blob Storage
     - candidate_embeddings.pkl  (PCA 33 components, 85% variance retained)
     - user_profiles.pkl
     - popularity_scores.pkl
     - candidate_ids.pkl / candidate_norms.pkl / article_to_idx.pkl
     - user_seen.pkl / pca_final.pkl / manifest.json
     ↑ warm start: global cache in memory
Azure Function
     ↓ JSON {"user_id": x, "recommendations": [...]}
Streamlit App
```

---

## 📊 Dataset


| Property   | Value                                                                                                                     |
| ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| Source     | [Globo.com](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom) — Brazilian news portal |
| Articles   | 364,047                                                                                                                   |
| Users      | 322,897                                                                                                                   |
| Clicks     | ~3M over 43 days                                                                                                          |
| Embeddings | 250D vectors (1D CNN)                                                                                                     |
| Period     | 01/10/2017 → 13/11/2017                                                                                                   |


**Temporal split:**

- Train: 01/10 → 09/10 (881,392 clicks · 64,734 users · ~14 clicks/user)
- Test:  10/10 → 17/10 (771,758 clicks · ~12 clicks/user)

> **Key insight**: 77% of articles read in the test set are new articles, unseen during training. This makes Collaborative Filtering structurally limited on this dataset.

---

## 🤖 Models

### Content-Based (CB)


| Strategy | Hit@5 | Soft@5 |
| -------- | ----- | ------ |
| Mean     | 2.47% | 50.93% |
| Recency  | 2.58% | 50.96% |
| Category | 2.59% | 51.04% |


### Collaborative Filtering (CF) — `implicit` library


| Strategy  | Hit@5 | Soft@5 |
| --------- | ----- | ------ |
| Item-Item | 1.01% | 13.56% |
| ALS       | 1.05% | 15.09% |
| BPR       | 0.84% | 13.55% |
| LMF       | 0.65% | 8.17%  |


### Hybrid & Final Model


| Model                                | Hit@5      | Soft@5     |
| ------------------------------------ | ---------- | ---------- |
| Hybrid CB + ALS                      | 0.44%      | 49.78%     |
| **CB Category + Popularity (β=0.8)** | **48.18%** | **74.09%** |


> The CB + recent popularity combination is the retained model — **×18 improvement over CB alone**, **×4,800 over random baseline**.

### Dimensionality Reduction (PCA)

Original embeddings (250D, 424 MB) reduced to 33 components (85% variance retained, 54 MB) with only -1.5% loss on Hit@5.

---

## 🗂️ Project Structure

```
P10-content_recommandation_app/
├── azure_function/
│   ├── function_app.py          # Azure Function entry point
│   ├── recommender.py           # Production inference engine
│   ├── requirements.txt         # Azure dependencies
│   ├── host.json                # Azure Functions runtime config
│   └── local.settings.json      # Local env variables (not committed)
├── app/
│   ├── streamlit_app.py         # Management interface
│   └── templates/
│       └── index.html           # FastAPI template (alternative)
├── notebooks/
│   ├── eda.ipynb                # Exploratory data analysis
│   └── modeling.ipynb           # Model experiments & training
├── scripts/
│   └── export_artifacts.py      # Export artifacts → Azure Blob Storage
├── documentation/
│   └── learnings.md             # Full theoretical memo (CB, CF, Hybrid, Serverless)
├── data/
│   ├── articles_metadata.csv
│   ├── articles_embeddings.pickle
│   └── clicks/                  # 385 CSV files
├── loaders.py                   # DataLoader
├── constants.py                 # Global constants
├── pyproject.toml               # Poetry dependencies
└── README.md
```

---

## 🚀 Installation

This project uses **Poetry** and requires **Python >= 3.11**.

### Prerequisites

- Install Python 3.12 (check with `python --version`)
- Install Poetry (recommended via `[pipx](https://pipx.pypa.io/stable/)`):

```bash
pipx install poetry
poetry --version
```

- Install [Azure Functions Core Tools v4](https://learn.microsoft.com/en-us/azure/azure-functions/functions-run-local)

### Install dependencies

From the project root:

```bash
poetry install --with dev
```

If Poetry is not using the right Python version, select one explicitly:

```bash
poetry env use 3.12
poetry install --with dev
```

### Activate the virtual environment

Either spawn a shell:

```bash
poetry shell
```

Or prefix commands with `poetry run`:

```bash
poetry run python -V
```

### Environment variables

Create a `.env` file at the project root:

```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_FUNCTION_URL=http://localhost:7071/api
```

---

## 💻 Running Locally

### 1. Azure Function

```bash
cd azure_function
# Configure local.settings.json with your connection string
poetry run func start
```

The Function is available at `http://localhost:7071/api`

### 2. Streamlit App

```bash
AZURE_FUNCTION_URL=http://localhost:7071/api poetry run streamlit run app/streamlit_app.py
```

### 3. Test the endpoints

```bash
# Health check
curl "http://localhost:7071/api/health"

# Recommendations
curl "http://localhost:7071/api/recommend/5890?beta=0.8&topk=5"
```

### Useful commands (dev)

```bash
poetry run pytest
poetry run ruff check .
poetry run black .
```

---

## ☁️ Azure Deployment

### Export artifacts → Blob Storage

From the notebook, after fitting the model:

```python
# See notebooks/modeling.ipynb — "Export artifacts" cell
# Artifacts are serialized with pickle and uploaded directly from memory
```

### Deploy Azure Function

```bash
cd azure_function
poetry run func azure functionapp publish <your-function-app-name>
```

### Production environment variables

In **Azure Portal → Function App → Settings → Environment variables**:

```
AZURE_STORAGE_CONNECTION_STRING = <your connection string>
ARTIFACTS_CONTAINER             = artifacts   (default)
DEFAULT_BETA                    = 0.8         (default)
DEFAULT_TOPK                    = 5           (default)
```

---

## 🔌 API Reference

### `GET /api/recommend/{user_id}`

Returns top-K recommended article IDs for a given user.


| Parameter | Type  | Default | Description                                      |
| --------- | ----- | ------- | ------------------------------------------------ |
| `user_id` | int   | —       | User ID (required)                               |
| `beta`    | float | 0.8     | Popularity weight (0=CB only, 1=popularity only) |
| `topk`    | int   | 5       | Number of recommendations                        |


**Response:**

```json
{
  "user_id": 5890,
  "recommendations": [160974, 96210, 234698, 336221, 331116],
  "beta": 0.8,
  "topk": 5
}
```

### `GET /api/health`

```json
{
  "status": "ok",
  "n_articles": 364047,
  "n_users": 64734,
  "model": "CBRecommender + popularity"
}
```

---

## 🗺️ Target Architecture (V2)

To address V1 limitations (static artifacts, cold start for new users and articles):

```
New articles (CMS)   ──►  Azure Event Hub  ──►  Azure Stream Analytics
New clicks           ──►                         - Article → embedding (Azure Function trigger)
                                                 - Click   → update CB profile + adaptive beta
                                                       ↓
                                            Azure Blob Storage (versioned)
                                                       ↓
                                            Azure Function (unchanged)
```

**Adaptive beta strategy** — as the user profile grows richer, beta decreases to favor semantic personalization over global popularity:


| Clicks | Beta     | Strategy                              |
| ------ | -------- | ------------------------------------- |
| 0      | 1.0      | Pure popularity (trending articles)   |
| 1–5    | 0.9      | CB session profile + slight coloring  |
| 5–20   | 0.8      | Full CB profile — V1 model            |
| 20+    | 0.5      | Rich profile — strong personalization |
| 50+    | adaptive | Stable profile — near full CB         |


---

## 📚 Documentation

- `[documentation/learnings.md](documentation/learnings.md)` — Full theoretical memo (CB, CF, Hybrid, Serverless, PCA)

---

## ⚠️ Important Notes

- `local.settings.json` and `.env` contain secrets — **never commit to GitHub**
- Add `azure_function/.python_packages/` to `.gitignore`
- Azure Function cold start may take 20–30 seconds (loading ~54 MB artifacts from Blob Storage)
- `pca_final.pkl` must be kept alongside `candidate_embeddings.pkl` — it is required to project new article embeddings into the reduced 33D space

---

*AI Engineer Track — OpenClassrooms | April 2026*