# Content Recommandation App

OpenClassrooms • [Ai Engineer track](https://openclassrooms.com/fr/paths/795-ai-engineer) • [Laetitia Ikusawa](https://www.linkedin.com/in/laetitia-ikusawa/)
***

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

***
## Installation

This project uses **Poetry** and requires **Python >= 3.11**.

### Prerequisites
- Install Python 3.11+ (check with `python --version`)
- Install Poetry (recommended via [`pipx`](https://pipx.pypa.io/stable/)):

```bash
pipx install poetry
poetry --version
```

### Install dependencies
From the project root:

```bash
poetry install --with dev
```

If Poetry is not using the right Python version, select one explicitly:

```bash
poetry env use 3.11
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

### Useful commands (dev)

```bash
poetry run pytest
poetry run ruff check .
poetry run black .
```
