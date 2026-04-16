# Comprendre le système de recommandation content-based

## Table des matières

1. [Vue d'ensemble de l'architecture](#1-vue-densemble-de-larchitecture)
2. [Les données disponibles](#2-les-données-disponibles)
3. [Le concept clé : les embeddings](#3-le-concept-clé--les-embeddings)
4. [Comment fonctionne la recommandation content-based](#4-comment-fonctionne-la-recommandation-content-based)
5. [Détail des scripts](#5-détail-des-scripts)
6. [Les métriques d'évaluation](#6-les-métriques-dévaluation)
7. [Comment tester le modèle](#7-comment-tester-le-modèle)
8. [Pièges fréquents à connaître](#8-pièges-fréquents-à-connaître)

---

## 1. Vue d'ensemble de l'architecture

Le projet suit un pattern **orienté objet avec classe abstraite**. Voici comment les trois scripts s'articulent :

```
Recommender (ABC)                  ← recommender.py
    │  méthode abstraite: recommend()
    │  méthode concrète:  evaluate()
    │
    ├── ContentBasedRecommender    ← content_base_recommender.py
    └── PopularityRecommender      ← popularity_recommender.py

DataLoader                         ← loaders.py
    │  charge les données depuis data/
    └── utilisé par ContentBasedRecommender
```

La classe `Recommender` est une **classe abstraite** (ABC = Abstract Base Class). Elle définit le contrat que tout recommandeur doit respecter :
- Il **doit** implémenter la méthode `recommend()`.
- Il **hérite gratuitement** de la méthode `evaluate()` qui calcule les métriques.

---

## 2. Les données disponibles

Le `DataLoader` gère trois types de données :

| Donnée | Fichier | Description |
|--------|---------|-------------|
| Métadonnées des articles | `data/articles_metadata.csv` | `article_id`, `category_id`, `created_date`, etc. |
| Embeddings des articles | `data/articles_embeddings.pickle` | Matrice NumPy de forme `(n_articles, dim)` — un vecteur par article |
| Interactions utilisateurs | `data/clicks/clicks_hour_*.csv` | Chaque ligne = un clic d'un `user_id` sur un `click_article_id` |

**Point important :** l'index de la matrice d'embeddings correspond directement à l'`article_id`. Ainsi `matrix[42]` est le vecteur de l'article 42.

---

## 3. Le concept clé : les embeddings

Un **embedding** est une représentation numérique dense d'un article sous forme de vecteur (ex : 250 dimensions). Deux articles au contenu similaire auront des vecteurs proches dans l'espace vectoriel.

```
Article "Football - Coupe du monde" → [0.12, -0.45, 0.78, ...]  (250 valeurs)
Article "Rugby - Top 14"            → [0.10, -0.42, 0.81, ...]  (proche !)
Article "Recette de tarte aux pommes" → [-0.33, 0.91, -0.12, ...] (loin)
```

Ces embeddings ont été pré-calculés et stockés dans `articles_embeddings.pickle`.

---

## 4. Comment fonctionne la recommandation content-based

L'idée centrale est de répondre à la question : *"Quels articles ressemblent le plus à ce que l'utilisateur a déjà lu ?"*

### Étape 1 : Construire le profil utilisateur

On récupère tous les articles que l'utilisateur a lus, on prend leurs vecteurs d'embeddings, et on en fait la **moyenne**. Ce vecteur moyen représente le "goût" de l'utilisateur.

```python
# Exemple conceptuel
user_history = [article_12, article_45, article_89]
user_profile = mean([embedding[12], embedding[45], embedding[89]])
# user_profile est un vecteur unique qui résume les préférences
```

### Étape 2 : Calculer la similarité cosinus

Pour chaque article candidat (non encore lu), on calcule la **similarité cosinus** entre le profil utilisateur et l'article.

La similarité cosinus mesure l'angle entre deux vecteurs :
- **Score de 1.0** → vecteurs identiques (article parfaitement similaire)
- **Score de 0.0** → vecteurs perpendiculaires (aucun lien)
- **Score de -1.0** → vecteurs opposés (thèmes opposés)

```
cosine_similarity(user_profile, article_vector) = cos(θ)
```

### Étape 3 : Trier et retourner les top-K

On trie tous les articles par score décroissant et on retourne les K meilleurs.

---

## 5. Détail des scripts

### `loaders.py` — Le gestionnaire de données

Le `DataLoader` utilise le **lazy loading** : les données ne sont chargées qu'à la première demande, puis mises en cache (attribut `_articles_metadata`, `_articles_embeddings`, etc.).

Méthodes essentielles :

```python
loader = DataLoader()

# Charge la matrice (n_articles, dim)
matrix = loader.load_article_embeddings_matrix()

# Charge tout l'historique de clics
interactions = loader.load_user_interactions()

# Historique d'un utilisateur spécifique, trié du plus récent
history = loader.get_user_history(user_id=12345)

# Métadonnées d'un article
info = loader.get_article_info(article_id=42)
```

---

### `recommender.py` — La classe abstraite de base

La méthode la plus importante à comprendre est `evaluate()`. Elle :

1. Appelle `prepare_embeddings()` pour préparer la matrice (optimisation mémoire).
2. Pour chaque utilisateur du jeu de test, récupère ses **vrais articles** (ground truth).
3. Génère les **K recommandations** du modèle.
4. Compare les deux ensembles et calcule les métriques.

```python
# Schéma de la boucle d'évaluation
for user_id in test_data["user_id"].unique():
    true_items  = { articles réellement cliqués dans le test }
    recommended = recommender.recommend(user_id, num_recommendations=k)
    top_k       = { les k articles recommandés }

    hits      = |true_items ∩ top_k|  # intersection
    precision = hits / k
    recall    = hits / len(true_items)
    f1        = 2 * precision * recall / (precision + recall)
```

---

### `content_base_recommender.py` — L'algorithme principal

Ce fichier contient deux méthodes de recommandation :

#### `_recommend_prepared()` — Mode rapide (utilisé pendant l'évaluation)

Appelé quand `prepare_embeddings()` a été invoqué au préalable. La matrice est pré-filtrée aux seuls articles du train + test, ce qui accélère les calculs.

```
prepare_embeddings(test_data)   → filtre la matrice aux articles pertinents
_recommend_prepared(user_id)    → utilise cette matrice restreinte
```

#### `_recommend_from_full_catalog()` — Mode catalogue complet

Utilisé en production (recommandation en temps réel) quand on n'a pas pré-calculé de matrice restreinte. Parcourt tous les articles du catalogue.

#### `prepare_embeddings()` — Optimisation mémoire

Cette méthode construit un `DataFrame` indexé par `article_id`, filtré uniquement aux articles présents dans le train **ou** le test. Cela évite de charger tous les ~100k articles en mémoire pour chaque utilisateur.

---

## 6. Les métriques d'évaluation

La méthode `evaluate()` retourne 4 métriques, toutes calculées **@K** (sur les K premières recommandations) :

### Hit@K
**"Est-ce qu'au moins un article recommandé est correct ?"**
- Vaut 1 si au moins 1 article recommandé est dans la ground truth, 0 sinon.
- C'est une métrique binaire par utilisateur, on en fait la moyenne.

### Precision@K
**"Parmi mes K recommandations, quelle fraction est correcte ?"**
```
Precision@K = (nombre de hits parmi les K recommandations) / K
```
- Ex : 2 bons articles sur 5 recommandés → Precision@5 = 0.40

### Recall@K
**"Parmi tous les vrais articles, quelle fraction j'ai retrouvée ?"**
```
Recall@K = (nombre de hits) / (nombre total de vrais articles dans le test)
```
- Ex : 2 bons articles retrouvés sur 10 vrais → Recall@10 = 0.20

### F1@K
**Compromis entre Precision et Recall** (moyenne harmonique).
```
F1@K = 2 * Precision@K * Recall@K / (Precision@K + Recall@K)
```

> **Interprétation pratique :** un Hit@5 de 0.35 signifie que pour 35% des utilisateurs, au moins un des 5 articles recommandés était réellement cliqué dans le test. C'est la métrique la plus intuitive pour commencer.

---

## 7. Comment tester le modèle

### Prérequis

```python
from recommenders.loaders import DataLoader
from recommenders.content_base_recommender import ContentBasedRecommender
import pandas as pd
```

### Étape 1 : Charger les données et créer un split train/test

```python
loader = DataLoader()
interactions = loader.load_user_interactions()

# Sort by time to avoid data leakage
interactions = interactions.sort_values("click_timestamp")

# 80% train, 20% test (temporal split)
split_idx = int(len(interactions) * 0.8)
train_data = interactions.iloc[:split_idx]
test_data  = interactions.iloc[split_idx:]

print(f"Train: {len(train_data):,} interactions")
print(f"Test:  {len(test_data):,} interactions")
```

### Étape 2 : Instancier le recommandeur

```python
recommender = ContentBasedRecommender(
    data_loader=loader,
    train_data=train_data,
    k=5,                            # nombre de recommandations
    item_col="click_article_id",    # colonne article dans les données
)
```

### Étape 3 : Tester la recommandation pour un utilisateur

```python
# Récupérer un utilisateur actif
active_users = loader.get_most_active_users(limit=5)
user_id = active_users[0]["user_id"]

# Générer des recommandations
recommendations = recommender.recommend(user_id, num_recommendations=5)

for rec in recommendations:
    print(f"Article {rec['article_id']} | Score: {rec['score']:.4f} | {rec['reason']}")
```

### Étape 4 : Évaluer le modèle sur le jeu de test

```python
# Limiter à un sous-ensemble d'utilisateurs pour aller plus vite
sample_users = test_data["user_id"].unique()[:200]
test_sample  = test_data[test_data["user_id"].isin(sample_users)]

metrics = recommender.evaluate(test_sample)
print(metrics)
# {
#   "Hit@5":       0.3500,
#   "Precision@5": 0.0900,
#   "Recall@5":    0.2100,
#   "F1@5":        0.1260,
# }
```

### Étape 5 : Comparer avec la baseline de popularité

```python
from recommenders.popularity_recommender import PopularityRecommender

# Préparer les données pour le recommandeur de popularité
# (il attend une colonne "article_id" et "category_id")
metadata = loader.load_articles_metadata()
train_with_meta = train_data.merge(
    metadata[["article_id", "category_id"]],
    left_on="click_article_id",
    right_on="article_id",
    how="left"
)

popularity_rec = PopularityRecommender(train_df=train_with_meta)
popularity_rec.fit_with_holdout()

# Évaluation manuelle car PopularityRecommender retourne des int, pas des dicts
# (son interface diffère légèrement de ContentBasedRecommender)
```

---

## 8. Pièges fréquents à connaître

### Le split doit être temporel
Toujours trier par `click_timestamp` avant de splitter. Si on coupe aléatoirement, on crée une **fuite de données** (data leakage) : le modèle verrait des articles "futurs" à l'entraînement.

### `prepare_embeddings()` doit être appelé avant `evaluate()`
La méthode `evaluate()` de la classe parente appelle `prepare_embeddings()` automatiquement. Mais si on appelle `recommend()` directement sans passer par `evaluate()`, la matrice restreinte n'est pas construite et le modèle tombe sur `_recommend_from_full_catalog()` (plus lent mais correct).

### Les utilisateurs sans historique d'entraînement sont exclus
Si un utilisateur du jeu de test n'a aucune interaction dans le train, `_recommend_prepared()` retourne une liste vide. C'est le problème du **cold start** — ce modèle content-based ne le gère pas.

### La similarité cosinus n'est pas affectée par la magnitude
Deux articles avec des embeddings de normes très différentes peuvent quand même avoir une similarité cosinus élevée si leur **direction** est similaire. C'est un avantage : pas besoin de normaliser les embeddings au préalable.

---

*Documentation générée le 16 avril 2026.*
