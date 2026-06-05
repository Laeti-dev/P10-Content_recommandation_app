# Mémo Apprentissages — Projet 10 : Système de Recommandation de Contenu

> Laetitia Ikusawa — Parcours AI Engineer  
> Dataset : Globo.com (portail de news brésilien)

---

## Table des matières

1. [Les systèmes de recommandation — théorie générale](https://claude.ai/chat/f32a4070-017f-4f2b-bd33-5cddd22c1681#1-les-syst%C3%A8mes-de-recommandation--th%C3%A9orie-g%C3%A9n%C3%A9rale)
2. [Content-Based Filtering (CB)](https://claude.ai/chat/f32a4070-017f-4f2b-bd33-5cddd22c1681#2-content-based-filtering-cb)
3. [Collaborative Filtering (CF)](https://claude.ai/chat/f32a4070-017f-4f2b-bd33-5cddd22c1681#3-collaborative-filtering-cf)
4. [Approche Hybride](https://claude.ai/chat/f32a4070-017f-4f2b-bd33-5cddd22c1681#4-approche-hybride)
5. [Ce que les résultats m'ont appris](https://claude.ai/chat/f32a4070-017f-4f2b-bd33-5cddd22c1681#5-ce-que-les-r%C3%A9sultats-mont-appris)
6. [Le Serverless](https://claude.ai/chat/f32a4070-017f-4f2b-bd33-5cddd22c1681#6-le-serverless)

---

## 1. Les systèmes de recommandation — théorie générale

### Définition

Un système de recommandation est un outil qui prédit les contenus (articles, films, produits) qu'un utilisateur est susceptible d'apprécier, à partir de ses comportements passés et/ou des caractéristiques des contenus.

### Les trois grandes familles


| Famille                 | Principe                                                  | Exemple                                                                     |
| ----------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------- |
| Content-Based           | Recommande ce qui ressemble à ce que l'utilisateur a aimé | "Tu as lu des articles sur le sport → voici d'autres articles sur le sport" |
| Collaborative Filtering | Recommande ce qu'ont aimé des utilisateurs similaires     | "Des gens comme toi ont aussi lu ça"                                        |
| Hybride                 | Combine les deux                                          | Netflix, Spotify                                                            |


### Les données utilisées dans ce projet

- **Embeddings d'articles** : vecteurs de 250 dimensions générés par un réseau de neurones 1D CNN à partir du texte des articles. Chaque article est représenté par un point dans un espace sémantique — deux articles proches dans cet espace partagent un sujet similaire.
- **Métadonnées** : catégorie, date de publication, nombre de mots.
- **Clics** : 385 fichiers CSV représentant les interactions utilisateur-article sur 43 jours (01/10/17 → 13/11/17).

### Le problème du Cold Start

Le cold start est le principal défi des systèmes de recommandation. Il se pose dans deux situations :

- **Nouvel utilisateur** : pas d'historique → impossible de construire un profil ou de trouver des utilisateurs similaires.
- **Nouvel article** : pas encore cliqué → le CF ne peut pas le recommander. Le CB peut le faire immédiatement grâce à son embedding.

Sur un dataset news, le cold start article est critique : les articles ont une durée de vie de quelques heures et 77% des articles lus chaque jour sont nouveaux.

### Métriques d'évaluation utilisées

**Hit@K (ou Acc@K)** : pour chaque utilisateur, on pose la question "est-ce qu'au moins 1 des K articles recommandés correspond à ce que l'utilisateur a réellement lu ?"

```
Hit = 1 si oui, 0 si non
Hit@K = moyenne sur tous les utilisateurs

```

C'est une métrique binaire, indulgente : elle ne distingue pas si le bon article est en position 1 ou en position K.

**Soft Hit@K** : version sémantique du Hit Rate.

- 1.0 si match exact
- 0.5 si la cosine similarity entre les recommandations et le profil utilisateur dépasse un seuil (0.7)
- 0.0 sinon

**Recall@K** : parmi tous les vrais articles lus, combien sont retrouvés dans le Top-K ?

```
Recall@K = hits dans Top-K / nombre de vrais articles lus

```

**Split temporel** : le train couvre le 01/10 au 09/10, le test couvre le 10/10 au 17/10. Ce split simule la vraie situation de production : le modèle ne connaît pas les articles futurs.

---

## 2. Content-Based Filtering (CB)

### Principe général

Le CB construit un **profil sémantique** pour chaque utilisateur à partir des embeddings des articles qu'il a lus, puis recommande les articles dont l'embedding est le plus proche de ce profil.

Le scoring utilise la **similarité cosine** :

```
score(user, article) = cos(profil_user, embedding_article)
                     = (profil · embedding) / (||profil|| × ||embedding||)

```

La cosine similarity mesure l'angle entre deux vecteurs — elle vaut 1 si les vecteurs pointent dans la même direction (articles très similaires) et 0 s'ils sont perpendiculaires.

### Stratégie 1 : Mean (Moyenne)

**Principe** : le profil utilisateur est la moyenne simple de tous les embeddings des articles lus.

```python
user_profile = user_embeddings.mean(axis=0)

```

**Avantages** : simple, stable, peu sensible à un clic aberrant si l'historique est riche.

**Inconvénients** : effet de lissage — mélange tous les intérêts sans distinction. Un utilisateur qui lit 80% sport et 20% politique aura un profil "dilué". Pas réactif aux changements récents.

### Stratégie 2 : Recency (Récence)

**Principe** : pondération exponentielle des embeddings selon leur ancienneté. Les articles récents ont plus de poids.

```python
deltas_days = (max_ts - timestamps) / np.timedelta64(1, 'D')
weights = np.exp(-deltas_days / half_life_days)
weights /= weights.sum()
user_profile = np.dot(weights, user_embeddings)

```

Le paramètre `half_life_days` contrôle la vitesse de décroissance : avec `half_life_days=7`, un article vieux d'une semaine vaut ~37% d'un article d'aujourd'hui.

**Avantages** : plus réactif aux changements d'intérêt récents. Pertinent sur un dataset news.

**Inconvénients** : "amnésie" si l'utilisateur a eu une période de lecture atypique récemment. Sensible au bruit.

### Stratégie 3 : Category (Catégorie)

**Principe** : identifie les top N catégories préférées de l'utilisateur et applique un boost (×2) aux articles qui en font partie.

```python
top_cats = set(pd.Series(cats).value_counts().head(top_n_categories).index)
weights = np.array([2.0 if category in top_cats else 1.0 for aid in aids])
weights /= weights.sum()
user_profile = np.dot(weights, user_embeddings)

```

**Avantages** : profil plus net thématiquement, garde-fou contre la dispersion sémantique.

**Inconvénients** : risque de bulle de filtre (enfermement thématique). Lecteurs éclectiques mal servis. Dépend de la qualité des métadonnées.

### CB + Popularité (modèle final retenu)

**Découverte clé du projet** : combiner le score CB avec un score de popularité récente améliore drastiquement les résultats.

```python
score_final = (1 - beta) * cosine_sim + beta * popularity_score

```

Avec `beta=0.8` (80% popularité, 20% sémantique), le Hit@5 passe de 2.59% à **48.18%**.

Le score de popularité est calculé comme un ratio clics/âge de l'article (en mois), normalisé entre 0 et 1. Il favorise les articles récents très cliqués.

### Optimisation technique : PCA

Pour réduire les artefacts de 424 MB à 54 MB sans perte significative de performance, on applique une PCA sur les embeddings :

```python
pca = PCA(n_components=33, random_state=42)  # 85% de variance conservée
candidate_embeddings_reduced = pca.fit_transform(candidate_embeddings)
# Profils : transform (pas fit_transform) avec le MÊME pca
user_profiles_reduced = pca.transform(user_profiles_matrix)

```

**Règle critique** : `fit_transform` sur les embeddings candidats, `transform` uniquement sur les profils utilisateurs. Les deux doivent vivre dans le même espace réduit pour que la cosine similarity soit valide.

Résultat : 33 composantes, 85% de variance, perte de performance de seulement 1.5% sur le Hit@5.

---

## 3. Collaborative Filtering (CF)

### Principe général

Le CF ne regarde pas le contenu des articles. Il s'appuie uniquement sur les comportements collectifs : si beaucoup d'utilisateurs similaires ont lu un article, il est probablement pertinent.

Les données sont représentées dans une **matrice User × Items** où chaque cellule contient le nombre de clics de l'utilisateur sur l'article.

### Le problème de la matrice dense

```
64 734 users × 364 047 articles × 4 bytes = 91 GB RAM

```

Solution : matrice **sparse (CSR)** — on ne stocke que les cellules non-nulles (les clics effectifs). Avec ~14 clics/utilisateur sur 364k articles, la sparsité est de ~99.99%.

```python
from scipy.sparse import csr_matrix
user_item_matrix = csr_matrix(
    (clicks, (user_idxs, item_idxs)),
    shape=(n_users, n_items)
)

```

**Point important** : les IDs doivent être des entiers continus (0, 1, 2...) — `pd.Categorical` permet de les réindexer.

### Stratégie 1 : Item-Item Cosine

**Principe** : calcule la similarité cosine entre articles basée sur leurs co-clics. Recommande les articles similaires à ceux déjà lus.

```
score(user, article_j) = Σ clics(user, article_i) × cos(article_i, article_j)

```

C'est un modèle **mémoriel** — pas de paramètres appris, juste des similarités calculées. Simple à expliquer et auditer.

**Limite** : un article jamais cliqué ne peut pas être recommandé (cold start total).

### Stratégie 2 : ALS (Alternating Least Squares)

**Principe** : factorise la matrice User × Items en deux matrices de facteurs latents U (users) et V (items). Le score est un produit scalaire :

```
score(user, article) = U[user] · V[article]

```

L'algorithme alterne entre deux étapes :

1. Figer V, optimiser U (chaque utilisateur ajuste son profil pour minimiser l'erreur)
2. Figer U, optimiser V (chaque article ajuste son profil)

Le paramètre `alpha=40` amplifie le signal de confiance (formulation Hu & Koren) : plus un utilisateur clique sur un article, plus le signal est fort.

**Avantage** : capture des "relations cachées" entre articles que le contenu seul ne voit pas.

**Inconvénients** : boîte noire, sensible aux hyperparamètres, cold start sur les nouveaux articles.

### Stratégie 3 : BPR (Bayesian Personalized Ranking)

**Principe** : optimise directement l'**ordre** de recommandation plutôt que de reconstruire les valeurs de clics. Il apprend que l'utilisateur préfère l'article i à l'article j.

L'apprentissage se fait par triplets `(user, article_positif, article_négatif)`. Pour chaque triplet, le modèle ajuste les facteurs latents pour que `score(user, positif) > score(user, négatif)`.

Utilise une matrice **binaire** (0/1) — le nombre exact de clics n'a pas de sens dans ce cadre.

**Avantage** : moins de biais de popularité que ALS, fonctionne bien sur des données très sparses.

**Inconvénient** : entraînement par sampling stochastique — peut être instable.

### Stratégie 4 : LMF (Logistic Matrix Factorization)

**Principe** : même structure de facteurs latents qu'ALS, mais le score est transformé par une sigmoïde pour être interprété comme une **probabilité de clic** :

```
score(user, article) = σ(U[user] · V[article]) = 1 / (1 + exp(-U · V))

```

**Avantage** : scores interprétables comme probabilités (0 à 1).

**Inconvénient** : les non-clics sont traités comme des désintérêts, ce qui est une approximation — un article non cliqué peut juste ne pas avoir été vu.

### Note technique : implicit et la transposée

La librairie `implicit` est conçue pour les matrices Items × Users :

- `model.fit()` attend **Items × Users** → passer `user_item_matrix.T`
- `model.recommend()` attend **Users × Items** → passer `user_item_matrix` (sans transposée)

---

## 4. Approche Hybride

### Principe : Two-Stage Recommender

L'hybride combine CB et CF en deux étapes séquentielles :

**Stage 1 — CB Retrieval** : sélectionner les N articles les plus proches sémantiquement du profil utilisateur (ex: Top 100). Le CB agit comme un filtre de pertinence thématique.

**Stage 2 — CF Re-ranking** : parmi ces 100 candidats, réordonner selon les scores ALS (comportement collectif).

### Score hybride pondéré

```python
# Normaliser les rangs CB (position 0 = score 1.0)
cb_rank_score = {aid: (n - i) / n for i, aid in enumerate(candidates)}

# Normaliser les scores CF entre 0 et 1
cf_norm = {aid: (score - cf_min) / cf_range for aid, score in cf_scores.items()}

# Score final
score = alpha * cf_norm.get(aid, 0.0) + (1 - alpha) * cb_rank_score[aid]

```

### Pourquoi le CB doit être en premier

Si on commence par le CF, le pool de candidats contient uniquement des articles du train (articles connus). Les 77% d'articles nouveaux du test sont exclus d'office. Le CB en premier garantit que les nouveaux articles restent dans le pool.

---

## 5. Ce que les résultats m'ont appris

### Tableau comparatif final


| Modèle                      | Hit@5      | Soft@5     | Observation           |
| --------------------------- | ---------- | ---------- | --------------------- |
| CB Mean                     | 2.47%      | 50.93%     | Baseline CB           |
| CB Recency                  | 2.58%      | 50.96%     | Marginal              |
| CB Category                 | 2.59%      | 51.04%     | Meilleur CB pur       |
| Item-Item                   | 1.01%      | 13.56%     | Cold start total      |
| ALS                         | 1.05%      | 15.09%     | Limité par cold start |
| BPR                         | 0.84%      | 13.55%     | Idem                  |
| LMF                         | 0.65%      | 8.17%      | Idem                  |
| Hybride CB+ALS              | 0.44%      | 49.78%     | Dégradation           |
| **CB + Popularité (β=0.8)** | **48.18%** | **74.09%** | ✅ Modèle retenu       |


### Pourquoi le CB bat le CF sur ce dataset

Le split temporel crée une **rupture nette** entre train et test : seulement 23% des articles lus dans le test apparaissent dans le train. Le CF ne peut recommander que des articles connus — il est structurellement aveugle à 77% des bonnes réponses.

Ce n'est pas un échec du CF en tant qu'algorithme. Sur un dataset films ou musique (contenu stable), le CF serait probablement meilleur. Sur un dataset news à forte rotation, c'est une inadéquation fondamentale.

### Pourquoi les 3 stratégies CB donnent des résultats similaires

Avec une moyenne de 14 clics par utilisateur, la pondération (recency, category) apporte peu de différence par rapport à la moyenne simple. Il n'y a pas assez de signal pour que la pondération fasse une vraie différence — les résultats convergent vers le même profil moyen.

### Pourquoi l'hybride CB→CF dégrade les résultats

Le CF re-rank en mettant en tête les articles qu'il connaît (vieux articles du train), et en pénalisant les articles nouveaux (`score = -inf` si absent du CF). Résultat : les bons articles (nouveaux) tombent en bas de liste.

La correction par score pondéré (alpha=0) confirme : sans contribution CF, l'hybride retrouve exactement les scores CB.

### Pourquoi CB + popularité fonctionne si bien

La popularité récente est un proxy puissant de l'intention de lecture sur un site news. Un article très cliqué dans les dernières heures est probablement pertinent pour la plupart des utilisateurs — indépendamment de leur profil sémantique.

Avec `beta=0.8`, le modèle dit : *"recommande principalement ce qui est populaire maintenant, avec une légère coloration sémantique selon le profil de l'utilisateur"*. C'est exactement le comportement d'un éditorial de news bien fait.

### Conclusion sur le choix du modèle

Sur un dataset news, la **fraîcheur et la popularité priment sur la personnalisation sémantique**. La personnalisation reste utile en complément (le beta optimal n'est pas 1.0 mais 0.8), mais elle ne peut pas compenser l'absence de signal comportemental sur les nouveaux articles.

---

## 6. Le Serverless

### L'analogie

Imagine la différence entre avoir un cuisinier à demeure 24h/24 (serveur dédié) et appeler un traiteur uniquement quand tu as des invités (serverless). Avec le traiteur, tu ne paies que quand il cuisine.

### Ce que "serverless" veut vraiment dire

Le terme est un abus de langage marketing — il y a bien des serveurs physiques derrière. Ce que "serverless" signifie réellement : **tu n'as pas à gérer de serveur**. Azure s'occupe de l'infrastructure, tu déploies juste ton code.

En coulisses, Azure dispose de data centers avec des milliers de serveurs physiques. Quand ta Function est appelée, Azure prend un serveur disponible dans son pool, exécute ton code, puis le libère pour un autre client. Tu partages une infrastructure mutualisée et ne paies que ta consommation réelle.

### Cold Start vs Warm Start

**Cold start** : l'instance s'est endormie. Le premier appel réveille le code et charge les artefacts — 1 à 3 secondes de latence. C'est pour ça que le cache global `_recommender` est important : une fois chargé, les appels suivants sont quasi-instantanés.

**Warm start** : l'instance est encore active (appels réguliers). Réponse en quelques ms.

### Pourquoi c'est le bon choix pour ce projet


|             | Serveur classique (VM)   | Serverless (Azure Functions) |
| ----------- | ------------------------ | ---------------------------- |
| Coût        | 24h/24 même sans traffic | Seulement à l'usage          |
| Scalabilité | Manuelle                 | Automatique                  |
| Maintenance | À ta charge              | Gérée par Azure              |
| Cold start  | Aucun                    | 1-3 sec parfois              |
| Gratuit     | Non                      | 1M appels/mois               |


Pour un MVP avec un traffic faible et imprévisible, le serverless est optimal.

### Pourquoi Azure offre ce service gratuitement au départ

C'est un modèle économique intelligent :

- **Mutualisation** : Azure empile des centaines de fonctions sur le même serveur physique. Un serveur à 90% de charge au lieu de 10% = 9× plus rentable.
- **Lock-in** : une fois que ton code tourne sur Azure Functions, tu utilises aussi Blob Storage, Application Insights, Data Factory... Chaque service en attire un autre. Plus tu utilises l'écosystème, plus il est coûteux de partir chez un concurrent (AWS, GCP).
- **Acquisition** : les étudiants et startups démarrent gratuitement, grandissent, et deviennent des clients payants. C'est un pari sur ton succès.

### Fichiers clés d'une Azure Function Python

`function_app.py` : le point d'entrée. Définit les routes HTTP et la logique de l'API.

`host.json` : indique à Azure comment piloter le moteur de fonctions (timeout, version du runtime, logging).

`requirements.txt` : dépendances Python installées par Azure lors du déploiement.

`local.settings.json` : variables d'environnement **uniquement en local**. Jamais commité sur GitHub. L'équivalent en production est le panneau "Environment variables" dans le portail Azure.

### Architecture déployée

```
App Streamlit (local/cloud)
        ↓ HTTP GET /recommend/{user_id}
Azure Function (serverless)
        ↓ Cold start : charge les artefacts une fois
Azure Blob Storage
        - candidate_embeddings.pkl  (45.8 MB, PCA 33 composantes)
        - user_profiles.pkl         (10.0 MB)
        - popularity_scores.pkl     (1.0 MB)
        - pca_final.pkl             (0.0 MB)
        - [+ 4 autres artefacts]
        ↑ Warm start : cache global _recommender en mémoire
Azure Function
        ↓ JSON {"user_id": x, "recommendations": [...]}
App Streamlit

```

---

*Document généré à l'issue du Projet 10 — Parcours AI Engineer*