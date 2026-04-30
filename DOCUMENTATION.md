# Documentation — Projet Airbnb : Prédiction du prix de location

**Cours :** Analyse de Données — ESILV A3 S2  
**Objectif :** Prédire le logarithme du prix (`log_price`) d'un logement Airbnb  
**Métrique :** R² (coefficient de détermination)  
**Rendu :** 15 mai 2026 avant 23h59 sur DVL  

---

## Table des matières

1. [Présentation du problème](#1-présentation-du-problème)
2. [Description des données](#2-description-des-données)
3. [Exploration qualitative](#3-exploration-qualitative)
4. [Feature Engineering](#4-feature-engineering)
5. [Modélisation](#5-modélisation)
6. [Optimisation & Prédiction finale](#6-optimisation--prédiction-finale)
7. [Prédiction finale](#7-prédiction-finale)

---

## 1. Présentation du problème

### Contexte

Airbnb est une plateforme de location de logements entre particuliers. Les hôtes fixent eux-mêmes leur prix, ce qui crée une grande variabilité. L'objectif est de construire un modèle capable de **prédire le prix d'un logement** à partir de ses caractéristiques (type, équipements, localisation, etc.).

### Variable cible : `log_price`

On prédit le **logarithme naturel** du prix (et non le prix directement). Ce choix est justifié par deux raisons :

1. **Distribution** : les prix bruts sont très asymétriques (quelques logements très chers tirent la moyenne vers le haut). Le log rend la distribution quasi-normale, ce qui est plus favorable pour les algorithmes de régression.
2. **Erreurs relatives** : une erreur de 10$ sur un logement à 50$/nuit est bien plus grave qu'une erreur de 10$ sur un logement à 500$/nuit. Travailler sur le log revient à minimiser les **erreurs relatives** plutôt qu'absolues.

### Métrique d'évaluation : R²

Le R² (coefficient de détermination) mesure la proportion de variance du `log_price` expliquée par le modèle :
- **R² = 1** → prédiction parfaite
- **R² = 0** → le modèle ne fait pas mieux que prédire la moyenne
- **R² < 0** → le modèle est pire que prédire la moyenne

La baseline fournie (LinearSVR avec 3 features) atteint **R² ≈ 0.318**.

---

## 2. Description des données

### Fichiers

| Fichier | Rôle | Dimensions |
|---|---|---|
| `airbnb_train.csv` | Données d'entraînement (avec `log_price`) | 22 234 lignes × 28 colonnes |
| `airbnb_test.csv` | Données à prédire (sans `log_price`) | 51 877 lignes × 27 colonnes |
| `prediction_example.csv` | Format de soumission attendu | 51 877 lignes × 2 colonnes |

### Description des colonnes

| Colonne | Type brut | Description |
|---|---|---|
| `id` | int | Identifiant unique du logement |
| `log_price` | float | **Cible** — logarithme naturel du prix par nuit |
| `property_type` | texte | Type de propriété (Apartment, House, Villa…) — 31 valeurs uniques |
| `room_type` | texte | Type de chambre (Entire home/apt, Private room, Shared room) |
| `bed_type` | texte | Type de lit (Real Bed, Futon, Pull-out Sofa, Airbed, Couch) |
| `cancellation_policy` | texte | Politique d'annulation (flexible, moderate, strict…) |
| `city` | texte | Ville (NYC, LA, SF, DC, Chicago, Boston) |
| `accommodates` | int | Nombre maximum de personnes pouvant séjourner |
| `bathrooms` | float | Nombre de salles de bain (0.5 = WC uniquement, convention américaine) |
| `bedrooms` | float | Nombre de chambres |
| `beds` | float | Nombre de lits |
| `cleaning_fee` | bool | Présence de frais de ménage (True/False) |
| `instant_bookable` | t/f | Réservation instantanée possible |
| `host_has_profile_pic` | t/f | L'hôte a une photo de profil |
| `host_identity_verified` | t/f | Identité de l'hôte vérifiée par Airbnb |
| `host_response_rate` | texte | Taux de réponse de l'hôte (ex : "90%") |
| `host_since` | date | Date d'inscription de l'hôte |
| `first_review` | date | Date du premier avis |
| `last_review` | date | Date du dernier avis |
| `number_of_reviews` | int | Nombre total d'avis |
| `review_scores_rating` | float | Note moyenne sur 100 |
| `latitude` | float | Coordonnée GPS |
| `longitude` | float | Coordonnée GPS |
| `neighbourhood` | texte | Nom du quartier |
| `zipcode` | texte | Code postal |
| `amenities` | texte | Liste des équipements au format `{A,"B C",D}` |
| `description` | texte | Description longue du logement |
| `name` | texte | Titre de l'annonce |

### Valeurs manquantes

| Colonne | % manquant | Stratégie |
|---|---|---|
| `host_response_rate` | 24.6% | Imputation médiane + flag `_is_null` |
| `review_scores_rating` | 22.4% | Imputation médiane + flag `_is_null` |
| `first_review` / `last_review` | ~21% | Conversion en jours + flag `has_reviews` |
| `neighbourhood` | 9.4% | Remplacement par "Unknown" |
| `host_since` | 0.25% | Imputation médiane après conversion |
| `bathrooms` / `bedrooms` / `beds` | < 0.3% | Imputation médiane |

---

## 3. Exploration qualitative

L'exploration a pour but de **comprendre les données** et de **motiver les choix de features** pour la partie prédiction. Elle représente environ 1/3 du travail.

### 3.1 Variable cible (`log_price`)

- Distribution quasi-normale (histogramme), ce qui valide le choix du log.
- Prix médian : **~110$/nuit**, prix moyen : **~120$/nuit**.
- Fourchette : de 10$ à ~2 000$/nuit.

### 3.2 Impact des variables catégorielles

**`city`** — très discriminante :

| Ville | Prix médian |
|---|---|
| San Francisco | ~175$ |
| New York | ~120$ |
| Washington DC | ~110$ |
| Los Angeles | ~100$ |
| Boston | ~95$ |
| Chicago | ~85$ |

**`room_type`** — très discriminante :
- *Entire home/apt* → ~150$/nuit
- *Private room* → ~65$/nuit
- *Shared room* → ~45$/nuit

**`property_type`** — discriminante : villas et condominiums sont nettement plus chers que les dortoirs et guesthouses.

**`cancellation_policy`** — modérément discriminante : les politiques strictes sont associées à des logements plus chers (logements premium qui se permettent d'être inflexibles).

**`cleaning_fee`** — modérément discriminant : les logements avec frais de ménage sont en moyenne plus chers (hôtes plus professionnels).

### 3.3 Variables numériques (corrélations avec `log_price`)

| Variable | Corrélation Pearson |
|---|---|
| `accommodates` | ~0.45 |
| `bedrooms` | ~0.42 |
| `bathrooms` | ~0.40 |
| `beds` | ~0.37 |
| `amenities_count` | ~0.35 |
| `number_of_reviews` | ~0.05 |
| `review_scores_rating` | ~0.05 |

Tendance claire : **plus le logement est grand, plus il est cher**. La relation est quasi-logarithmique (passer de 1 à 2 chambres augmente plus le prix que passer de 4 à 5).

### 3.4 Géographie

Les cartes lat/lon colorées par prix montrent que la **localisation géographique** a un fort impact : à NYC, Manhattan concentre les logements les plus chers ; à SF, le centre et les quartiers nord sont les plus prisés. Cela justifie l'inclusion de `latitude`, `longitude` et `neighbourhood`.

### 3.5 Amenities (équipements)

- La colonne est stockée dans un format non-standard : `{TV,"Wireless Internet",Kitchen}`.
- Certaines entrées contiennent `translation missing: en.hosting_amenity_XX` — ce sont des **bugs de la plateforme** (traductions manquantes) sans signification connue. Elles sont exclues du traitement.
- Le **nombre total d'équipements** (`amenities_count`) est corrélé avec le prix (~0.35).
- Les équipements les plus impactants sur le prix : Elevator, Gym, Pool, Cable TV.
- Les équipements quasi-universels (Essentials, Internet) sont peu discriminants.

---

## 4. Feature Engineering

L'exploration a identifié les variables utiles. On les transforme maintenant en features numériques exploitables. Chaque transformation est encapsulée dans une **classe dédiée** avec deux méthodes :
- `fit_transform(df)` : apprend les paramètres **sur le train** et transforme
- `transform(df)` : applique les paramètres appris sur le test

Ce design est crucial : si on recalculait les médianes ou les encodages sur le test, on introduirait un **data leakage** (fuite d'information).

### 4.1 `BasePreprocessor`

Traite les variables déjà simples.

**Booléens `t`/`f` → 0/1**  
Les colonnes `instant_bookable`, `host_has_profile_pic`, `host_identity_verified` contiennent les chaînes `'t'` et `'f'`. On les convertit en 0 et 1.

**`cleaning_fee`** : déjà un booléen Python (`True`/`False`), converti en int.

**`host_response_rate` : `"90%"` → `90.0`**  
La colonne est stockée en texte. On retire le symbole `%` et on convertit en float.

**Encodage catégoriel avec `LabelEncoder`**  
Les colonnes `property_type`, `room_type`, `bed_type`, `cancellation_policy`, `city` sont converties en indices numériques. L'encodeur est sauvegardé pour garantir les mêmes correspondances sur le test. Les valeurs inconnues dans le test reçoivent l'indice `max + 1`.

**Imputation des NaN**  
- `bathrooms`, `bedrooms`, `beds` (<0.3% de NaN) : remplacement par la **médiane du train**.
- `review_scores_rating`, `host_response_rate` (~22-25% de NaN) : remplacement par la médiane **+ création d'un flag binaire `_is_null`**. Ce flag permet au modèle de distinguer "note de 94" de "pas de note du tout" — ces deux situations n'ont pas la même signification.

### 4.2 `AmenitiesEncoder`

Extrait de l'information depuis la colonne `amenities`.

**Parsing**  
Le format `{TV,"Wireless Internet",Kitchen}` ne peut pas être parsé avec un simple `.split(',')` car certains équipements contiennent des virgules dans leur nom. On utilise une **expression régulière** qui gère correctement les guillemets :
```
re.findall(r'"([^"]+)"|([^,]+)', amenity_str)
```
Les entrées `translation missing: ...` sont exclues car elles correspondent à des bugs de la plateforme Airbnb (équipements sans traduction connue).

**`amenities_count`**  
Nombre total d'équipements après nettoyage. Corrélé avec le prix (~0.35) — un logement bien équipé tend à être plus cher.

**Features binaires (top 20)**  
On calcule les 20 amenities les plus fréquentes **sur le train** et on crée une colonne `am_xxx` pour chacune (1 = présent, 0 = absent). On se limite aux 20 plus fréquentes pour éviter des features avec trop peu d'exemples.

Exemples de colonnes générées : `am_wireless_internet`, `am_kitchen`, `am_tv`, `am_air_conditioning`, `am_washer`, `am_dryer`…

### 4.3 `DateFeatures`

Transforme les colonnes dates en valeurs numériques.

**Principe**  
Un algorithme de ML ne peut pas lire une date comme `"2015-03-12"`. On la convertit en **nombre de jours** par rapport à une date de référence fixe (01/12/2017, approximativement la fin du dataset).

**Features créées**

| Feature | Calcul | Interprétation |
|---|---|---|
| `host_experience_days` | ref − `host_since` | Ancienneté de l'hôte — plus il est expérimenté, mieux il optimise son prix |
| `days_since_first_review` | ref − `first_review` | Ancienneté du logement sur la plateforme |
| `days_since_last_review` | ref − `last_review` | Fraîcheur de l'activité — un logement récemment réservé est actif |
| `has_reviews` | `first_review` non-NaN | Flag binaire : le logement a-t-il déjà reçu des avis ? |

Les NaN (logements sans avis ou hôte sans date) sont imputés par la médiane calculée sur le train.

### 4.4 `TextFeatures`

Extrait une information simple depuis les colonnes texte `description` et `name`.

**`has_luxury_keyword`**  
Flag binaire : vaut 1 si le titre ou la description contient un mot-clé premium (`luxury`, `penthouse`, `villa`, `designer`, `panoramic`…). Ces logements tendent à afficher des prix plus élevés.

Les colonnes `description` et `name` brutes sont ensuite supprimées — un algo de ML ne peut pas lire du texte libre directement.

### 4.5 `GeoFeatures`

Encode la colonne `neighbourhood`.

**Regroupement des quartiers rares**  
Les quartiers avec moins de 5 occurrences dans le train sont regroupés sous la catégorie `"Other"`. Raison : un quartier représenté par 2 logements ne permet pas au modèle d'apprendre quoi que ce soit de fiable. Ce regroupement règle aussi le problème des quartiers qui apparaissent dans le test mais pas dans le train.

Les colonnes `latitude` et `longitude` sont gardées telles quelles — elles portent une information géographique précise que le modèle peut exploiter directement.

### 4.6 Features dérivées

On combine des variables existantes pour créer de nouvelles features potentiellement plus informatives.

| Feature | Calcul | Interprétation |
|---|---|---|
| `beds_per_person` | `beds` / `accommodates` | Confort relatif : 1 lit par personne vs 0.5 lit par personne |
| `bathrooms_per_person` | `bathrooms` / `accommodates` | Confort relatif des sanitaires |

La division utilise `.replace(0, 1)` pour éviter les divisions par zéro.

### 4.7 Récapitulatif — Matrix de features finale (X)

Après toutes les transformations, X contient **48 features**, sans aucun NaN.

| Groupe | Nombre de features | Exemples |
|---|---|---|
| Variables de base | 17 | `property_type`, `room_type`, `city`, `accommodates`, `bathrooms`… |
| Amenities | 21 | `amenities_count`, `am_wifi`, `am_tv`, `am_kitchen`… |
| Temporelles | 4 | `host_experience_days`, `days_since_last_review`, `has_reviews`… |
| Texte | 1 | `has_luxury_keyword` |
| Géographie | 1 | `neighbourhood` |
| Flags NaN | 2 | `review_scores_rating_is_null`, `host_response_rate_is_null` |
| Dérivées | 2 | `beds_per_person`, `bathrooms_per_person` |
| **Total** | **48** | |

Les colonnes exclues de X : `id` (identifiant sans info prédictive), `log_price` (c'est la cible `y`), `amenities` (déjà traitée), `zipcode` (redondant avec lat/lon et neighbourhood).

---

## 5. Modélisation

### 5.1 Protocole d'évaluation

Toutes les expériences utilisent le même split **75% train / 25% validation** (`random_state=98` pour la reproductibilité). On mesure le R² sur les deux ensembles pour détecter l'overfitting.

### 5.2 Expériences

**Expérience 1 — Baseline**

Reproduction du notebook fourni : LinearSVR avec seulement 3 features.

| Modèle | Features | R² Train | R² Val |
|---|---|---|---|
| LinearSVR | 3 (property_type, accommodates, bathrooms) | ~0.33 | ~0.32 |

**Expérience 2 — Feature set complet, comparaison de modèles**

On utilise les **48 features** construites en Partie 2 et on teste 5 modèles différents.

| Modèle | R² Train | R² Val | Overfitting |
|---|---|---|---|
| Ridge | ~0.59 | ~0.58 | Faible |
| RandomForest (100) | ~0.95 | ~0.67 | Fort |
| GradientBoosting | ~0.72 | ~0.68 | Modéré |
| **HistGradientBoosting** | **~0.79** | **~0.69** | **Faible** |
| KNeighbors (k=10) | ~0.16 | ~-0.03 | N/A |

### 5.3 Choix du modèle final : HistGradientBoostingRegressor

**HistGradientBoosting** est retenu car il offre :
- Le meilleur R² de validation
- Le moins d'overfitting (écart train/val le plus faible parmi les modèles performants)
- Une robustesse aux valeurs manquantes (gestion native)

**Rejeté — Ridge** : modèle linéaire, sous-performe car les relations prix-features sont non-linéaires.  
**Rejeté — KNeighbors** : très mauvais en haute dimension (*curse of dimensionality* sur 48 features).  
**Rejeté — RandomForest** : fort overfitting malgré de bonnes performances de validation.

### 5.4 Features les plus importantes (RandomForest)

1. `room_type` — type du logement (entier vs chambre privée)
2. `accommodates` — capacité d'accueil
3. `bedrooms` — nombre de chambres
4. `latitude` / `longitude` — localisation géographique
5. `bathrooms` — nombre de salles de bain
6. `city` — ville
7. `amenities_count` — nombre d'équipements
8. `neighbourhood` — quartier
9. `host_experience_days` — ancienneté de l'hôte
10. `cancellation_policy` — politique d'annulation

**Gain total du feature engineering :** R² ~0.32 (baseline) → ~0.69 (feature set complet)

---

## 6. Optimisation & Prédiction finale

### 6.1 Optimisation des hyperparamètres

On utilise `RandomizedSearchCV` avec 20 combinaisons aléatoires et une cross-validation à 5 folds sur `HistGradientBoostingRegressor`.

**Hyperparamètres explorés :**

| Paramètre | Plage |
|---|---|
| `max_iter` | 200 – 600 |
| `learning_rate` | 0.03 – 0.20 |
| `max_depth` | 3 – 10 |
| `min_samples_leaf` | 10 – 60 |
| `l2_regularization` | 0 – 1.0 |
| `max_leaf_nodes` | 20 – 60 |

### 6.2 Analyse des résidus

On vérifie que les erreurs sont aléatoires (pas de biais systématique) :
- Résidus centrés sur 0 → pas de sur/sous-estimation globale
- Distribution quasi-normale → comportement attendu pour un bon modèle

### 6.3 Résultats finaux

| Étape | R² Validation |
|---|---|
| Baseline (LinearSVR, 3 features) | ~0.32 |
| HistGradientBoosting (48 features) | ~0.69 |
| **HistGradientBoosting optimisé** | **~0.71+** |

**Gain total : R² ~0.32 → ~0.71**

## 7. Prédiction finale

Le pipeline complet est appliqué sur `airbnb_test.csv` avec `fit=False` (réutilisation des paramètres appris sur le train). Le fichier `MaPredictionFinale.csv` est généré et validé avec `estConforme()`.

**Format de sortie :**

Le fichier suit exactement le format de `prediction_example.csv` : la première colonne contient les IDs (sans nom de colonne) et la deuxième colonne est `logpred`.

```
,logpred
14282777,4.78
17029381,5.12
...
```

51 877 lignes au total, validées par `estConforme('MaPredictionFinale.csv')`.
