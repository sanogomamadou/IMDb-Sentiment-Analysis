# 🎬 Classification des Sentiments sur les Avis de Films IMDb

## 📌 Présentation du Projet

Ce projet porte sur **l’analyse de sentiment des avis de films IMDb**, avec pour objectif de classifier automatiquement les critiques en positives ou négatives selon leur contenu textuel. En utilisant des techniques classiques de traitement du langage naturel (NLP), l’ingénierie des caractéristiques et des modèles de machine learning, le projet étudie l’impact de différentes représentations textuelles sur la performance de classification.

Il s’agit d’une exploration pratique du prétraitement, de l’extraction de features et de l’optimisation des modèles sur un jeu de données réel.

---

## 📓 Contenu du Notebook

### 1. 🧹 Prétraitement des Données

* Nettoyage des données : suppression des ponctuations, caractères spéciaux et mise en minuscules.
* Application de la lemmatisation pour réduire les mots à leur forme canonique.
* Élimination des stopwords afin de ne conserver que les mots porteurs de sens.

### 2. 📊 Analyse Exploratoire des Données (EDA) avec Plotly

Visualisations interactives permettant de :

* Explorer la **distribution des avis positifs et négatifs**.
* Identifier les **mots les plus fréquents** dans chaque catégorie via des word clouds et des graphiques de fréquence.
* Étudier la **distribution de la longueur des commentaires** selon le sentiment exprimé.

### 3. 🧠 Ingénierie des Caractéristiques

#### a. TF-IDF avec N-grams

* Extraction des caractéristiques textuelles à l’aide du vectoriseur TF-IDF avec des n-grams allant des unigrammes aux trigrammes.
* Optimisation des hyperparamètres (`ngram_range`, `max_features`) via GridSearchCV.

#### b. Embeddings Word2Vec

* Entraînement de modèles Word2Vec pour capturer la sémantique des mots.
* Calcul de la moyenne des vecteurs de mots pour obtenir une représentation fixe par avis.
* Recherche des meilleurs paramètres (taille des vecteurs, fenêtre, min\_count) via GridSearch.

#### c. Combinaison des Features

* Expérimentation de la fusion des caractéristiques TF-IDF et Word2Vec afin d’évaluer l’amélioration potentielle des performances par la combinaison d’informations syntaxiques et sémantiques.

---

## 🤖 Modèles et Résultats

Les modèles suivants ont été entraînés et optimisés avec GridSearchCV :

* **Linear Support Vector Classifier (LinearSVC)**
* **Random Forest Classifier**
* **Régression Logistique**

### ✅ Meilleurs résultats obtenus :

* **Modèle** : LinearSVC avec features TF-IDF n-grams
* **Exactitude (Accuracy)** : 89%
* **F1-Score (Macro)** : 0.90

### 🧠 Interprétation :

Les features TF-IDF basées sur les n-grams associées à LinearSVC surpassent Word2Vec et la combinaison des deux. Cela suggère que :

* Les motifs explicites de phrases (capturés par les n-grams) constituent des indicateurs forts de sentiment.
* La capture sémantique offerte par Word2Vec est moins efficace dans ce contexte, probablement en raison de la taille du dataset ou de la nature des indices de sentiment.
* Les caractéristiques simples et creuses combinées à des modèles linéaires offrent souvent de meilleures performances pour la classification de texte sur des jeux de données de taille moyenne.

---

## 🛠️ Application Personnalisée de Formation et Test

Une application a été développée pour :

* Permettre l’**entraînement d’un modèle sur des données annotées personnalisées**.
* Autoriser le **test en temps réel de commentaires personnalisés** pour prédire leur sentiment.

Cet outil rend le projet accessible, pratique et réutilisable sur tout type de dataset ou cas d’usage.

---

## 📂 Technologies Utilisées

* Python
* Pandas, NumPy
* Plotly, Seaborn, Matplotlib
* Scikit-learn
* Gensim (Word2Vec)
* Streamlit / Gradio (pour l’application interactive)

---

## 🤝 Ouverture à la Collaboration et Opportunités

Ouvert à :

* Collaborations sur des projets en IA, NLP et Data Science
* Contributions open-source
* Opportunités professionnelles en Data Science / Machine Learning

N’hésitez pas à prendre contact !

---

## 📧 Contact

* LinkedIn : [LinkedIn](https://linkedin.com/in/mamadou-sanogo-3b22a9263)

---

Tu veux que je te prépare ça en markdown prêt à coller aussi ?
