# AirParadis – Analyse de sentiment

## Objectif

Développer un modèle capable de prédire le sentiment (positif / négatif) d’un tweet afin d’anticiper les bad buzz.

---

## Modèles testés

* Modèle simple (Embedding + GlobalAveragePooling)
* Modèle avancé (BiLSTM) => retenu
* Modèle BERT (testé mais trop lourd)

---

## Suivi des performances

* Utilisation de MLflow pour comparer les modèles (accuracy, F1-score)
* Stockage des modèles dans "api/artifacts"

---

## Déploiement

* API développée avec FastAPI
* Interface utilisateur via Streamlit

---

## Monitoring

* Utilisation de Azure Application Insights pour le suivi des prédictions
* Collecte des feedbacks utilisateurs (satisfait / non satisfait)
* Mise en place d’alertes en cas de dérive du modèle

---

## Remarque importante

Le projet inclut des composants de monitoring (Azure Application Insights) qui nécessitent une configuration spécifique (variables d’environnement).
Ces éléments ne sont pas activés par défaut afin d’éviter toute utilisation involontaire ou surcoût.

---

## Projet OpenClassrooms – Ingénieur IA
