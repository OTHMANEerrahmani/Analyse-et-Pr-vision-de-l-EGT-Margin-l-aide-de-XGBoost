# Analyse et Prévision de l'EGT Margin

Cette application Streamlit permet d'analyser et de prévoir l'EGT Margin d'un moteur d'avion en utilisant XGBoost.

## Fonctionnalités

- Chargement et traitement des données historiques
- Création de variables de lag pour la prédiction
- Entraînement d'un modèle XGBoost
- Prévision sur 200 cycles
- Visualisation interactive avec Plotly
- Affichage des métriques de performance
- Export des prévisions en Excel

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Assurez-vous que le fichier Excel "802290 data ready to use.xlsx" est présent dans le même répertoire que le script
2. Lancez l'application :
```bash
streamlit run egt_margin_forecast.py
```

## Structure des données requises

Le fichier Excel doit contenir les colonnes suivantes :
- Flight DateTime
- EGT Margin
- Vibration of the core
- CSN

## Notes

- Les prévisions sont générées sur 200 cycles à partir des dernières données disponibles
- La zone critique (15°C - 18°C) est mise en évidence en rouge sur le graphique
- Les métriques de performance sont calculées uniquement sur les données historiques # moghit
# moghit
# Analyse-et-Pr-vision-de-l-EGT-Margin-l-aide-de-XGBoost-avec-Visualisation-dans-Streamlit
# Analyse-et-Pr-vision-de-l-EGT-Margin-l-aide-de-XGBoost-avec-Visualisation-dans-Streamlit
# Analyse-et-Pr-vision-de-l-EGT-Margin-l-aide-de-XGBoost-avec-Visualisation-dans-Streamlit
# Analyse-et-Pr-vision-de-l-EGT-Margin-l-aide-de-XGBoost
# Analyse-et-Pr-vision-de-l-EGT-Margin-l-aide-de-XGBoost
