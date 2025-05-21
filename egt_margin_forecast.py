import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import io

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prévision EGT Margin",
    page_icon="📈",
    layout="wide"
)

# Titre de l'application
st.title("📈 Analyse et Prévision de l'EGT Margin")

# === Affichage du graphique initial EGT Margin vs CSN ===
try:
    df_diag = pd.read_excel("802290 data ready to use.xlsx")
    df_diag['EGT Margin'] = pd.to_numeric(df_diag['EGT Margin'], errors='coerce')
    df_diag['CSN'] = pd.to_numeric(df_diag['CSN'], errors='coerce')
    df_diag = df_diag.dropna(subset=['EGT Margin', 'CSN'])

    fig_diag, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_diag['CSN'], df_diag['EGT Margin'], color='blue')
    ax.set_title("Évolution du EGT Margin en fonction du CSN")
    ax.set_xlabel("CSN (Cycle Since New)")
    ax.set_ylabel("EGT Margin (°C)")
    ax.grid(True)
    st.pyplot(fig_diag)
except Exception as e:
    st.warning(f"Impossible d'afficher le graphique initial : {str(e)}")

# Fonction pour charger et préparer les données
@st.cache_data
def load_and_prepare_data():
    # Chargement des données
    df = pd.read_excel("802290 data ready to use.xlsx")
    
    # Conversion des colonnes en types appropriés
    df['Flight DateTime'] = pd.to_datetime(df['Flight DateTime'])

    # Correction : forcer la conversion en string avant .str.replace
    for col in ['EGT Margin', 'Vibration of the core']:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['CSN'] = pd.to_numeric(df['CSN'], errors='coerce')
    
    # Suppression des lignes avec des valeurs manquantes
    df = df.dropna()
    
    return df

# Fonction pour créer les variables de lag
def create_lag_features(df, n_lags=50):
    for i in range(1, n_lags + 1):
        df[f'EGT_Margin_lag_{i}'] = df['EGT Margin'].shift(i)
    return df.dropna()

# Fonction pour entraîner le modèle
def train_model(X, y):
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    return model

# Fonction pour évaluer le modèle
def evaluate_model(y_true, y_pred):
    metrics = {
        'R² Score': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics

# Fonction pour générer les prévisions
def generate_forecasts(model, last_lags, last_vibration, last_csn, n_forecasts=200):
    forecasts = []
    current_lags = last_lags.copy()
    
    for i in range(n_forecasts):
        input_data = np.array([current_lags + [last_vibration, last_csn + i]])
        pred = model.predict(input_data)[0]
        forecasts.append(pred)
        current_lags = [pred] + current_lags[:-1]
    
    return forecasts

# Chargement et préparation des données
try:
    df = load_and_prepare_data()
    
    # Création des variables de lag
    df = create_lag_features(df)
    
    # Préparation des features et de la target
    feature_columns = [col for col in df.columns if col.startswith('EGT_Margin_lag_')] + ['Vibration of the core', 'CSN']
    X = df[feature_columns]
    y = df['EGT Margin']
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Entraînement du modèle
    model = train_model(X_train, y_train)
    
    # Évaluation sur les données de test
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    # Génération des prévisions
    last_lags = df[feature_columns].iloc[-1].values[:50].tolist()
    last_vibration = df['Vibration of the core'].iloc[-1]
    last_csn = df['CSN'].iloc[-1]
    last_date = df['Flight DateTime'].iloc[-1]
    
    forecasts = generate_forecasts(model, last_lags, last_vibration, last_csn)
    
    # Création des dates futures
    future_dates = [last_date + timedelta(days=i) for i in range(1, 201)]
    
    # Création du graphique avec Plotly
    fig = go.Figure()
    
    # Ajout des données historiques
    fig.add_trace(go.Scatter(
        x=df['Flight DateTime'],
        y=df['EGT Margin'],
        name='Données historiques',
        line=dict(color='blue')
    ))
    
    # Ajout des prévisions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecasts,
        name='Prévisions',
        line=dict(color='green', dash='dash')
    ))
    
    # Ajout de la zone critique
    fig.add_shape(
        type="rect",
        x0=min(df['Flight DateTime']),
        x1=max(future_dates),
        y0=15,
        y1=18,
        fillcolor="red",
        opacity=0.3,
        layer="below",
        line_width=0,
    )
    
    # Mise à jour du layout
    fig.update_layout(
        title="Prévision EGT Margin sur 200 cycles",
        xaxis_title="Date",
        yaxis_title="EGT Margin (°C)",
        showlegend=True,
        hovermode="x unified"
    )
    
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    # Affichage du graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Affichage des métriques
    st.subheader("📊 Métriques de performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{metrics['R² Score']:.4f}")
    with col2:
        st.metric("MAE", f"{metrics['MAE']:.4f}")
    with col3:
        st.metric("MSE", f"{metrics['MSE']:.4f}")
    with col4:
        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
    
    # Création du DataFrame pour l'export
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'EGT Margin Forecast': forecasts
    })
    
    # Correction : export Excel en mémoire pour le bouton de téléchargement
    buffer = io.BytesIO()
    forecast_df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    
    # Bouton de téléchargement
    st.download_button(
        "📥 Télécharger le fichier Excel des prévisions",
        buffer,
        "EGT_Margin_Forecast_802290_200_Cycles.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"Une erreur est survenue : {str(e)}") 