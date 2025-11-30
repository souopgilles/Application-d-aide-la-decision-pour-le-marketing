import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt
from utils import load_data, filter_data, calculate_rfm_segments, calculate_clv_metrics, predict_clv, get_empirical_clv

# --- CONFIGURATION ---
st.set_page_config(page_title="CLV Dashboard", layout="wide")

# --- DATA LOADING ---
DATA_PATH = r"C:\Users\rolan\projet_data_viz_main\data\processed\online_retail_clean.csv"


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


df_raw = get_data()

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.title("Analyse CLV")
    st.write("Filtres Globaux")
    # Liste déroulante de tous les pays triés par ordre alphabétique
    countries = st.multiselect("Pays", sorted(df_raw['Country'].unique()), default=["United Kingdom"])

    min_d, max_d = df_raw['InvoiceDate'].min().date(), df_raw['InvoiceDate'].max().date()
    dates = st.date_input("Période", [min_d, max_d]) # affiche le calendrier en respecant min et max

    # Paramètres spécifiés dans le périmètre fonctionnel
    min_order = st.number_input("Seuil Commande (£)", 0, 500, 0) # controler le montant min des factures

    # Application
    df = filter_data(df_raw, countries, dates, min_order)
    st.markdown("---")
    st.caption(f"Données filtrées: {len(df)} transactions") # Affichage du nombre de données après filtrage

# --- CALCULS INITIAUX ---
if df.empty:
    st.error("Aucune donnée avec ces filtres.")
    st.stop()

# 1. Segmentation RFM (Base de l'analyse)
with st.spinner("Calcul des segments..."):
    rfm = calculate_rfm_segments(df)
# On attribue une étiquette à chaque client


# 2. Métriques CLV (Panier, Fréquence par segment) - On récupère toutes les infos nécessaires pour le CLV
global_basket, global_freq, seg_metrics = calculate_clv_metrics(df, rfm)

# 3. CLV Empirique (Courbe historique) - On cécupère toutes les infos nécessaires pour le CLV au global
clv_empirique_curve = get_empirical_clv(df)
clv_historical_total = clv_empirique_curve['CLV_Empirique'].max()

# --- MAIN LAYOUT ---

st.title("Analyse de la Valeur Vie Client (CLV)")

# TABS pour organiser votre partie
tab1, tab2, tab3 = st.tabs(["1. Vue Globale (KPIs)", "2. CLV par Segment", "3. Simulateur & Scénarios"])

# ---------------------------------------------------------------------
# TAB 1 : CLV GLOBALE
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Performance Globale")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "CLV Historique (Empirique)",
        f"£{clv_historical_total:.2f}",
        help="Revenu cumulé moyen généré par un client historique sur toute la période."
    )

    col2.metric("Panier Moyen Global", f"£{global_basket:.2f}")
    col3.metric("Fréquence d'Achat Moyenne", f"{global_freq:.2f}")

    st.divider()

    st.subheader("Courbe de valeur cumulée (Méthode Empirique)")
    st.markdown("Comment la valeur d'un client moyen augmente-t-elle mois après mois ?")

    fig_emp = px.line(
        clv_empirique_curve,
        x='CohortIndex',
        y='CLV_Empirique',
        markers=True,
        labels={'CohortIndex': 'Mois après acquisition', 'CLV_Empirique': 'Revenu Cumulé Moyen (£)'},
        title="CLV Empirique : Croissance de la valeur client dans le temps"
    )
    st.plotly_chart(fig_emp, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2 : CLV PAR SEGMENT (VOTRE COEUR DE SUJET)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Valeur par Segment Client")
    st.markdown("Comparaison de la valeur générée par les différents profils RFM.")

    # Paramètres de base pour l'estimation théorique
    # On utilise des valeurs par défaut raisonnables pour afficher une estimation
    marge_defaut = 0.25
    retention_defaut = 0.72
    discount_defaut = 0.1 # taux d'actualisation

    # Calcul de la CLV Théorique pour chaque segment
    seg_metrics['CLV_Estimee'] = seg_metrics.apply(
        lambda row: predict_clv(
            row['Panier_Moyen'],
            row['Frequence_Achat'],
            marge_defaut,
            retention_defaut,
            discount_defaut
        ), axis=1
    )

    # Visualisation Bar Chart
    fig_seg = px.bar(
        seg_metrics,
        x='Segment',
        y='CLV_Estimee',
        color='Segment',
        text_auto='.1f',
        title=f"CLV Estimée par Segment (Marge {marge_defaut * 100}%)"
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    # Tableau détaillé
    st.write("#### Détails des métriques par segment")

    # Formatage pour l'affichage
    display_cols = ['Segment', 'Nb_Clients', 'Panier_Moyen', 'Frequence_Achat', 'CLV_Estimee']
    st.dataframe(
        seg_metrics[display_cols].style.format({
            'Panier_Moyen': '£{:.2f}',
            'Frequence_Achat': '{:.2f}',
            'CLV_Estimee': '£{:.2f}'
        }).background_gradient(subset=['CLV_Estimee'], cmap='Greens'),
        use_container_width=True
    )

# ---------------------------------------------------------------------
# TAB 3 : SIMULATEUR / SCÉNARIOS
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Simulateur d'Impact Business")
    st.markdown("Ajustez les paramètres pour voir l'impact sur la CLV Globale et par Segment.")

    # Zone de paramètres (Sliders comme demandé dans le périmètre)
    col_params, col_viz = st.columns([1, 2])

    with col_params:
        st.markdown("### Hypothèses")
        marge_sim = st.slider("Marge (%)", 10, 80, 25, 1) / 100
        retention_sim = st.slider("Taux de Rétention (r)", 0.1, 0.95, 0.6, 0.05,
                                  help="Probabilité qu'un client rachète l'année suivante.")
        discount_sim = st.slider("Taux d'Actualisation (d)", 0.05, 0.20, 0.1, 0.01,
                                 help="Coût du capital / dépréciation de l'argent.")

        remise = st.checkbox("Appliquer remise globale 10%")
        impact_panier = 0.9 if remise else 1.0

    with col_viz:
        # 1. Calcul nouvelle CLV Globale
        clv_global_sim = predict_clv(
            global_basket * impact_panier,
            global_freq,
            marge_sim,
            retention_sim,
            discount_sim
        )

        # Delta par rapport à la config par défaut du Tab 2
        clv_global_ref = predict_clv(global_basket, global_freq, 0.25, 0.6, 0.1)
        delta_global = clv_global_sim - clv_global_ref

        st.metric(
            "CLV Globale Projetée",
            f"£{clv_global_sim:.2f}",
            delta=f"{delta_global:.2f} £ vs Baseline",
            help="Formule : (Panier x Freq x Marge) * (r / (1+d-r))"
        )

        st.divider()

        # 2. Impact sur les Segments
        seg_metrics['CLV_Simulee'] = seg_metrics.apply(
            lambda row: predict_clv(
                row['Panier_Moyen'] * impact_panier,
                row['Frequence_Achat'],
                marge_sim,
                retention_sim,
                discount_sim
            ), axis=1
        )

        st.markdown("#### Impact par Segment")

        # Comparaison Avant/Après sur un graph groupé
        comp_data = pd.melt(
            seg_metrics,
            id_vars=['Segment'],
            value_vars=['CLV_Estimee', 'CLV_Simulee'],
            var_name='Scenario', value_name='CLV'
        )

        # Renommer pour la légende
        comp_data['Scenario'] = comp_data['Scenario'].map(
            {'CLV_Estimee': 'Baseline (25% Marge)', 'CLV_Simulee': 'Scénario Actuel'})

        fig_sim = px.bar(
            comp_data,
            x='Segment',
            y='CLV',
            color='Scenario',
            barmode='group',
            title="Comparaison Baseline vs Scénario"
        )
        st.plotly_chart(fig_sim, use_container_width=True)