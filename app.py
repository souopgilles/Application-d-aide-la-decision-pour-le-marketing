import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt
from utils import load_data, filter_data, calculate_rfm_segments,calculate_clv_metrics, predict_clv, get_empirical_clv , get_cohort_matrix, kpis_block, scenario_block, calculate_rfm, segment_rfm

# --- CONFIGURATION ---
st.set_page_config(page_title="CLV Dashboard", layout="wide")

# --- DATA LOADING ---
DATA_PATH = r"C:\Users\rolan\projet_data_viz_main\data\processed\online_retail_clean.csv"


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


df_raw = get_data()
#df_raw = preprocess_data(df_raw)


# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.title("Analyse CLV")
    st.write("Filtres Globaux")
    # Liste déroulante de tous les pays triés par ordre alphabétique
    countries = st.multiselect("Pays", sorted(df_raw['Country'].unique()), default=["United Kingdom"])
    if not countries:
        countries = sorted(df_raw['Country'].unique()) # Fallback si rien sélectionné



    min_d, max_d = df_raw['InvoiceDate'].min().date(), df_raw['InvoiceDate'].max().date()
    dates = st.date_input("Période", [min_d, max_d]) # affiche le calendrier en respecant min et max

    # Paramètres spécifiés dans le périmètre fonctionnel
    min_order = st.number_input("Seuil Commande (£)", 0, 2000, 0) # controler le montant min des factures

    # Application
    df = filter_data(df_raw, countries, dates, min_order)
    st.markdown("---")
    st.caption(f"Données filtrées: {len(df)} transactions") # Affichage du nombre de données après filtrage

    # Application des filtres
    mask =  (df_raw['InvoiceDate'].dt.date >= dates[0]) & \
            (df_raw['InvoiceDate'].dt.date <= dates[1]) & \
            (df_raw['Country'].isin(countries))
           
    df_filtered = df_raw[mask].copy()

# --- CALCULS INITIAUX ---
if df.empty:
    st.error("Aucune donnée avec ces filtres.")
    st.stop()


#
ref_date = df_filtered['InvoiceDate'].max() + dt.timedelta(days=1)
rfm_df = calculate_rfm(df_filtered, ref_date) 
#Attribution segments
rfm_df['Segment'] = rfm_df.apply(segment_rfm, axis=1)


    

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
tabC, tab1, tab2, tab4, tab5 = st.tabs(["1. Rétention par cohorte d’acquisition","1. Vue Globale (KPIs)", "2. CLV par Segment", "3. Simulateur & Scénarios","Exports"])
# TAB 2 : COHORTES
with tabC:
        st.subheader("Analyse de la Rétention par Cohorte d'Acquisition")
        st.markdown("""
        Ce graphique montre le pourcentage de clients d'une cohorte (mois de première commande) 
        qui reviennent acheter lors des mois suivants (M+1, M+2, etc.).
        """)
        
        # Préparation données cohortes
        # Attention: Cohortes doivent être calculées sur le dataset global pour avoir la vraie date de 1er achat, 
        # puis filtrées pour l'affichage si besoin, mais ici on recalcule sur la sélection pour voir la rétention "dans la fenêtre"
        # Pour une vraie analyse cohorte, il vaut mieux utiliser df_clean complet pour définir la cohorte
        
        retention_matrix, cohort_sizes = get_cohort_matrix(df_filtered, metric='retention')
        
        # Heatmap Rétention
        fig_cohort = px.imshow(
            retention_matrix,
            labels=dict(x="Mois après 1er achat", y="Mois de Cohorte", color="Rétention"),
            x=retention_matrix.columns,
            y=retention_matrix.index.astype(str),
            color_continuous_scale="RdYlGn",
            text_auto='.1%',
            aspect="auto"
        )
        fig_cohort.update_layout(title="Taux de Rétention par Cohorte")
        st.plotly_chart(fig_cohort, use_container_width=True)
        
        st.info(f"💡 **Lecture :** La cohorte du {retention_matrix.index[0].strftime('%Y-%m')} comportait {cohort_sizes.iloc[0]} nouveaux clients.")

        # Vue Revenu par cohorte
        st.subheader("Densité de Valeur par Cohorte")
        revenue_matrix, _ = get_cohort_matrix(df_filtered, metric='monetary')
        
        fig_rev_cohort = px.imshow(
            revenue_matrix,
            labels=dict(x="Mois après 1er achat", y="Mois de Cohorte", color="CA Généré"),
            x=revenue_matrix.columns,
            y=revenue_matrix.index.astype(str),
            color_continuous_scale="Blues",
            aspect="auto"
        )
        fig_rev_cohort.update_layout(title="Chiffre d'Affaires par Cohorte")
        st.plotly_chart(fig_rev_cohort, use_container_width=True)


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
# TAB 2 : CLV PAR SEGMENT (VOTRE COEUR DE SUJET) calculate_rfm_segments
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

    
    # Visualisation Distribution Segments
    col_seg1, col_seg2 = st.columns([1, 2])
        
    with col_seg1:
            seg_counts = rfm_df['Segment'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig_pie = px.pie(seg_counts, values='Count', names='Segment', title="Répartition des Clients", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("**Statistiques Rapides**")
            st.dataframe(rfm_df.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).round(1), use_container_width=True)

    with col_seg2:
            st.markdown("### Cartographie RF")
            st.markdown("Visualisation des clients selon la Fréquence et la Récence. La taille des bulles représente le Montant.")
            fig_scatter = px.scatter(
                rfm_df, 
                x='Recency', 
                y='Frequency', 
                color='Segment',
                size='Monetary',
                hover_data=['Customer ID'],
                log_y=True, # Log scale souvent nécessaire pour Frequency
                title="Matrice Récence vs Fréquence"
            )
            # Inverser l'axe X pour Récence (les plus récents à droite ou gauche selon convention, ici à gauche = petit chiffre)
            fig_scatter.update_xaxes(autorange="reversed") 
            st.plotly_chart(fig_scatter, use_container_width=True)


with tab4:
    ca_total = df_filtered["TotalPrice"].sum()
    n_clients = df_filtered["Customer ID"].nunique()
    ca_mean = ca_total / n_clients

    orders = df_filtered.groupby("Customer ID")["Invoice"].nunique()
    r_base = (orders >= 2).mean()

    marge_base = 0.30
    remise_base = 0.0
    d_base = 0.10

    st.subheader("CLV   CA   Rétention")

    kpis_block(ca_mean, r_base, marge_base, remise_base, d_base)

    st.subheader("Scénario global : Marge + Remise + Rétention + d")

    col1, col2 = st.columns(2)
    with col1:
        val_m = st.slider("Marge  (%)", 0.0, 100.0, marge_base * 100, 1.0)
        val_rem = st.slider("Remise  (%)", 0.0, 50.0, remise_base * 100, 1.0)

        val_r = st.slider("Rétention  (0–1)", 0.0, 0.95, min(0.9, r_base + 0.05), 0.01)
 
        d_scen = st.slider("Taux d’actualisation d ", 0.0, 0.5, d_base, 0.01)
        remise = st.checkbox("Appliquer remise globale 10%")

    marge_scen = val_m / 100
    remise_scen = val_rem / 100
    r_scen = val_r
    with col2:
        scenario_block(
        "Scénario global : comparaison baseline vs scénario",
        ca_mean, r_base, marge_base, remise_base,
        d_base, d_scen, remise_scen, marge_scen, r_scen
        )
    impact_panier = 0.9 if remise else 1.0
     # 1. Calcul nouvelle CLV Globale
    clv_global_sim = predict_clv(
            global_basket * impact_panier,
            global_freq,
            marge_scen,
            r_scen,
            d_scen
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


    # 2. Impact sur les Segments
    seg_metrics['CLV_Simulee'] = seg_metrics.apply(
            lambda row: predict_clv(
                row['Panier_Moyen'] * impact_panier,
                row['Frequence_Achat'],
                marge_scen,
                r_scen,
                d_scen
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
    st.plotly_chart(fig_sim, use_container_width=True, key="tab4_simulator")

    with tab5:
        st.subheader("Plan d'Action & Exports")
        
        st.markdown("Téléchargez les listes de clients pour vos campagnes CRM (Emailing, Facebook Ads, etc.).")
        
        # Filtre sur le segment à exporter
        target_segment = st.selectbox("Sélectionner le segment à exporter", ["Tous"] + list(rfm_df['Segment'].unique()))
        
        if target_segment == "Tous":
            export_df = rfm_df
        else:
            export_df = rfm_df[rfm_df['Segment'] == target_segment]
            
        st.write(f"Prévisualisation ({len(export_df)} clients) :")
        st.dataframe(export_df.head())
        
        # Bouton CSV
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=f" Télécharger la liste '{target_segment}' (CSV)",
            data=csv,
            file_name=f'export_marketing_{target_segment}_{dt.date.today()}.csv',
            mime='text/csv',
        )
        