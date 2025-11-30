import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from io import BytesIO

# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="App Décision Marketing - Retail",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés pour l'accessibilité et le rendu visuel
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .highlight {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# FONCTIONS DE CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES

@st.cache_data
def generate_dummy_data():
    """Génère des données de test si le fichier source n'est pas présent."""
    dates = pd.date_range(start='2009-12-01', end='2011-12-09', freq='H')
    n_samples = 5000  # Échantillon réduit pour la démo
    
    data = {
        'InvoiceNo': [f"{np.random.randint(500000, 600000)}" for _ in range(n_samples)],
        'StockCode': [f"ITEM_{np.random.randint(100, 200)}" for _ in range(n_samples)],
        'Description': ['Produit Test' for _ in range(n_samples)],
        'Quantity': np.random.randint(1, 20, size=n_samples),
        'InvoiceDate': np.random.choice(dates, n_samples),
        'Price': np.random.uniform(1.0, 100.0, size=n_samples).round(2),
        'Customer ID': np.random.randint(10000, 10500, size=n_samples),
        'Country': np.random.choice(['United Kingdom', 'France', 'Germany', 'EIRE'], size=n_samples, p=[0.8, 0.1, 0.05, 0.05])
    }
    df = pd.DataFrame(data)
    
    # Simuler quelques retours (Quantités négatives et InvoiceNo commençant par C)
    mask_returns = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
    df.loc[mask_returns, 'Quantity'] = df.loc[mask_returns, 'Quantity'] * -1
    df.loc[mask_returns, 'InvoiceNo'] = 'C' + df.loc[mask_returns, 'InvoiceNo']
    
    return df

@st.cache_data
def load_data(file_path=None):
    """
    Charge les données. 
    Si file_path est None ou le fichier introuvable, génère des données synthétiques.
    """
    try:
        if file_path and file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
        elif file_path and (file_path.endswith('.xlsx') or file_path.endswith('.xls')):
            df = pd.read_excel(file_path)
        else:
            raise FileNotFoundError("Aucun fichier fourni")
    except Exception as e:
        # En production, on enlèverait ce fallback, mais pour la démo c'est essentiel
        df = generate_dummy_data()
        
    return df

def preprocess_data(df):
    """Nettoyage et Feature Engineering de base."""
    df = df.copy()
    
    # Standardisation des noms de colonnes
    df.columns = [c.strip() for c in df.columns]
    
    # Conversion date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Suppression des Customer ID nuls (essentiel pour RFM)
    df.dropna(subset=['Customer ID'], inplace=True)
    df['Customer ID'] = df['Customer ID'].astype(str).str.split('.').str[0]
    
    # Calcul montant total ligne
    df['TotalAmount'] = df['Quantity'] * df['Price']
    
    # Colonnes temporelles
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
    df['InvoiceDate_Date'] = df['InvoiceDate'].dt.date
    
    return df

# FONCTIONS MÉTIER (Cohortes, RFM)

def calculate_rfm(df, reference_date):
    """Calcule le score RFM."""
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    # Suppression des montants négatifs ou nuls pour le calcul des scores (cas retours purs)
    rfm = rfm[rfm['Monetary'] > 0]
    
    # Scoring (1 à 4, 4 étant le meilleur sauf pour Récence où 1 est le meilleur en logique qcut par défaut, on inverse après)
    # Note : qcut peut échouer s'il y a trop de doublons, on utilise rank(method='first') pour stabiliser
    labels = [1, 2, 3, 4]
    
    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), q=4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=labels)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=4, labels=labels)
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    return rfm

def segment_rfm(row):
    """Définit les segments marketing basés sur les scores R et F."""
    r = int(row['R_Score'])
    f = int(row['F_Score'])
    
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Fidèles'
    elif r >= 3 and f <= 2:
        return 'Prometteurs'
    elif r <= 2 and f >= 3:
        return 'À Risque'
    elif r <= 2 and f <= 2:
        return 'Hibernants'
    else:
        return 'Nécessite Attention'

def get_cohort_matrix(df, metric='retention'):
    """Génère la matrice de cohorte."""
    # 1. Date de première commande par client
    df['CohortMonth'] = df.groupby('Customer ID')['InvoiceMonth'].transform('min')
    
    # 2. Index de cohorte (Mois Facture - Mois Cohorte)
    def get_date_int(df, column):
        year = df[column].dt.year
        month = df[column].dt.month
        return year, month

    invoice_year, invoice_month = get_date_int(df, 'InvoiceMonth')
    cohort_year, cohort_month = get_date_int(df, 'CohortMonth')
    
    years_diff = invoice_year - cohort_year
    months_diff = invoice_month - cohort_month
    
    df['CohortIndex'] = years_diff * 12 + months_diff + 1
    
    # 3. Pivot
    if metric == 'retention':
        grouping = df.groupby(['CohortMonth', 'CohortIndex'])
        cohort_data = grouping['Customer ID'].apply(pd.Series.nunique).reset_index()
        cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='Customer ID')
        cohort_sizes = cohort_counts.iloc[:, 0]
        retention = cohort_counts.divide(cohort_sizes, axis=0)
        return retention, cohort_sizes
    else: # monetary
        grouping = df.groupby(['CohortMonth', 'CohortIndex'])
        cohort_data = grouping['TotalAmount'].sum().reset_index()
        cohort_sums = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='TotalAmount')
        return cohort_sums, None

# INTERFACE PRINCIPALE
def main():
    # --- Sidebar : Chargement et Filtres ---
    st.sidebar.title("🔍 Filtres & Paramètres")
    
    # Note informative sur les données
    if 'data_loaded' not in st.session_state:
        st.sidebar.info("💡 Par défaut, l'application utilise des données synthétiques. Décommentez le code pour charger le vrai fichier.")
    
    # Chargement (Simulation ou Réel)
    # df_raw = load_data("online_retail_II.xlsx") # <- Décommenter pour réel
    DATA_PATH = r"C:\Users\alban\Documents\GitHub\projet_data_viz\data\processed\online_retail_clean.csv"
    df_raw = load_data(DATA_PATH) # Utilise le dummy generator
    df_clean = preprocess_data(df_raw)
        
    # Filtres Globaux
    st.sidebar.subheader("Périmètre d'analyse")
    
    # Filtre Date
    min_date = df_clean['InvoiceDate'].min().date()
    max_date = df_clean['InvoiceDate'].max().date()
    date_range = st.sidebar.date_input(
        "Période",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtre Pays
    all_countries = sorted(df_clean['Country'].unique())
    selected_countries = st.sidebar.multiselect("Pays", all_countries, default=['United Kingdom'])
    if not selected_countries:
        selected_countries = all_countries # Fallback si rien sélectionné
        
    # Filtre Retours
    return_mode = st.sidebar.radio("Gestion des Retours", ["Inclure", "Exclure (Ventes nettes)", "Neutraliser (Ignorer)"])
    
    # Application des filtres
    mask = (df_clean['InvoiceDate'].dt.date >= date_range[0]) & \
           (df_clean['InvoiceDate'].dt.date <= date_range[1]) & \
           (df_clean['Country'].isin(selected_countries))
           
    df_filtered = df_clean[mask].copy()
    
    if return_mode == "Exclure (Ventes nettes)":
        # On garde tout, le total sera calculé (Positif + Négatif) = Net
        pass 
    elif return_mode == "Neutraliser (Ignorer)":
        # On ne garde que les ventes positives
        df_filtered = df_filtered[df_filtered['Quantity'] > 0]
    
    # Indicateur visuel des filtres
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Données actives :** {len(df_filtered):,} lignes")
    if return_mode == "Neutraliser (Ignorer)":
        st.sidebar.warning("Badge : Retours Exclus")

    # --- Titre Principal ---
    st.title("Cockpit Aide à la Décision Marketing")
    st.markdown("Analyse des transactions, Segmentation RFM et Simulation de la CLV.")

    # --- Onglets ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " KPIs (Overview)", 
        " Cohortes (Retention)", 
        " Segments (RFM)", 
        " Scénarios (Simulateur)", 
        " Export"
    ])

    # TAB 1 : KPIs / OVERVIEW
    with tab1:
        st.subheader("Performance Globale sur la période")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_revenue = df_filtered['TotalAmount'].sum()
        n_customers = df_filtered['Customer ID'].nunique()
        n_invoices = df_filtered['Invoice'].nunique()
        avg_basket = total_revenue / n_invoices if n_invoices > 0 else 0
        
        with col1:
            st.metric("Chiffre d'Affaires", f"{total_revenue:,.0f} €")
            with st.expander(" Définition"):
                st.write("Somme totale des transactions (Quantité * Prix). Inclut l'impact des retours si sélectionnés.")
        
        with col2:
            st.metric("Clients Actifs", f"{n_customers:,}")
            with st.expander(" Définition"):
                st.write("Nombre de clients uniques ayant effectué au moins une transaction sur la période.")
        
        with col3:
            st.metric("Panier Moyen", f"{avg_basket:.2f} €")
            with st.expander(" Définition"):
                st.write("CA Total / Nombre de factures uniques.")
                
        with col4:
            # Calcul simple du taux de retour en volume
            total_items = df_filtered[df_filtered['Quantity'] > 0]['Quantity'].sum()
            returned_items = abs(df_filtered[df_filtered['Quantity'] < 0]['Quantity'].sum())
            return_rate = (returned_items / total_items * 100) if total_items > 0 else 0
            st.metric("Taux de Retours (Qté)", f"{return_rate:.1f} %")
            with st.expander(" Définition"):
                st.write("Volume d'articles retournés / Volume d'articles vendus.")

        # Graphiques de tendances
        st.markdown("###  Dynamique des Ventes")
        
        # Aggregation par mois ou semaine
        df_filtered['Period'] = df_filtered['InvoiceDate'].dt.to_period('W').dt.to_timestamp()
        trend_data = df_filtered.groupby('Period')['TotalAmount'].sum().reset_index()
        
        fig_trend = px.line(trend_data, x='Period', y='TotalAmount', title="Évolution du CA Hebdomadaire", markers=True)
        fig_trend.update_layout(xaxis_title="Date", yaxis_title="Revenu (€)")
        st.plotly_chart(fig_trend, use_container_width=True)

    # TAB 2 : COHORTES
    with tab2:
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

    # TAB 3 : SEGMENTS RFM
    with tab3:
        st.subheader("Segmentation RFM (Récence - Fréquence - Montant)")
        
        # Calcul RFM sur les données filtrées
        # Date de référence = lendemain de la dernière date du dataset filtré
        ref_date = df_filtered['InvoiceDate'].max() + dt.timedelta(days=1)
        rfm_df = calculate_rfm(df_filtered, ref_date)
        
        # Attribution segments
        rfm_df['Segment'] = rfm_df.apply(segment_rfm, axis=1)
        
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

    # TAB 4 : SCÉNARIOS & CLV
    with tab4:
        st.subheader("🛠️ Simulateur d'Impact & CLV")
        
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            st.markdown("#### Paramètres de Simulation")
            
            # Paramètres actuels (Baseline) approximatifs
            current_avg_order = rfm_df['Monetary'].mean() / rfm_df['Frequency'].mean()
            current_freq = rfm_df['Frequency'].mean()
            
            # Formule CLV Simple : (Panier Moyen * Fréquence * Marge) * (Retention / (1 + Discount - Retention))
            # On simplifie pour l'app : CLV sur horizon 1 an ou formule infinie. Utilisons formule infinie.
            
            st.markdown("---")
            param_margin = st.slider("Marge Commerciale (%)", 5, 50, 20, step=1) / 100
            param_retention = st.slider("Taux de Rétention Cible (r)", 0.1, 0.9, 0.6, step=0.05)
            param_discount = st.slider("Taux d'Actualisation (d)", 0.05, 0.20, 0.10, step=0.01)
            
            st.markdown("#### Levier Marketing")
            param_promo = st.slider("Remise moyenne accordée (%)", 0, 30, 0) / 100
            
        with col_sim2:
            st.markdown("#### Projection d'Impact")
            
            # Calcul Baseline
            # Hypothèses simplifiées pour la démo
            base_retention = 0.5  # Valeur par défaut
            base_margin = 0.20
            base_discount = 0.10
            
            clv_baseline = (current_avg_order * current_freq * base_margin) * (base_retention / (1 + base_discount - base_retention))
            
            # Calcul Simulation
            # Impact remise : baisse le panier moyen mais peut augmenter rétention (non modélisé ici, l'utilisateur bouge le slider rétention manuellement)
            sim_avg_order = current_avg_order * (1 - param_promo)
            
            # Formule CLV
            try:
                clv_scenario = (sim_avg_order * current_freq * param_margin) * (param_retention / (1 + param_discount - param_retention))
            except ZeroDivisionError:
                clv_scenario = 0
            
            # Affichage comparatif
            col_kpi1, col_kpi2 = st.columns(2)
            col_kpi1.metric("CLV Baseline (Estimée)", f"{clv_baseline:.2f} €")
            
            delta_clv = clv_scenario - clv_baseline
            col_kpi2.metric("CLV Scénario", f"{clv_scenario:.2f} €", delta=f"{delta_clv:.2f} €")
            
            # Graphique impact sur portefeuille
            total_impact = delta_clv * n_customers
            
            data_impact = pd.DataFrame({
                'Scenario': ['Baseline', 'Simulé'],
                'CLV Moyenne': [clv_baseline, clv_scenario]
            })
            
            fig_impact = px.bar(data_impact, x='Scenario', y='CLV Moyenne', color='Scenario', title="Comparaison Valeur Client")
            st.plotly_chart(fig_impact, use_container_width=True)
            
            st.success(f"💰 **Impact Financier Total :** Si ce scénario s'applique à tous vos {n_customers} clients, la variation de valeur du portefeuille est estimée à **{total_impact:,.0f} €**.")
            
            with st.expander("ℹ️ Formule utilisée"):
                st.latex(r'''
                CLV = (Panier \times Freq \times Marge) \times \frac{r}{1 + d - r}
                ''')
                st.write("""
                Où **r** est le taux de rétention et **d** le taux d'actualisation financière.
                Cette simulation suppose un horizon infini.
                """)
    # TAB 5 : EXPORT
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

if __name__ == "__main__":
    main()

