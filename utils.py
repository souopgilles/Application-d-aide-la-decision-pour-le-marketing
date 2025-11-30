import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st

# --- 1. CHARGEMENT & NETTOYAGE ---
@st.cache_data
def load_data(filepath):
    cols = ['Invoice', 'Quantity', 'InvoiceDate', 'Customer ID', 'Country', 'TotalPrice']
    try:
        df = pd.read_csv(filepath, usecols=cols)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df = df.dropna(subset=['Customer ID']) #supprime les NaN si présent
        df['Customer ID'] = df['Customer ID'].astype(int).astype(str) #eviter d'avoir des .0 et le mettre sous forme d'étiquette

        # Période pour les cohortes (Mois)
        # On le fait ici pour l'avoir dans le cache global
        df['CohortMonth'] = df.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M') # on s'interesse au mois de la 1ere fois que le client achete
        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M') #mois d'achat
        return df
    except Exception as e:
        # En cas d'erreur, on retourne un DF vide pour éviter le crash immédiat
        return pd.DataFrame()


# --- 2. FILTRES (REQ. PROJET) ---
@st.cache_data
def filter_data(df, countries, date_range, min_order):
    if df.empty: return df
    data = df.copy()

    # Filtre Pays
    if countries:
        data = data[data['Country'].isin(countries)] # Si vide alors on affiche tous les pays

    # Filtre Date
    if len(date_range) == 2:
        # Conversion en date pure pour la comparaison
        mask = (data['InvoiceDate'].dt.date >= date_range[0]) & (data['InvoiceDate'].dt.date <= date_range[1]) # InvoiceDate contient des heures donc on les retire avec .dt.date
        # on veut que ce soit après le début et avant la fin
        data = data[mask]

    # Seuil commande
    if min_order > 0:
        # On filtre les factures dont le total est inférieur au seuil
        valid_inv = data.groupby('Invoice')['TotalPrice'].sum().loc[lambda x: x >= min_order].index # On va grouper avec index invoice puis on sum et on garde seulement lsl
        data = data[data['Invoice'].isin(valid_inv)] #on filtre

    return data


# --- 3. SEGMENTATION RFM (PRÉ-REQUIS CLV PAR SEGMENT) --- (A REMPLACER PAR LA PARTIE DE GILLE)
def calculate_rfm_segments(df):

    if df.empty: return pd.DataFrame()

    # Date de référence = lendemain de la dernière commande
    now = df['InvoiceDate'].max() + dt.timedelta(days=1) # On prend l'invoiceDate la plus tard et on ajoute 1 pour definir notre date

    # Agrégation par client
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (now - x.max()).days, #pour chaque client on prend sa date la plus récente et on soustrait par now
        'Invoice': 'nunique', # on compte le nombre de facture unique par client
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'Monetary'})

    # Scores (Quartiles)
    labels = [1, 2, 3, 4]
    
    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), q=4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=labels)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=4, labels=labels)
    
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    def segment_label(score):
        if score >= 11:
            return 'Champions'
        elif score >= 9:
            return 'Fidèles'
        elif score >= 7:
            return 'Prometteurs'
        elif score >= 5:
            return 'À Risque'
        else:
            return 'Hibernants'

    #rfm['Segment'] = rfm['RFM_Sum'].apply(segment_label)
    rfm['Segment'] = rfm.apply(segment_rfm, axis=1)
    return rfm

# --- 3. FORMULE RFM GILLE ---
def calculate_rfm(df, reference_date):
    """Calcule le score RFM."""
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
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
# --- 3.suite . SEGMENTATION RFM ---
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
    

# --- 4. MÉTRIQUES CLV (GLOBAL & PAR SEGMENT) ---
def calculate_clv_metrics(df, rfm_df):
    """
    Calcule les variables clés pour la formule CLV (Panier, Fréquence)
    """
    # Métriques Globales
    if df['Invoice'].nunique() > 0:
        avg_basket = df['TotalPrice'].sum() / df['Invoice'].nunique() # On divise le chiffre d'affaire total par le nombre de commande
    else:
        avg_basket = 0

    if df['Customer ID'].nunique() > 0:
        purchase_freq = df['Invoice'].nunique() / df['Customer ID'].nunique() # On divise le nombre de factures unique par le nbr de clients uniques
    else:
        purchase_freq = 0

    # Métriques par Segment
    # On joint les infos brutes avec le segment
    if not rfm_df.empty:
        df_merged = df.merge(rfm_df[['Segment']], on='Customer ID', how='left') # Il nous rest 5 lignes pour champions, fideles etc

        # Agrégation par segment
        seg_metrics = df_merged.groupby('Segment').agg( # Pour chaque catégorie de score on calcule ces éléments
            CA_Total=('TotalPrice', 'sum'),
            Nb_Commandes=('Invoice', 'nunique'),
            Nb_Clients=('Customer ID', 'nunique')
        ).reset_index()

        # Seg metrics contiendra donc ces 3 colonnes avec comme index les 5 lignes des 5 catégories

        # On créer 2 nouvelles colonnes
        seg_metrics['Panier_Moyen'] = seg_metrics['CA_Total'] / seg_metrics['Nb_Commandes']
        seg_metrics['Frequence_Achat'] = seg_metrics['Nb_Commandes'] / seg_metrics['Nb_Clients']
    else:
        seg_metrics = pd.DataFrame()

    return avg_basket, purchase_freq, seg_metrics


# --- 5. FORMULE CLV ---
def predict_clv(panier, frequence, marge, retention, discount):
    """
    Formule : CLV = (Panier * Frequence * Marge) * (r / (1 + d - r))
    """
    valeur_client = panier * frequence * marge

    # Sécurité division par zéro
    denom = (1 + discount - retention)
    if denom <= 0.01: denom = 0.01

    return valeur_client * (retention / denom)


# --- 6. CLV EMPIRIQUE (HISTORIQUE) ---
def get_empirical_clv(df):
    """
    Calcule le revenu cumulé moyen par client en fonction de l'ancienneté (Cohorte).
    """
    data = df.copy()

    # --- CORRECTIF ROBUSTESSE ---
    # Si InvoiceMonth ou CohortMonth manquent (à cause du cache Streamlit), on les recalcule ici
    if 'InvoiceMonth' not in data.columns:
        data['InvoiceMonth'] = data['InvoiceDate'].dt.to_period('M')

    if 'CohortMonth' not in data.columns:
        # Recalcul du mois de la première commande par client pour ce dataset
        data['CohortMonth'] = data.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M')
    # -----------------------------

    # Calcul de l'index de mois (0, 1, 2...) depuis le premier achat
    # .apply(lambda x: x.n) convertit la différence de périodes en nombre entier
    data['CohortIndex'] = (data['InvoiceMonth'] - data['CohortMonth']).apply(lambda x: x.n)

    # Revenu total par Cohorte et par Index (Mois 0, Mois 1, etc.)
    cohort_rev = data.groupby(['CohortMonth', 'CohortIndex'])['TotalPrice'].sum().reset_index() #il faut imaginer quelque chose comme Mois 0, janvier (A et B) 150

    # Taille initiale de chaque cohorte (combien de clients ont commencé ce mois-là ?)
    cohort_sizes = data.groupby('CohortMonth')['Customer ID'].nunique().reset_index()
    cohort_sizes.columns = ['CohortMonth', 'CohortSize'] # Cohorte A et sa taille

    # Fusion pour avoir le revenu et la taille
    cohort_data = cohort_rev.merge(cohort_sizes, on='CohortMonth')
    # Cohorte A | Mois 0 | Revenu sum | taille nunique
    # Cohorte A | Mois 1 | Revenu sum | taille nunique

    # Revenu moyen par client pour ce mois spécifique (ex: mois +1)
    cohort_data['RevPerClient'] = cohort_data['TotalPrice'] / cohort_data['CohortSize']
    # Cohorte A | Mois 0 | Revenu sum / taille nunique = avg0
    # Cohorte A | Mois 1 | Revenu sum / taille nunique = avg1

    # Moyenne globale par Index (Mois 0, Mois 1...) toutes cohortes confondues
    avg_curve = cohort_data.groupby('CohortIndex')['RevPerClient'].mean().reset_index()
    # Cohorte ABCDE... | Mois 0 | avgtotale0
    # Cohorte ABCDE... | Mois 1 | avgtotale1

    # CLV Empirique = Somme cumulée du revenu moyen au fil du temps
    avg_curve['CLV_Empirique'] = avg_curve['RevPerClient'].cumsum()
    # Cohorte ABCDE... | Mois 0 | avgtotale0
    # Cohorte ABCDE... | Mois 1 | avgtotale0 + avgtotale1

    return avg_curve

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
        cohort_data = grouping['TotalPrice'].sum().reset_index()
        cohort_sums = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='TotalPrice')
        return cohort_sums, None
    
def preprocess_data(df):
    """Nettoyage et Feature Engineering de base."""
    df = df.copy()
    
    # Suppression des Customer ID nuls (essentiel pour RFM)
    df.dropna(subset=['Customer ID'], inplace=True)
    df['Customer ID'] = df['Customer ID'].astype(str).str.split('.').str[0]
    
    
    # Colonnes temporelles
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
    df['InvoiceDate_Date'] = df['InvoiceDate'].dt.date
    
    return df

def clv(ca, marge, r, d):
    denom = 1 + d - r
    return ca * marge * r / denom

def kpis_block(ca_mean, r_base, marge_base, remise_base, d):
    st.markdown(
        """
        CLV = (CA × marge × r) / (1 + d − r)

        r = probabilité de rétention (entre 0 et 1)  
        d = taux d’actualisation (entre 0 et 1)  

        les données sont calculées sur l'ensemble de la période disponible.
        """
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("CLV baseline", f"{clv(ca_mean, marge_base, r_base, d):.2f} €")
    with col2:
        st.metric("CA moyen (€/client)", f"{ca_mean:.2f} €")
    with col3:
        st.metric("Marge baseline", f"{marge_base*100:.1f} %")
    with col4:
        st.metric("Remise baseline", f"{remise_base*100:.1f} %")
    with col5:
        st.metric("Rétention baseline", f"{r_base:.1%}")

def scenario_block(titre, ca_mean, r_base, marge_base, remise_base, d_base,
                   d_new, remise_new, marge_new, r_new):

    st.subheader(titre)

    ca_base = ca_mean
    ca_scenario = ca_mean * (1 - remise_new)

    clv_base = clv(ca_base, marge_base, r_base, d_base)
    clv_new = clv(ca_scenario, marge_new, r_new, d_new)

    df_ca = pd.DataFrame(
        {"Type": ["Baseline", "Scénario global"], "CA": [ca_base, ca_scenario]}
    ).set_index("Type")
    df_r = pd.DataFrame(
        {"Type": ["Baseline", "Scénario global"], "Rétention": [r_base, r_new]}
    ).set_index("Type")
    df_clv = pd.DataFrame(
        {"Type": ["Baseline", "Scénario global"], "CLV": [clv_base, clv_new]}
    ).set_index("Type")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("CA  Baseline vs Scénario global")
        st.bar_chart(df_ca, height=250)

    with col2:
        st.markdown(" Rétention  Baseline vs Scénario global")
        st.bar_chart(df_r, height=250)

    with col3:
        st.markdown(" CLV  Baseline vs Scénario global")
        st.bar_chart(df_clv, height=250)

    st.markdown(
        f"Baseline  Marge : {marge_base*100:.1f}%, Remise : {remise_base*100:.1f}%, "
        f"Rétention : {r_base:.1%}, d : {d_base:.2f}  \n"
        f"Scénario global  Marge : {marge_new*100:.1f}%, Remise : {remise_new*100:.1f}%, "
        f"Rétention : {r_new:.1%}, d : {d_new:.2f}"
    )