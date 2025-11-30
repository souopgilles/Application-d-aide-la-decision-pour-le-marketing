import streamlit as st
import pandas as pd

@st.cache_data
def data():
    df = pd.read_excel(r"C:\Users\alban\Documents\GitHub\projet_data_viz\data\online_retail_II.xlsx")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["Customer_ID"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df = df[df["Invoice"].astype(str).str.startswith("C") == False]
    df["Revenue"] = df["Quantity"] * df["Price"]
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

def main():
    st.set_page_config(layout="wide")

    df = data()

    ca_total = df["Revenue"].sum()
    n_clients = df["Customer_ID"].nunique()
    ca_mean = ca_total / n_clients

    orders = df.groupby("Customer_ID")["Invoice"].nunique()
    r_base = (orders >= 2).mean()

    marge_base = 0.30
    remise_base = 0.0
    d_base = 0.10

    st.title("CLV   CA   Rétention")

    kpis_block(ca_mean, r_base, marge_base, remise_base, d_base)

    st.subheader("Scénario global : Marge + Remise + Rétention + d")

    col1, col2, col3 = st.columns(3)
    with col1:
        val_m = st.slider("Marge  (%)", 0.0, 100.0, marge_base * 100, 1.0)
        val_rem = st.slider("Remise  (%)", 0.0, 50.0, remise_base * 100, 1.0)
    with col2:
        val_r = st.slider("Rétention  (0–1)", 0.0, 0.95, min(0.9, r_base + 0.05), 0.01)
    with col3:
        d_scen = st.slider("Taux d’actualisation d ", 0.0, 0.5, d_base, 0.01)

    marge_scen = val_m / 100
    remise_scen = val_rem / 100
    r_scen = val_r

    scenario_block(
        "Scénario global : comparaison baseline vs scénario",
        ca_mean, r_base, marge_base, remise_base,
        d_base, d_scen, remise_scen, marge_scen, r_scen
    )

if __name__ == "__main__":
    main()
