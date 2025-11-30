# Application d'Aide à la Décision Marketing (Retail)

## Description du Projet

Ce projet est une application de Business Intelligence (BI) interactive conçue pour les équipes marketing dans le secteur du e-commerce. Elle transforme des données transactionnelles brutes en indicateurs actionnables.

L'objectif est de piloter la stratégie CRM en répondant à trois questions clés :

Rétention : Est-ce que nos nouveaux clients reviennent acheter les mois suivants ? (Analyse de Cohortes)

Segmentation : Qui sont nos meilleurs clients et lesquels sont à risque ? (Segmentation RFM)

Prévision : Quel est l'impact financier d'une modification de la marge ou de la rétention ? (Simulation CLV)

## Fonctionnalités Principales

L'application est divisée en 5 volets stratégiques :

Vue d'ensemble (KPIs) : Tableau de bord instantané (CA, Panier moyen, Clients actifs, Taux de retour).

* Analyse de Cohortes : Heatmap interactive pour visualiser la rétention client (M+1, M+2...) et la densité de revenu par ancienneté.

* Segmentation RFM : Classification automatique des clients (Champions, Fidèles, À risque, Hibernants) basée sur la Récence, la Fréquence et le Montant.

* Simulateur de Scénarios : Outil de "What-if analysis". Permet de simuler l'impact d'une remise ou d'une hausse de rétention sur la Valeur Vie Client (CLV).

* Export de Données : Génération de listes de clients ciblées (CSV) prêtes à être injectées dans des outils d'emailing ou de publicité (Facebook Ads/Google Ads).

## Installation et Démarrage

Prérequis

Python 3.8 ou supérieur

Un environnement virtuel est recommandé

1. Cloner ou télécharger le projet

Placez tous les fichiers (app.py, requirements.txt) dans un dossier.

2. Installer les dépendances

Ouvrez votre terminal dans le dossier du projet et exécutez : ## pip install -r requirements.txt


3. Lancer l'application

Toujours dans le terminal : streamlit run app.py


L'application s'ouvrira automatiquement dans votre navigateur à l'adresse http://localhost:8501.

📂 Gestion des Données

Mode Démo (Par défaut)

L'application démarre avec un générateur de données synthétiques. Vous n'avez besoin d'aucun fichier pour tester l'interface. Elle crée automatiquement des transactions réalistes pour la démonstration.

Mode Réel (Vos données)

Pour utiliser vos propres données (fichier Online Retail II ou équivalent) :

Placez votre fichier .xlsx ou .csv dans le dossier du projet.

Ouvrez app.py.

Cherchez la fonction main() (vers la ligne 150).

Commentez la ligne de chargement fictif et décommentez la ligne de chargement réel :

# Dans app.py :

# df_raw = load_data(None)           # <--- Commenter cette ligne
df_raw = load_data("votre_fichier.xlsx") # <--- Décommenter celle-ci


 Méthodologies Utilisées

Segmentation RFM

Les clients sont notés de 1 à 4 sur trois axes :

Récence (R) : Date de la dernière commande.

Fréquence (F) : Nombre total de commandes.

Montant (M) : Chiffre d'affaires total généré.

Calcul de la CLV (Customer Lifetime Value)

Le simulateur utilise une formule de CLV simplifiée sur horizon infini pour estimer la valeur future :

$$CLV = (Panier Moyen \times Fréquence \times Marge) \times \frac{r}{1 + d - r}$$

r : Taux de rétention

d : Taux d'actualisation (coût du capital)



Source des Données

Le jeu de données de référence utilisé pour la structure est le Online Retail II Data Set, fourni par l'UCI Machine Learning Repository. Il contient les transactions d'un détaillant en ligne britannique (cadeaux, maison, déco) entre 2009 et 2011.
