# 📊 Quant Valuation & Monte Carlo Brownian Motion Suite

Bienvenue dans la suite d'analyse quantitative et financière conçue pour évaluer les actions d'entreprises et modéliser leurs trajectoires stochastiques selon les standards des fonds d'investissement et des quants.

---

## 🎯 Vue d'ensemble du Programme

Cette suite combine les deux approches majeures de la finance de marché :
1. **L'Analyse Fondamentale Relative (Multiples de Valorisation)** : Comparaison des ratios P/E (PER), EV/EBITDA, P/S, P/B par rapport aux concurrents du même secteur pour déterminer objectivement si une action est sous-évaluée (décote d'achat) ou surévaluée (prime de risque).
2. **L'Analyse Stochastique Prédictive (Mouvement Brownien Géométrique / Monte Carlo)** : Simulation vectorisée ultra-rapide de milliers de trajectoires futures possibles sur 10 ans de volatilité historique, identifiant la variation médiane la plus probable ainsi que les intervalles de confiance (5% et 95%).
3. **Le Module de Corrélation & Hypothèses Macro** : Analyse de synchronisation des séries temporelles (Base 100, corrélation de Pearson et corrélation des rendements journaliers) pour tester la robustesse des intuitions causales.

---

## 🚀 Comment lancer le programme en 1 clic

### Prérequis
- Avoir **Python 3.9+** installé sur votre machine.

### Méthode 1 : Interface Web Interactive (Recommandé)
- **Sur Windows** : Double-cliquez simplement sur `run.bat`
- **Sur Mac / Linux** : Ouvrez un terminal, placez-vous dans le dossier et exécutez :
  ```bash
  chmod +x run.sh
  ./run.sh
  ```
- **Manuellement via le Terminal** :
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```
Le tableau de bord s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`.

### Méthode 2 : Lancement Rapide en Ligne de Commande (CLI)
Pour une analyse instantanée sans navigateur :
```bash
python quant_cli.py AAPL
python quant_cli.py TSLA
python quant_cli.py NVDA
```

---

## 📐 Fondements Mathématiques & Financiers

### 1. Le Mouvement Brownien Géométrique (Geometric Brownian Motion - GBM)
Le cours de l'actif suit une équation différentielle stochastique (EDS) standard :
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Où :
- $S_t$ : Prix de l'action au temps $t$.
- $\mu$ : Rendement moyen arithmétique annualisé (*drift* ou dérive).
- $\sigma$ : Volatilité historique annualisée (écart-type des rendements log).
- $W_t$ : Processus de Wiener / Mouvement brownien standard avec $W_t \sim \mathcal{N}(0, t)$.

En appliquant le **Lemme d'Itô** pour résoudre l'équation en temps continu :
$$S_t = S_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma W_t \right)$$

### 2. Optimisation et Vectorisation (Numpy)
Pour permettre à n'importe quel ordinateur portable (même sans carte graphique) de calculer 10 000 chemins en une fraction de seconde, le code évite les boucles lentes `for` et utilise des matrices 2D sous `numpy` :
- Matrice de chocs aléatoires $\mathbf{Z} \in \mathbb{R}^{\text{jours} \times \text{simulations}} \sim \mathcal{N}(0, 1)$
- Somme cumulée vectorisée `np.cumsum`

---

## 📊 Interprétation des Résultats Graphiques

1. **Graphique en Barres de Valorisation** :
   - **Barre Rouge (ou > Médiane)** : L'action se négocie avec un multiple plus élevé que ses pairs. Elle est **surévaluée** par rapport au marché (ou le marché anticipe une croissance future très forte).
   - **Barre Verte (ou < Médiane)** : L'action est moins chère que ses pairs. Elle est **sous-évaluée** (décote d'opportunité d'achat).
2. **Faisceau Monte Carlo & Médiane Jaune** :
   - La ligne dorée représente le **percentile 50% (la médiane)** : c'est la valeur centrale statistiquement la plus probable au terme de l'horizon de temps.
   - Les bornes en pointillés (5% rouge et 95% cyan) délimitent l'intervalle de confiance à 90%.

---

## 📁 Structure des Fichiers du Projet

```text
quant_valuation_monte_carlo/
├── app.py              # Application principale Streamlit avec graphiques interactifs Plotly
├── quant_cli.py        # Version terminal ultra-légère
├── requirements.txt    # Bibliothèques Python requises
├── run.bat             # Script de lancement automatique pour Windows
├── run.sh              # Script de lancement automatique pour Mac/Linux
└── README.md           # Documentation complète et guide théorique
```
