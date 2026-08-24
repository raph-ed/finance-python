import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QuantValuation & Monte Carlo Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/clean financial dashboard look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 8px; border: 1px solid #2d3139; }
    .metric-card { background-color: #1a1c24; padding: 20px; border-radius: 10px; border: 1px solid #2d3139; margin-bottom: 15px; }
    h1, h2, h3 { color: #f0f2f6; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🛠️ Paramètres & Actifs")
st.sidebar.markdown("---")

ticker_input = st.sidebar.text_input("Ticker Principal (Yahoo Finance)", value="AAPL").upper().strip()
peers_input = st.sidebar.text_input("Tickers Comparables (Pairs)", value="MSFT, GOOGL, AMZN, META").upper().strip()
peer_list = [p.strip() for p in peers_input.split(",") if p.strip()]

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Paramètres Monte Carlo")
n_simulations = st.sidebar.slider("Nombre de simulations (chemins)", min_value=100, max_value=10000, value=2500, step=100)
forecast_days = st.sidebar.slider("Horizon de prédiction (jours ouvrés)", min_value=30, max_value=756, value=252, step=21, help="252 jours = 1 an boursier")
history_years = st.sidebar.selectbox("Historique de volatilité", options=[1, 3, 5, 10], index=3)

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Actif de Corrélation Externe")
macro_ticker = st.sidebar.text_input("Actif/Matière/Indice à corréler", value="GC=F", help="Ex: GC=F (Or), CL=F (Pétrole), ^GSPC (S&P 500), BTC-USD").upper().strip()

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_stock_history(ticker, years=10):
    start_date = datetime.datetime.now() - datetime.timedelta(days=years*365 + 30)
    data = yf.download(ticker, start=start_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close'] if 'Close' in data else data.iloc[:, 0]
    elif 'Close' in data.columns:
        data = data['Close']
    return pd.Series(data.dropna().values.ravel(), index=data.dropna().index, name=ticker)

@st.cache_data(ttl=3600)
def fetch_financial_metrics(tickers):
    metrics = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            pe = info.get('trailingPE', np.nan)
            forward_pe = info.get('forwardPE', np.nan)
            ev_ebitda = info.get('enterpriseToEbitda', np.nan)
            ps_ratio = info.get('priceToSalesTrailing12Months', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            market_cap = info.get('marketCap', np.nan)
            name = info.get('shortName', t)
            
            metrics.append({
                'Ticker': t,
                'Nom': name,
                'Market Cap ($B)': market_cap / 1e9 if pd.notnull(market_cap) else np.nan,
                'P/E (Trailing)': pe,
                'Forward P/E': forward_pe,
                'EV/EBITDA': ev_ebitda,
                'P/S': ps_ratio,
                'P/B': pb_ratio
            })
        except Exception:
            metrics.append({'Ticker': t, 'Nom': t})
    return pd.DataFrame(metrics)

# --- MAIN APP LAYOUT ---
st.title("📈 Platforme Quantitative : Valorisation & Monte Carlo (GBM)")
st.markdown(f"Analyse approfondie de **{ticker_input}** couplée à un modèle d'évaluation par multiples sectoriels et simulation stochastique.")

tabs = st.tabs(["📊 1. Modèle de Valorisation (Multiples)", "🎲 2. Simulation Monte Carlo (Mouvement Brownien)", "🔗 3. Étude de Corrélation (Macro & Métriques)"])

# -------------------------------------------------------------
# TAB 1: VALUATION MODEL
# -------------------------------------------------------------
with tabs[0]:
    st.header("1. Valorisation Relative par les Multiples (Peer Valuation)")
    st.markdown("""
    Ce module compare les ratios clés (**P/E, EV/EBITDA, P/S, P/B**) de l'action ciblée avec ses pairs sectoriels.
    - **Barre supérieure à la médiane sectorielle** ➔ L'action se négocie avec une prime (**Surévaluée**).
    - **Barre inférieure à la médiane sectorielle** ➔ L'action se négocie avec une décote (**Sous-évaluée**).
    """)
    
    with st.spinner("Téléchargement des données fondamentales..."):
        all_tickers = [ticker_input] + peer_list
        df_metrics = fetch_financial_metrics(all_tickers)
    
    if not df_metrics.empty:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Tableau des Ratios Financiers")
            st.dataframe(df_metrics.set_index('Ticker').style.format({
                'Market Cap ($B)': '{:.2f}B',
                'P/E (Trailing)': '{:.2f}',
                'Forward P/E': '{:.2f}',
                'EV/EBITDA': '{:.2f}',
                'P/S': '{:.2f}',
                'P/B': '{:.2f}'
            }), height=300)
            
            selected_ratio = st.selectbox(
                "Sélectionnez le Multiple à analyser en Bar Chart :",
                ["P/E (Trailing)", "Forward P/E", "EV/EBITDA", "P/S", "P/B"]
            )
            
        with col2:
            st.subheader(f"Comparaison du Multiple : {selected_ratio}")
            
            # Filter clean data
            df_plot = df_metrics.dropna(subset=[selected_ratio]).copy()
            if not df_plot.empty:
                median_val = df_plot[selected_ratio].median()
                
                # Colors: Accent for target, grey/blue for peers
                colors = []
                for t in df_plot['Ticker']:
                    if t == ticker_input:
                        val = df_plot.loc[df_plot['Ticker'] == t, selected_ratio].values[0]
                        colors.append('#00e676' if val <= median_val else '#ff5252') # Green if cheap, Red if expensive
                    else:
                        colors.append('#3f51b5')
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=df_plot['Ticker'],
                    y=df_plot[selected_ratio],
                    marker_color=colors,
                    text=df_plot[selected_ratio].apply(lambda x: f"{x:.1f}x"),
                    textposition='outside'
                ))
                
                fig_bar.add_hline(
                    y=median_val, 
                    line_dash="dash", 
                    line_color="#ffd600",
                    annotation_text=f"Médiane Sectorielle ({median_val:.1f}x)",
                    annotation_position="top right"
                )
                
                fig_bar.update_layout(
                    template="plotly_dark",
                    title=f"Positionnement relatif du {selected_ratio} vs Secteur",
                    xaxis_title="Ticker",
                    yaxis_title=f"{selected_ratio} (Multiples)",
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Assessment Text
                target_ratio_val = df_metrics.loc[df_metrics['Ticker'] == ticker_input, selected_ratio].values
                if len(target_ratio_val) > 0 and not np.isnan(target_ratio_val[0]):
                    diff_pct = ((target_ratio_val[0] - median_val) / median_val) * 100
                    if diff_pct > 5:
                        st.warning(f"⚠️ **{ticker_input}** est **SURÉVALUÉE** de **{diff_pct:.1f}%** par rapport à la médiane de ses pairs sur le ratio {selected_ratio}.")
                    elif diff_pct < -5:
                        st.success(f"✅ **{ticker_input}** est **SOUS-ÉVALUÉE** (Décote) de **{abs(diff_pct):.1f}%** par rapport à la médiane de ses pairs sur le ratio {selected_ratio}.")
                    else:
                        st.info(f"ℹ️ **{ticker_input}** est alignée avec la valorisation médiane de ses pairs.")

# -------------------------------------------------------------
# TAB 2: MONTE CARLO & BROWNIAN MOTION
# -------------------------------------------------------------
with tabs[1]:
    st.header("2. Simulation Stochastique (Geometric Brownian Motion)")
    st.markdown(r"""
    Le modèle standard de valorisation quantitative repose sur l'équation différentielle stochastique du **Mouvement Brownien Géométrique (GBM)** :
    $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
    En appliquant le Lemme d'Itô, la solution exacte pour le prix à l'instant $t$ est :
    $$S_t = S_0 \exp\left(\left(\mu - rac{1}{2}\sigma^2ight)t + \sigma W_tight)$$
    """)
    
    with st.spinner(f"Téléchargement de l'historique sur {history_years} ans pour {ticker_input}..."):
        hist_prices = fetch_stock_history(ticker_input, years=history_years)
    
    if len(hist_prices) > 30:
        # Calculate daily log returns
        log_returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
        
        mu_daily = log_returns.mean()
        sigma_daily = log_returns.std()
        
        # Annualized values (252 trading days)
        mu_annual = mu_daily * 252
        sigma_annual = sigma_daily * np.sqrt(252)
        s0 = float(hist_prices.iloc[-1])
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Prix Actuel (S₀)", f"${s0:.2f}")
        m_col2.metric("Rendement Moyen Historique (Drift μ)", f"{mu_annual*100:.2f}% /an")
        m_col3.metric("Volatilité Historique (σ)", f"{sigma_annual*100:.2f}% /an")
        m_col4.metric("Simulations Vectorisées", f"{n_simulations:,}")
        
        # Vectorized GBM Simulation (Numpy)
        dt = 1 # 1 day step
        drift = (mu_daily - 0.5 * (sigma_daily ** 2))
        
        # Random standard normal matrix of shape (forecast_days, n_simulations)
        random_shocks = np.random.normal(0, 1, (forecast_days, n_simulations))
        daily_log_returns_sim = drift + sigma_daily * random_shocks
        
        # Cumsum to get price trajectory
        price_paths = np.vstack([np.zeros((1, n_simulations)), np.cumsum(daily_log_returns_sim, axis=0)])
        simulated_prices = s0 * np.exp(price_paths)
        
        final_prices = simulated_prices[-1, :]
        p5 = np.percentile(final_prices, 5)
        p25 = np.percentile(final_prices, 25)
        p50 = np.percentile(final_prices, 50) # Median (most probable path)
        p75 = np.percentile(final_prices, 75)
        p95 = np.percentile(final_prices, 95)
        
        # Layout plots
        col_chart1, col_chart2 = st.columns([3, 2])
        
        with col_chart1:
            st.subheader(f"Trajectoires Monte Carlo ({min(n_simulations, 100)} affichées)")
            fig_sim = go.Figure()
            
            # Plot first 100 sample paths for performance and clear display
            display_samples = min(n_simulations, 120)
            time_axis = list(range(forecast_days + 1))
            
            for i in range(display_samples):
                fig_sim.add_trace(go.Scatter(
                    x=time_axis,
                    y=simulated_prices[:, i],
                    mode='lines',
                    line=dict(width=0.8, color='rgba(0, 230, 118, 0.15)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add Median Path
            median_path = np.median(simulated_prices, axis=1)
            fig_sim.add_trace(go.Scatter(
                x=time_axis,
                y=median_path,
                mode='lines',
                line=dict(color='#ffd600', width=3),
                name='Trajectoire Médiane (La + probable)'
            ))
            
            # Percentiles 5% and 95%
            p5_path = np.percentile(simulated_prices, 5, axis=1)
            p95_path = np.percentile(simulated_prices, 95, axis=1)
            fig_sim.add_trace(go.Scatter(
                x=time_axis, y=p95_path, mode='lines',
                line=dict(color='#00e5ff', width=1.5, dash='dash'), name='Percentile 95% (Scénario Haussier)'
            ))
            fig_sim.add_trace(go.Scatter(
                x=time_axis, y=p5_path, mode='lines',
                line=dict(color='#ff5252', width=1.5, dash='dash'), name='Percentile 5% (Scénario Baissier / VaR)'
            ))
            
            fig_sim.update_layout(
                template="plotly_dark",
                xaxis_title="Jours de Bourse Futurs",
                yaxis_title="Prix ($)",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
            )
            st.plotly_chart(fig_sim, use_container_width=True)
            
        with col_chart2:
            st.subheader("Distribution du Prix Final")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=final_prices,
                nbinsx=50,
                marker_color='#3f51b5',
                opacity=0.75,
                name="Fréquence"
            ))
            fig_dist.add_vline(x=p50, line_color="#ffd600", line_width=2, line_dash="solid", annotation_text=f"Médiane : ${p50:.2f}")
            fig_dist.add_vline(x=p5, line_color="#ff5252", line_width=2, line_dash="dash", annotation_text=f"5% : ${p5:.2f}")
            fig_dist.add_vline(x=p95, line_color="#00e5ff", line_width=2, line_dash="dash", annotation_text=f"95% : ${p95:.2f}")
            
            fig_dist.update_layout(
                template="plotly_dark",
                xaxis_title=f"Prix après {forecast_days} jours ($)",
                yaxis_title="Nombre d'occurrences",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        # Summary Quant Table
        st.subheader("📊 Métriques de Risque & Probabilités Associées")
        q1, q2, q3, q4, q5 = st.columns(5)
        prob_gain = (np.sum(final_prices > s0) / n_simulations) * 100
        
        q1.metric("Prix Plus Probable (Médiane)", f"${p50:.2f}", f"{((p50-s0)/s0)*100:+.1f}%")
        q2.metric("Scénario Baissier (5%)", f"${p5:.2f}", f"{((p5-s0)/s0)*100:+.1f}%")
        q3.metric("Scénario Haussier (95%)", f"${p95:.2f}", f"{((p95-s0)/s0)*100:+.1f}%")
        q4.metric("Probabilité de Plus-Value", f"{prob_gain:.1f}%")
        q5.metric("Value at Risk (95% VaR)", f"-${s0 - p5:.2f}")

    else:
        st.error(f"Données insuffisantes pour {ticker_input}. Veuillez vérifier le symbole boursier.")

# -------------------------------------------------------------
# TAB 3: CORRELATION & MACRO ANALYSIS
# -------------------------------------------------------------
with tabs[2]:
    st.header("3. Analyse de Corrélation (Test d'Hypothèses & Données Historiques)")
    st.markdown("""
    Comparez l'historique sur 10 ans de votre action avec un indice macroéconomique, une matière première ou un actif externe 
    pour vérifier si les hypothèses intuitives (ex: corrélation entre deux marchés) sont statistiquement valables ou illusoires.
    """)
    
    with st.spinner("Téléchargement des séries temporelles synchronisées..."):
        s_target = fetch_stock_history(ticker_input, years=history_years)
        s_macro = fetch_stock_history(macro_ticker, years=history_years)
        
        df_corr = pd.DataFrame({
            ticker_input: s_target,
            macro_ticker: s_macro
        }).dropna()
        
    if not df_corr.empty:
        # Normalized performance (base 100)
        df_norm = (df_corr / df_corr.iloc[0]) * 100
        
        # Pearson & Spearman correlation
        pearson_corr = df_corr[ticker_input].corr(df_corr[macro_ticker])
        returns_corr = df_corr[ticker_input].pct_change().corr(df_corr[macro_ticker].pct_change())
        
        c_col1, c_col2 = st.columns(2)
        c_col1.metric("Corrélation de Prix (Niveau)", f"{pearson_corr:.3f}", help="De -1 (opposition totale) à +1 (parfaite synchronisation)")
        c_col2.metric("Corrélation des Rendements Quotidiens", f"{returns_corr:.3f}", help="Plus pertinent pour le trading : mesure si les variations journalières vont dans le même sens")
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=df_norm.index, y=df_norm[ticker_input], name=f"{ticker_input} (Base 100)", line=dict(color='#00e676', width=2)))
        fig_dual.add_trace(go.Scatter(x=df_norm.index, y=df_norm[macro_ticker], name=f"{macro_ticker} (Base 100)", line=dict(color='#ff9100', width=2)))
        
        fig_dual.update_layout(
            template="plotly_dark",
            title=f"Performance Comparée (Base 100) : {ticker_input} vs {macro_ticker}",
            xaxis_title="Date",
            yaxis_title="Évolution relative (Base 100)",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_dual, use_container_width=True)
