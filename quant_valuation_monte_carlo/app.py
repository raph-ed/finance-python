import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Quantitative Valuation & Risk Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THÈME GRAPHIQUE PROFESSIONNEL (SOMBRE INSTITUTIONNEL) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #080a0f;
        color: #e1e7ec;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0e121a;
        border-right: 1px solid #1c2433;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] {
        background-color: #0f141f;
        border: 1px solid #1c2433;
        padding: 14px 18px;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1c2433;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 4px 4px 0 0;
        color: #8da2b5;
        background-color: transparent;
        border: none;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background-color: #161e2e !important;
        border-bottom: 2px solid #2979ff !important;
    }
    
    .header-box {
        padding: 16px 0 20px 0;
        border-bottom: 1px solid #1c2433;
        margin-bottom: 24px;
    }
    
    .badge-bull { color: #00c853; font-weight: 600; }
    .badge-bear { color: #d50000; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- RECHERCHE ET AUTOCOMPLÉTION D'ACTIFS ---
@st.cache_data(ttl=1800, show_spinner=False)
def search_securities(query: str):
    if not query or len(query.strip()) < 2:
        return []
    try:
        s = yf.Search(query, max_results=10)
        quotes = s.quotes
        results = []
        for q in quotes:
            symbol = q.get('symbol', '')
            name = q.get('shortname') or q.get('longname') or symbol
            quote_type = q.get('quoteType', 'ACTIF')
            exch = q.get('exchange', '')
            label = f"{symbol} - {name} ({exch} | {quote_type})"
            results.append((symbol, label))
        return results
    except Exception:
        return []

# --- CHARGEMENT DES SÉRIES TEMPORELLES ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_series(ticker: str, years: int = 5):
    start_date = datetime.datetime.now() - datetime.timedelta(days=years * 365 + 30)
    data = yf.download(ticker, start=start_date, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'] if 'Close' in data else data.iloc[:, 0]
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data.iloc[:, 0]
    return pd.Series(prices.dropna().values.ravel(), index=prices.dropna().index, name=ticker)

# --- CHARGEMENT DES FONDAMENTAUX ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(tickers: tuple):
    metrics = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            metrics.append({
                'Ticker': t,
                'Nom': info.get('shortName', t),
                'Capitalisation ($B)': (info.get('marketCap') or np.nan) / 1e9,
                'P/E (Trailing)': info.get('trailingPE', np.nan),
                'Forward P/E': info.get('forwardPE', np.nan),
                'EV/EBITDA': info.get('enterpriseToEbitda', np.nan),
                'P/S': info.get('priceToSalesTrailing12Months', np.nan),
                'P/B': info.get('priceToBook', np.nan)
            })
        except Exception:
            metrics.append({'Ticker': t, 'Nom': t})
    return pd.DataFrame(metrics)

# --- BARRE LATÉRALE DE CONTRÔLE ---
st.sidebar.markdown("### Univers & Paramètres")

search_term = st.sidebar.text_input("Recherche d'actif (ex: nasdaq, aapl, sp500, qqq)", value="NASDAQ")
search_results = search_securities(search_term)

if search_results:
    selected_tuple = st.sidebar.selectbox(
        "Résultats disponibles :",
        options=search_results,
        format_func=lambda x: x[1]
    )
    primary_ticker = selected_tuple[0]
else:
    fallback = search_term.upper().strip() if search_term else "QQQ"
    st.sidebar.caption(f"Recherche directe du symbole : **{fallback}**")
    primary_ticker = fallback

peers_input = st.sidebar.text_input(
    "Actifs comparables (séparés par des virgules)", 
    value="SPY, IWM, DIA" if "QQQ" in primary_ticker or "IXIC" in primary_ticker else "MSFT, GOOGL, AMZN, NVDA"
)
peer_list = [p.strip().upper() for p in peers_input.split(",") if p.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("### Modélisation Stochastique")
n_simulations = st.sidebar.slider("Trajectoires Monte Carlo", min_value=500, max_value=10000, value=2500, step=500)
forecast_days = st.sidebar.slider("Horizon de prévision (jours ouvrés)", min_value=21, max_value=504, value=252, step=21)
history_years = st.sidebar.selectbox("Profondeur historique (années)", options=[1, 3, 5, 10], index=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### Facteur Macro / Benchmark")
macro_ticker = st.sidebar.text_input("Symbole de référence", value="^GSPC")

# --- EN-TÊTE PRINCIPAL ---
st.markdown(f"""
<div class="header-box">
    <div style="font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; color: #5f748a;">Plateforme Quantitative Institutionnelle</div>
    <div style="font-size: 1.7rem; font-weight: 700; color: #ffffff; margin-top: 4px;">ANALYSE D'ACTIF : {primary_ticker}</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "1. Multiples & Valorisation",
    "2. Mouvement Brownien & Monte Carlo",
    "3. Gestion des Risques (VaR & CVaR)",
    "4. Analyse Factorielle & Corrélations"
])

# -------------------------------------------------------------
# ONGLET 1 : VALORISATION PAR MULTIPLES
# -------------------------------------------------------------
with tabs[0]:
    st.markdown("#### Valorisation Relative Sectorielle")
    
    with st.spinner("Extraction des ratios de marché..."):
        all_tickers = tuple([primary_ticker] + [p for p in peer_list if p != primary_ticker])
        df_metrics = fetch_fundamentals(all_tickers)
    
    if not df_metrics.empty and 'P/E (Trailing)' in df_metrics.columns:
        col_tbl, col_chart = st.columns([1, 1])
        
        with col_tbl:
            st.dataframe(
                df_metrics.set_index('Ticker').style.format({
                    'Capitalisation ($B)': lambda x: f"{x:.2f}B" if pd.notnull(x) else "-",
                    'P/E (Trailing)': lambda x: f"{x:.2f}" if pd.notnull(x) else "-",
                    'Forward P/E': lambda x: f"{x:.2f}" if pd.notnull(x) else "-",
                    'EV/EBITDA': lambda x: f"{x:.2f}" if pd.notnull(x) else "-",
                    'P/S': lambda x: f"{x:.2f}" if pd.notnull(x) else "-",
                    'P/B': lambda x: f"{x:.2f}" if pd.notnull(x) else "-"
                }),
                use_container_width=True
            )
            selected_ratio = st.selectbox(
                "Multiple de valorisation à analyser :",
                ["P/E (Trailing)", "Forward P/E", "EV/EBITDA", "P/S", "P/B"]
            )
            
        with col_chart:
            df_plot = df_metrics.dropna(subset=[selected_ratio]).copy()
            if not df_plot.empty:
                med_val = df_plot[selected_ratio].median()
                
                bar_colors = []
                for t in df_plot['Ticker']:
                    if t == primary_ticker:
                        val = df_plot.loc[df_plot['Ticker'] == t, selected_ratio].values[0]
                        bar_colors.append('#00c853' if val <= med_val else '#d50000')
                    else:
                        bar_colors.append('#2979ff')
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=df_plot['Ticker'],
                    y=df_plot[selected_ratio],
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=df_plot[selected_ratio].apply(lambda x: f"{x:.1f}x"),
                    textposition='outside'
                ))
                fig_bar.add_hline(
                    y=med_val,
                    line=dict(color='#ffab00', dash='dash', width=1.5),
                    annotation_text=f"Médiane : {med_val:.1f}x",
                    annotation_position="top right"
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#080a0f",
                    plot_bgcolor="#0e121a",
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis_title=f"Multiple ({selected_ratio})"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------------------
# ONGLET 2 : MONTE CARLO & GBM
# -------------------------------------------------------------
with tabs[1]:
    st.markdown("#### Modélisation Stochastique (Mouvement Brownien Géométrique)")
    st.markdown(r"""
    Équation différentielle stochastique (EDS) sous la filtration de Wiener :
    $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
    
    Solution analytique continue déduite par le Lemme d'Itô :
    $$S_t = S_0 \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma W_t \right)$$
    """)
    
    with st.spinner("Téléchargement des données de clôture..."):
        hist_prices = fetch_historical_series(primary_ticker, years=history_years)
    
    if len(hist_prices) > 30:
        log_ret = np.log(hist_prices / hist_prices.shift(1)).dropna()
        mu_daily = float(log_ret.mean())
        sigma_daily = float(log_ret.std())
        
        mu_ann = mu_daily * 252
        sigma_ann = sigma_daily * np.sqrt(252)
        s0 = float(hist_prices.iloc[-1])
        
        # Simulation vectorisée
        shocks = np.random.normal(0, 1, (forecast_days, n_simulations))
        drift = (mu_daily - 0.5 * (sigma_daily ** 2))
        daily_returns_sim = drift + sigma_daily * shocks
        price_trajectories = s0 * np.exp(np.vstack([np.zeros((1, n_simulations)), np.cumsum(daily_returns_sim, axis=0)]))
        
        final_prices = price_trajectories[-1, :]
        p5 = float(np.percentile(final_prices, 5))
        p50 = float(np.percentile(final_prices, 50))
        p95 = float(np.percentile(final_prices, 95))
        prob_bull = float(np.mean(final_prices > s0) * 100)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Prix Actuel ($S_0$)", f"${s0:.2f}")
        m2.metric("Drift Annualisé (μ)", f"{mu_ann * 100:+.2f}%")
        m3.metric("Volatilité (σ)", f"{sigma_ann * 100:.2f}%")
        m4.metric("Médiane à Terme", f"${p50:.2f}", f"{((p50 - s0) / s0) * 100:+.2f}%")
        m5.metric("Probabilité de Gain", f"{prob_bull:.1f}%")
        
        c_sim1, c_sim2 = st.columns([3, 2])
        
        with c_sim1:
            fig_paths = go.Figure()
            time_steps = list(range(forecast_days + 1))
            
            # Échantillon de trajectoires bicolores (vert si final > S0, rouge sinon)
            sample_size = min(n_simulations, 100)
            for j in range(sample_size):
                path = price_trajectories[:, j]
                is_bull = path[-1] >= s0
                path_color = 'rgba(0, 200, 83, 0.12)' if is_bull else 'rgba(213, 0, 0, 0.12)'
                fig_paths.add_trace(go.Scatter(
                    x=time_steps, y=path, mode='lines',
                    line=dict(color=path_color, width=1),
                    showlegend=False, hoverinfo='skip'
                ))
            
            # Lignes directrices quantiles
            fig_paths.add_trace(go.Scatter(
                x=time_steps, y=np.percentile(price_trajectories, 95, axis=1),
                mode='lines', line=dict(color='#00c853', width=1.5, dash='dash'), name='Percentile 95% (Hausse)'
            ))
            fig_paths.add_trace(go.Scatter(
                x=time_steps, y=np.median(price_trajectories, axis=1),
                mode='lines', line=dict(color='#ffab00', width=2), name='Trajectoire Médiane (50%)'
            ))
            fig_paths.add_trace(go.Scatter(
                x=time_steps, y=np.percentile(price_trajectories, 5, axis=1),
                mode='lines', line=dict(color='#d50000', width=1.5, dash='dash'), name='Percentile 5% (Baisse)'
            ))
            
            fig_paths.update_layout(
                template="plotly_dark",
                paper_bgcolor="#080a0f",
                plot_bgcolor="#0e121a",
                xaxis_title="Jours Ouvrés Futurs",
                yaxis_title="Prix ($)",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(14,18,26,0.8)")
            )
            st.plotly_chart(fig_paths, use_container_width=True)
            
        with c_sim2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=final_prices[final_prices >= s0],
                nbinsx=30, marker_color='#00c853', opacity=0.7, name="Trajectoires Haussières"
            ))
            fig_hist.add_trace(go.Histogram(
                x=final_prices[final_prices < s0],
                nbinsx=30, marker_color='#d50000', opacity=0.7, name="Trajectoires Baissières"
            ))
            fig_hist.add_vline(x=s0, line_color="#ffffff", line_dash="dash", line_width=1, annotation_text="Prix Initial")
            fig_hist.add_vline(x=p50, line_color="#ffab00", line_width=2, annotation_text=f"Médiane (${p50:.1f})")
            fig_hist.update_layout(
                barmode='overlay',
                template="plotly_dark",
                paper_bgcolor="#080a0f",
                plot_bgcolor="#0e121a",
                xaxis_title="Prix Terminal ($)",
                yaxis_title="Nombre de scénarios",
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning(f"Données de cours insuffisantes pour {primary_ticker}.")

# -------------------------------------------------------------
# ONGLET 3 : VALUE-AT-RISK (VaR) & CVaR / EXPECTED SHORTFALL
# -------------------------------------------------------------
with tabs[2]:
    st.markdown("#### Métriques Avancées : Value-at-Risk (VaR) & Expected Shortfall (CVaR)")
    st.markdown(r"""
    Ajustement non-gaussien de **Cornish-Fisher** pour séries financières asymétriques et à queues épaisses :
    $$z_c = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$
    $$\text{VaR}_{1-\alpha} = - \left( \mu + z_c \cdot \sigma \right)$$
    $$\text{CVaR}_{1-\alpha} = \mathbb{E}\left[ -R \mid R \le -\text{VaR}_{1-\alpha} \right]$$
    """)
    
    if len(hist_prices) > 30:
        returns = log_ret.values
        confidence_level = st.slider("Niveau de confiance statistique (1 - α)", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
        alpha = 1.0 - confidence_level
        
        # Statistiques d'ordre supérieur
        s = skew(returns)
        k = kurtosis(returns) # Kurtosis excessif
        z_norm = norm.ppf(alpha)
        
        # Formule Cornish-Fisher
        z_cf = (z_norm + 
                (1/6) * (z_norm**2 - 1) * s + 
                (1/24) * (z_norm**3 - 3*z_norm) * k - 
                (1/36) * (2*z_norm**3 - 5*z_norm) * (s**2))
        
        var_cf_daily = -(mu_daily + z_cf * sigma_daily)
        var_hist_daily = -np.percentile(returns, alpha * 100)
        cvar_hist_daily = -returns[returns <= -var_hist_daily].mean()
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Asymétrie (Skewness)", f"{s:.3f}")
        r2.metric("Aplatissement (Excess Kurtosis)", f"{k:.3f}")
        r3.metric(f"VaR {confidence_level*100:.0f}% (1 jour)", f"{var_cf_daily*100:.2f}%", f"-${s0 * var_cf_daily:.2f}")
        r4.metric(f"CVaR {confidence_level*100:.0f}% (Expected Shortfall)", f"{cvar_hist_daily*100:.2f}%", f"-${s0 * cvar_hist_daily:.2f}")
        
        # Distribution empirique vs seuil VaR
        fig_var = go.Figure()
        fig_var.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=80,
            marker_color='#2979ff',
            opacity=0.6,
            name="Rendements Quotidiens"
        ))
        fig_var.add_vline(x=-var_cf_daily * 100, line_color="#d50000", line_width=2, annotation_text=f"VaR CF ({var_cf_daily*100:.2f}%)")
        fig_var.add_vline(x=-cvar_hist_daily * 100, line_color="#ff1744", line_width=2, line_dash="dash", annotation_text=f"CVaR ({cvar_hist_daily*100:.2f}%)")
        fig_var.update_layout(
            template="plotly_dark",
            paper_bgcolor="#080a0f",
            plot_bgcolor="#0e121a",
            xaxis_title="Rendement Journalier (%)",
            yaxis_title="Effectif",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_var, use_container_width=True)

# -------------------------------------------------------------
# ONGLET 4 : CORRÉLATION FACTORIELLE & BENCHMARK
# -------------------------------------------------------------
with tabs[3]:
    st.markdown("#### Matrice et Dynamique de Corrélation")
    
    with st.spinner("Synchronisation des séries temporelles..."):
        s_target = fetch_historical_series(primary_ticker, years=history_years)
        s_bench = fetch_historical_series(macro_ticker, years=history_years)
        df_pair = pd.DataFrame({primary_ticker: s_target, macro_ticker: s_bench}).dropna()
    
    if not df_pair.empty:
        df_rebased = (df_pair / df_pair.iloc[0]) * 100
        returns_pair = df_pair.pct_change().dropna()
        
        corr_levels = df_pair[primary_ticker].corr(df_pair[macro_ticker])
        corr_returns = returns_pair[primary_ticker].corr(returns_pair[macro_ticker])
        
        # Calcul du Bêta de marché
        cov_matrix = np.cov(returns_pair[primary_ticker], returns_pair[macro_ticker])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        b1, b2, b3 = st.columns(3)
        b1.metric("Corrélation des Rendements", f"{corr_returns:.3f}")
        b2.metric("Corrélation des Niveaux de Prix", f"{corr_levels:.3f}")
        b3.metric(f"Bêta par rapport à {macro_ticker}", f"{beta:.2f}")
        
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=df_rebased.index, y=df_rebased[primary_ticker],
            name=f"{primary_ticker} (Base 100)", line=dict(color='#00c853', width=1.5)
        ))
        fig_corr.add_trace(go.Scatter(
            x=df_rebased.index, y=df_rebased[macro_ticker],
            name=f"{macro_ticker} (Base 100)", line=dict(color='#ffab00', width=1.5)
        ))
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="#080a0f",
            plot_bgcolor="#0e121a",
            xaxis_title="Date",
            yaxis_title="Base 100",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
