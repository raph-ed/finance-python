import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, t as student_t, skew, kurtosis
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="QuantTerminal Pro | Valuation & Stochastic Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GESTION DE LA NAVIGATION (PAGE D'ACCUEIL vs TERMINAL) ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "NVDA"
if "selected_peers" not in st.session_state:
    st.session_state.selected_peers = ["MSFT", "GOOGL", "AMZN", "AAPL", "META"]

def go_to_terminal(ticker=None, peers=None):
    if ticker:
        st.session_state.selected_ticker = ticker
    if peers:
        st.session_state.selected_peers = peers
    st.session_state.page = "terminal"

def go_to_home():
    st.session_state.page = "home"

# --- THÈME GRAPHIQUE INSTITUTIONNEL (CYAN / ÉMERAUDE / GLASSMORPHISM) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1b2a 0%, #070a10 55%, #05070b 100%);
        color: #e2e8f0;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid rgba(0, 229, 255, 0.14);
    }
    
    /* Cartes KPI Glassmorphism */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(16, 24, 39, 0.85), rgba(11, 15, 25, 0.95));
        border: 1px solid rgba(0, 229, 255, 0.18);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
        padding: 16px 20px;
        border-radius: 10px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(0, 229, 255, 0.45);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Onglets stylisés */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        color: #94a3b8;
        background-color: transparent;
        border: none;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 18px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00e5ff !important;
        background: rgba(0, 229, 255, 0.12) !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
    }
    
    /* Hero Page d'Accueil */
    .hero-card {
        background: linear-gradient(135deg, rgba(14, 24, 42, 0.9) 0%, rgba(9, 14, 24, 0.95) 100%);
        border: 1px solid rgba(0, 229, 255, 0.28);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }
    
    .feature-card {
        background: rgba(15, 22, 36, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px;
        height: 100%;
    }
    
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 230, 118, 0.12);
        border: 1px solid rgba(0, 230, 118, 0.4);
        color: #00e676;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
""", unsafe_allow_html=True)

# --- DICTIONNAIRE DE PROXY POUR LES FUTURES & INDICES (ÉVITE LES "None") ---
INDEX_PROXY_MAP = {
    "NQ=F": ("QQQ", "Invesco QQQ (Proxy Nasdaq 100)"),
    "^NDX": ("QQQ", "Invesco QQQ (Proxy Nasdaq 100)"),
    "^IXIC": ("QQQ", "Invesco QQQ (Proxy Nasdaq)"),
    "ES=F": ("SPY", "SPDR S&P 500 ETF (Proxy S&P 500)"),
    "^GSPC": ("SPY", "SPDR S&P 500 ETF (Proxy S&P 500)"),
    "YM=F": ("DIA", "SPDR Dow Jones ETF (Proxy Dow 30)"),
    "RTY=F": ("IWM", "iShares Russell 2000 ETF")
}

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "AMD",
    "JPM", "GS", "MS", "BLK", "V", "MA", "LLY", "UNH", "JNJ", "XOM", "CVX",
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "BNP.PA", "SPY", "QQQ", "IWM", "DIA"
]

# --- MOTEUR D'EXTRACTION TEMPS RÉEL (GARANTIT LE 3 OU 4 NOVEMBRE) ---
@st.cache_data(ttl=300, show_spinner=False)
def fetch_historical_series(ticker: str, years: int = 5):
    """
    Récupère l'historique complet ET injecte la dernière cotation disponible
    (clôture de la veille 3 nov ou cours intraday du jour 4 nov).
    Retourne: (pd.Series des prix, datetime de la dernière cotation, bool is_live_today)
    """
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=years * 365 + 15)
    # end_date à J+1 car yfinance exclut la borne supérieure 'end'
    end_date = now + datetime.timedelta(days=1)
    
    try:
        t_obj = yf.Ticker(ticker)
        data = t_obj.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=True)
        
        if data.empty or 'Close' not in data.columns:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty:
                return pd.Series(dtype=float), None, False
            prices = data['Close'].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data['Close']
        else:
            prices = data['Close']
            
        prices = prices.dropna()
        if prices.empty:
            return pd.Series(dtype=float), None, False

        # Normalisation de l'index temporel (sans fuseau horaire pour éviter les conflits)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        
        # Vérification du cours en direct (intraday du jour même, ex: 4 novembre)
        is_live = False
        try:
            intraday = t_obj.history(period="5d", interval="15m")
            if not intraday.empty and 'Close' in intraday.columns:
                last_intra_val = float(intraday['Close'].dropna().iloc[-1])
                last_intra_ts = pd.to_datetime(intraday.index[-1]).tz_localize(None)
                last_daily_date = prices.index[-1].date()
                
                if last_intra_ts.date() > last_daily_date:
                    # Ajoute le point du jour (ex: 4 novembre) s'il n'est pas encore dans les bougies daily
                    new_point = pd.Series([last_intra_val], index=[pd.Timestamp(last_intra_ts.date())])
                    prices = pd.concat([prices, new_point])
                    is_live = True
                elif last_intra_ts.date() == last_daily_date:
                    # Met à jour la bougie d'aujourd'hui avec la toute dernière minute cotée
                    prices.iloc[-1] = last_intra_val
                    is_live = (last_intra_ts.date() == now.date())
        except Exception:
            pass

        last_dt = prices.index[-1]
        return pd.Series(prices.values.ravel(), index=prices.index, name=ticker), last_dt, is_live
    except Exception:
        return pd.Series(dtype=float), None, False

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_snapshot(tickers: tuple):
    """Récupère les cours en direct pour le bandeau de la page d'accueil."""
    snapshots = []
    for sym in tickers:
        s, last_dt, _ = fetch_historical_series(sym, years=1)
        if len(s) >= 2:
            last_p = float(s.iloc[-1])
            prev_p = float(s.iloc[-2])
            chg = ((last_p - prev_p) / prev_p) * 100
            snapshots.append({
                "symbol": sym,
                "price": last_p,
                "change": chg,
                "date": last_dt.strftime("%d %b %Y") if last_dt else "N/A"
            })
    return snapshots

@st.cache_data(ttl=900, show_spinner=False)
def search_securities(query: str):
    if not query or len(query.strip()) < 2:
        return []
    try:
        s = yf.Search(query, max_results=8)
        results = []
        for q in s.quotes:
            symbol = q.get('symbol', '')
            name = q.get('shortname') or q.get('longname') or symbol
            qtype = q.get('quoteType', 'EQUITY')
            exch = q.get('exchange', '')
            results.append((symbol, f"{symbol} — {name} [{exch} | {qtype}]"))
        return results
    except Exception:
        return []

@st.cache_data(ttl=900, show_spinner=False)
def fetch_fundamentals(tickers: tuple):
    metrics = []
    for t in tickers:
        # Si l'utilisateur choisit un Future comme NQ=F, on utilise aussi un proxy pour les multiples
        effective_t = INDEX_PROXY_MAP[t][0] if t in INDEX_PROXY_MAP else t
        display_label = f"{t} (via {effective_t})" if t in INDEX_PROXY_MAP else t
        try:
            info = yf.Ticker(effective_t).info
            metrics.append({
                'Ticker': t,
                'Affichage': display_label,
                'Nom': info.get('shortName') or info.get('longName') or display_label,
                'Prix ($)': info.get('currentPrice') or info.get('regularMarketPrice') or np.nan,
                'Capitalisation ($B)': (info.get('marketCap') or np.nan) / 1e9,
                'P/E (Trailing)': info.get('trailingPE', np.nan),
                'Forward P/E': info.get('forwardPE', np.nan),
                'EV/EBITDA': info.get('enterpriseToEbitda', np.nan),
                'P/S': info.get('priceToSalesTrailing12Months', np.nan),
                'P/B': info.get('priceToBook', np.nan)
            })
        except Exception:
            metrics.append({'Ticker': t, 'Affichage': display_label, 'Nom': t})
    return pd.DataFrame(metrics)


# =====================================================================
# ÉCRAN 1 : PAGE D'ACCUEIL (LANDING PAGE INSTITUTIONNELLE)
# =====================================================================
if st.session_state.page == "home":
    now_str = datetime.datetime.now().strftime("%d %B %Y — %H:%M")
    
    st.markdown(f"""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <span style="color: #00e5ff; font-family: 'JetBrains Mono'; font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase;">
                QUANTVALUATION & STOCHASTIC RISK SUITE v2.5
            </span>
            <span class="live-badge">● FLUX MARCHÉ SYNCHRONISÉ : {now_str}</span>
        </div>
        <h1 style="font-size: 2.5rem; font-weight: 800; color: #ffffff; margin: 14px 0 10px 0;">
            Terminal Quantitatif & Modélisation Stochastique
        </h1>
        <p style="font-size: 1.05rem; color: #94a3b8; max-width: 820px; line-height: 1.6;">
            Évaluez la valorisation fondamentale relative, simulez des milliers de trajectoires Monte Carlo 
            (Mouvement Brownien Géométrique & queues épaisses t-Student) et mesurez le risque extrême (VaR Cornish-Fisher & CVaR) 
            avec les données de marché actualisées jusqu'à la séance en cours.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- BANDEAU MARCHÉ EN DIRECT (CLÔTURE 3 NOV / LIVE 4 NOV) ---
    st.markdown("##### Aperçu Temps Réel des Indices & Leaders")
    with st.spinner("Synchronisation des dernières cotations (3-4 Novembre)..."):
        market_snap = fetch_live_snapshot(("SPY", "QQQ", "NVDA", "MSFT", "AAPL", "^GSPC"))
    
    if market_snap:
        cols_snap = st.columns(len(market_snap))
        for idx, item in enumerate(market_snap):
            cols_snap[idx].metric(
                label=f"{item['symbol']} ({item['date']})",
                value=f"${item['price']:,.2f}",
                delta=f"{item['change']:+.2f}%"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SÉLECTEUR DE LANCEMENT RAPIDE ---
    col_launch, col_presets = st.columns([3, 2])
    
    with col_launch:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin-top:0; color:#ffffff;">🚀 Lancer une analyse sur mesure</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
                Recherchez un actif (action US/EU, ETF ou contrat Future) pour ouvrir directement le terminal quantitatif.
            </p>
        """, unsafe_allow_html=True)
        
        home_query = st.text_input("Rechercher un actif ou symbole :", value=st.session_state.selected_ticker)
        home_results = search_securities(home_query)
        if home_results:
            chosen = st.selectbox("Sélectionnez la valeur exacte :", options=home_results, format_func=lambda x: x[1])
            target_sym = chosen[0]
        else:
            target_sym = home_query.upper().strip() or "NVDA"
            
        if st.button("Ouvrir le Terminal d'Analyse →", type="primary", use_container_width=True):
            go_to_terminal(ticker=target_sym)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_presets:
        st.markdown("""
        <div class="feature-card">
            <h4 style="margin-top:0; color:#ffffff;">⚡ Portefeuilles & Univers Préconfigurés</h4>
            <p style="color:#94a3b8; font-size:0.9rem;">
                Chargez instantanément un groupe sectoriel complet avec ses comparables :
            </p>
        """, unsafe_allow_html=True)
        
        if st.button("🤖 AI & Semi-conducteurs (NVDA vs MSFT, GOOGL, AMD, AVGO)", use_container_width=True):
            go_to_terminal("NVDA", ["MSFT", "GOOGL", "AMD", "AVGO", "META"])
            st.rerun()
        if st.button("📈 Indices & Contrats Futures US (NQ=F / QQQ vs SPY, DIA, IWM)", use_container_width=True):
            go_to_terminal("NQ=F", ["QQQ", "SPY", "DIA", "IWM"])
            st.rerun()
        if st.button("🏦 Banques d'Investissement Wall Street (JPM vs GS, MS, BLK)", use_container_width=True):
            go_to_terminal("JPM", ["GS", "MS", "BLK", "V"])
            st.rerun()
        if st.button("🇫🇷 Champions CAC 40 Paris (MC.PA vs OR.PA, AIR.PA, TTE.PA, BNP.PA)", use_container_width=True):
            go_to_terminal("MC.PA", ["OR.PA", "AIR.PA", "TTE.PA", "BNP.PA"])
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 4 PILIERS MÉTHODOLOGIQUES ---
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <div style="color:#00e5ff; font-weight:700; margin-bottom:6px;">01. Multiples & Proxy</div>
            <div style="font-size:0.88rem; color:#94a3b8;">
                Comparaison simultanée du <b>P/E Trailing</b> et du <b>Forward P/E</b> avec substitution automatique de proxy ETF pour les contrats Futures.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <div style="color:#00e676; font-weight:700; margin-bottom:6px;">02. Cônes Monte Carlo</div>
            <div style="font-size:0.88rem; color:#94a3b8;">
                Simulation vectorisée de 10 000 trajectoires par Mouvement Brownien Géométrique ou loi <b>t-Student</b> (queues épaisses) sur calendrier réel.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <div style="color:#ffab00; font-weight:700; margin-bottom:6px;">03. VaR Cornish-Fisher</div>
            <div style="font-size:0.88rem; color:#94a3b8;">
                Intégration de l'asymétrie (Skewness) et du Kurtosis excessif pour mesurer fidèlement le risque de perte extrême (CVaR).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="feature-card">
            <div style="color:#2979ff; font-weight:700; margin-bottom:6px;">04. Bêta & Corrélation</div>
            <div style="font-size:0.88rem; color:#94a3b8;">
                Synchronisation en Base 100, corrélation glissante sur 60 jours et régression linéaire du Bêta face au S&P 500.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# ÉCRAN 2 : TERMINAL QUANTITATIF PRINCIPAL
# =====================================================================
else:
    # --- BARRE LATÉRALE DE CONTRÔLE ---
    if st.sidebar.button("⬅ Retour à la Page d'Accueil", use_container_width=True):
        go_to_home()
        st.rerun()
        
    if st.sidebar.button("🔄 Actualiser le flux temps réel", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("### 1. Sélection de l'Actif")
    search_term = st.sidebar.text_input(
        "Recherche d'actif (ex: NVDA, NQ=F, AAPL)",
        value=st.session_state.selected_ticker
    )
    search_results = search_securities(search_term)

    if search_results:
        selected_tuple = st.sidebar.selectbox(
            "Résultats disponibles :",
            options=search_results,
            format_func=lambda x: x[1]
        )
        primary_ticker = selected_tuple[0]
    else:
        primary_ticker = search_term.upper().strip() if search_term else "NVDA"

    st.session_state.selected_ticker = primary_ticker

    # Sélecteur par Tags (Chips) au lieu du champ texte brut
    available_options = sorted(list(set(DEFAULT_UNIVERSE + st.session_state.selected_peers + [primary_ticker])))
    default_peers = [p for p in st.session_state.selected_peers if p != primary_ticker]
    
    peer_list = st.sidebar.multiselect(
        "Actifs comparables (Tags interactifs) :",
        options=[x for x in available_options if x != primary_ticker],
        default=default_peers[:6]
    )
    custom_peer = st.sidebar.text_input("Ajouter un ticker personnalisé :", placeholder="Ex: ASML, CRM...")
    if custom_peer.strip():
        for cp in custom_peer.split(","):
            clean_cp = cp.strip().upper()
            if clean_cp and clean_cp not in peer_list and clean_cp != primary_ticker:
                peer_list.append(clean_cp)

    with st.sidebar.expander("⚙️ Modélisation Stochastique", expanded=True):
        stoch_model = st.selectbox(
            "Loi des chocs aléatoires",
            ["GBM Classique (Loi Normale)", "Queues Épaisses (t-Student, df=5)"]
        )
        n_simulations = st.slider("Trajectoires Monte Carlo", 500, 10000, 2500, 500)
        forecast_days = st.slider("Horizon (jours ouvrés)", 21, 504, 252, 21)
        history_years = st.selectbox("Profondeur historique (années)", [1, 3, 5, 10], index=2)

    with st.sidebar.expander("🌍 Facteur Macro / Benchmark", expanded=False):
        macro_ticker = st.text_input("Symbole de référence", value="^GSPC")

    # --- CHARGEMENT DES DONNÉES DE COURS JUSQU'AU 3 / 4 NOVEMBRE ---
    with st.spinner(f"Synchronisation des séries temporelles pour {primary_ticker}..."):
        hist_prices, last_quote_dt, is_live_quote = fetch_historical_series(primary_ticker, years=history_years)
        all_tickers = tuple([primary_ticker] + [p for p in peer_list if p != primary_ticker])
        df_metrics = fetch_fundamentals(all_tickers)

    # Formatage de la date de la dernière donnée appliquée au modèle
    if last_quote_dt is not None:
        date_str = last_quote_dt.strftime("%d %B %Y")
        status_label = f"● DERNIÈRE DONNÉE DU MODÈLE : {date_str} ({'LIVE / INTRADAY' if is_live_quote else 'CLÔTURE OFFICIELLE'})"
    else:
        status_label = "● DONNÉES INDISPONIBLES"

    # --- EN-TÊTE DU TERMINAL ---
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid rgba(0,229,255,0.2); padding-bottom:16px; margin-bottom:20px; flex-wrap:wrap; gap:12px;">
        <div>
            <div style="font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; color: #00e5ff; font-family:'JetBrains Mono';">
                PLATEFORME QUANTITATIVE INSTITUTIONNELLE
            </div>
            <div style="font-size: 1.9rem; font-weight: 800; color: #ffffff; margin-top: 2px;">
                ANALYSE D'ACTIF : {primary_ticker}
            </div>
        </div>
        <div class="live-badge">{status_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- BANDEAU DE 4 KPIs GLOBAUX EN HAUT DE PAGE ---
    if len(hist_prices) > 30:
        s0 = float(hist_prices.iloc[-1])
        s_prev = float(hist_prices.iloc[-2])
        daily_chg = ((s0 - s_prev) / s_prev) * 100
        
        log_ret = np.log(hist_prices / hist_prices.shift(1)).dropna()
        mu_daily = float(log_ret.mean())
        sigma_daily = float(log_ret.std())
        sigma_ann = sigma_daily * np.sqrt(252)
        
        # VaR 95% Cornish-Fisher rapide pour le bandeau supérieur
        s_val = float(skew(log_ret.values))
        k_val = float(kurtosis(log_ret.values))
        z95 = norm.ppf(0.05)
        z_cf95 = z95 + (1/6)*(z95**2 - 1)*s_val + (1/24)*(z95**3 - 3*z95)*k_val - (1/36)*(2*z95**3 - 5*z95)*(s_val**2)
        var95_pct = -(mu_daily + z_cf95 * sigma_daily) * 100

        pe_median = df_metrics['P/E (Trailing)'].median() if 'P/E (Trailing)' in df_metrics.columns else np.nan
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            f"Cours Actuel ({last_quote_dt.strftime('%d/%m/%Y')})",
            f"${s0:,.2f}",
            f"{daily_chg:+.2f}% (24h)"
        )
        k2.metric(
            "P/E Médian du Groupe",
            f"{pe_median:.1f}x" if pd.notnull(pe_median) else "N/A",
            f"{len(all_tickers)} actifs comparés"
        )
        k3.metric(
            "Volatilité Annualisée (σ)",
            f"{sigma_ann * 100:.2f}%",
            f"Historique {history_years} ans"
        )
        k4.metric(
            "VaR 95% (Cornish-Fisher 1j)",
            f"-{var95_pct:.2f}%",
            f"-${s0 * (var95_pct/100):,.2f} / action",
            delta_color="inverse"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "1. Multiples & Valorisation",
        "2. Mouvement Brownien & Monte Carlo",
        "3. Gestion des Risques (VaR & CVaR)",
        "4. Analyse Factorielle & Corrélations"
    ])

    # -------------------------------------------------------------
    # ONGLET 1 : VALORISATION PAR MULTIPLES (AVEC PROXY & DOUBLE BARRES)
    # -------------------------------------------------------------
    with tabs[0]:
        if primary_ticker in INDEX_PROXY_MAP:
            proxy_sym, proxy_desc = INDEX_PROXY_MAP[primary_ticker]
            st.info(f"💡 **{primary_ticker}** est un contrat à terme ou un indice sans bilan comptable direct. Les multiples fondamentaux ci-dessous sont automatiquement calculés via son ETF de réplication **{proxy_desc}** afin d'éviter les valeurs `None`.")

        col_tbl, col_chart = st.columns([1.15, 1])

        with col_tbl:
            st.markdown("#### Tableau Comparatif Sectoriel")
            st.dataframe(
                df_metrics[['Affichage', 'Nom', 'Capitalisation ($B)', 'P/E (Trailing)', 'Forward P/E', 'EV/EBITDA']].set_index('Affichage'),
                column_config={
                    "Nom": st.column_config.TextColumn("Entreprise / Fonds"),
                    "Capitalisation ($B)": st.column_config.NumberColumn("Cap. ($B)", format="$%.1fB"),
                    "P/E (Trailing)": st.column_config.NumberColumn("P/E Trailing", format="%.1fx"),
                    "Forward P/E": st.column_config.NumberColumn("Forward P/E", format="%.1fx"),
                    "EV/EBITDA": st.column_config.NumberColumn("EV/EBITDA", format="%.1fx"),
                },
                use_container_width=True
            )

        with col_chart:
            st.markdown("#### Comparaison Multiples & Médiane")
            chart_mode = st.radio(
                "Mode d'affichage :",
                ["Comparaison P/E Trailing vs Forward P/E", "Multiple Unique"],
                horizontal=True
            )

            fig_bar = go.Figure()
            if chart_mode == "Comparaison P/E Trailing vs Forward P/E":
                df_plot = df_metrics.dropna(subset=['P/E (Trailing)', 'Forward P/E'], how='all').copy()
                if not df_plot.empty:
                    med_pe = df_plot['P/E (Trailing)'].median()
                    fig_bar.add_trace(go.Bar(
                        x=df_plot['Ticker'],
                        y=df_plot['P/E (Trailing)'],
                        name="P/E Trailing",
                        marker_color="#00e5ff",
                        text=df_plot['P/E (Trailing)'].apply(lambda x: f"{x:.1f}x" if pd.notnull(x) else ""),
                        textposition="outside"
                    ))
                    fig_bar.add_trace(go.Bar(
                        x=df_plot['Ticker'],
                        y=df_plot['Forward P/E'],
                        name="Forward P/E",
                        marker_color="#00e676",
                        text=df_plot['Forward P/E'].apply(lambda x: f"{x:.1f}x" if pd.notnull(x) else ""),
                        textposition="outside"
                    ))
                    if pd.notnull(med_pe):
                        fig_bar.add_hline(
                            y=med_pe,
                            line=dict(color="#ffab00", dash="dash", width=1.8),
                            annotation_text=f"Médiane P/E : {med_pe:.1f}x",
                            annotation_position="top right"
                        )
            else:
                selected_ratio = st.selectbox("Multiple à analyser :", ["P/E (Trailing)", "Forward P/E", "EV/EBITDA", "P/S", "P/B"])
                df_plot = df_metrics.dropna(subset=[selected_ratio]).copy()
                if not df_plot.empty:
                    med_val = df_plot[selected_ratio].median()
                    colors = ['#00e676' if t == primary_ticker else '#00e5ff' for t in df_plot['Ticker']]
                    fig_bar.add_trace(go.Bar(
                        x=df_plot['Ticker'],
                        y=df_plot[selected_ratio],
                        marker_color=colors,
                        text=df_plot[selected_ratio].apply(lambda x: f"{x:.1f}x"),
                        textposition="outside",
                        name=selected_ratio
                    ))
                    fig_bar.add_hline(
                        y=med_val,
                        line=dict(color="#ffab00", dash="dash", width=1.8),
                        annotation_text=f"Médiane : {med_val:.1f}x"
                    )

            fig_bar.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,18,26,0.7)",
                margin=dict(l=20, r=20, t=35, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------------------------------------------
    # ONGLET 2 : MONTE CARLO AVEC CÔNES DE PROBABILITÉ & VRAIES DATES
    # -------------------------------------------------------------
    with tabs[1]:
        st.markdown("#### Simulation Monte Carlo & Cônes de Probabilité (Fan Chart)")
        
        if len(hist_prices) > 30:
            mu_ann = mu_daily * 252
            
            # Génération des chocs selon le modèle choisi
            if "t-Student" in stoch_model:
                df_t = 5
                raw_shocks = student_t.rvs(df_t, size=(forecast_days, n_simulations))
                shocks = raw_shocks / np.sqrt(df_t / (df_t - 2))
            else:
                shocks = np.random.normal(0, 1, (forecast_days, n_simulations))
                
            drift = (mu_daily - 0.5 * (sigma_daily ** 2))
            daily_returns_sim = drift + sigma_daily * shocks
            price_trajectories = s0 * np.exp(np.vstack([np.zeros((1, n_simulations)), np.cumsum(daily_returns_sim, axis=0)]))
            
            # Axe temporel réel démarrant exactement à la dernière date connue (3 ou 4 novembre 2026)
            future_dates = pd.bdate_range(start=last_quote_dt, periods=forecast_days + 1)
            
            final_prices = price_trajectories[-1, :]
            p5_path = np.percentile(price_trajectories, 5, axis=1)
            p25_path = np.percentile(price_trajectories, 25, axis=1)
            p50_path = np.percentile(price_trajectories, 50, axis=1)
            p75_path = np.percentile(price_trajectories, 75, axis=1)
            p95_path = np.percentile(price_trajectories, 95, axis=1)
            
            p50_end = float(p50_path[-1])
            prob_bull = float(np.mean(final_prices > s0) * 100)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Départ S₀ ({last_quote_dt.strftime('%d/%m/%Y')})", f"${s0:,.2f}")
            m2.metric("Drift Annualisé (μ)", f"{mu_ann * 100:+.2f}%")
            m3.metric(f"Médiane au {future_dates[-1].strftime('%d/%m/%Y')}", f"${p50_end:,.2f}", f"{((p50_end - s0) / s0) * 100:+.2f}%")
            m4.metric("Probabilité Haussière", f"{prob_bull:.1f}%")

            c_sim1, c_sim2 = st.columns([3, 1.8])
            with c_sim1:
                fig_paths = go.Figure()
                
                # Cône 5% - 95%
                fig_paths.add_trace(go.Scatter(
                    x=future_dates, y=p95_path, mode='lines',
                    line=dict(color='rgba(0, 229, 255, 0.45)', width=1, dash='dot'),
                    name='Percentile 95%'
                ))
                fig_paths.add_trace(go.Scatter(
                    x=future_dates, y=p5_path, mode='lines',
                    fill='tonexty', fillcolor='rgba(0, 229, 255, 0.08)',
                    line=dict(color='rgba(255, 82, 82, 0.45)', width=1, dash='dot'),
                    name='Intervalle 5% - 95%'
                ))
                
                # Cône 25% - 75% (Interquartile)
                fig_paths.add_trace(go.Scatter(
                    x=future_dates, y=p75_path, mode='lines',
                    line=dict(color='rgba(0, 230, 118, 0.5)', width=1),
                    name='Percentile 75%'
                ))
                fig_paths.add_trace(go.Scatter(
                    x=future_dates, y=p25_path, mode='lines',
                    fill='tonexty', fillcolor='rgba(0, 230, 118, 0.16)',
                    line=dict(color='rgba(0, 230, 118, 0.5)', width=1),
                    name='Cœur 25% - 75%'
                ))

                # Quelques trajectoires fines en arrière-plan
                for j in range(min(n_simulations, 35)):
                    fig_paths.add_trace(go.Scatter(
                        x=future_dates, y=price_trajectories[:, j], mode='lines',
                        line=dict(color='rgba(255, 255, 255, 0.07)', width=0.8),
                        showlegend=False, hoverinfo='skip'
                    ))

                # Médiane centrale
                fig_paths.add_trace(go.Scatter(
                    x=future_dates, y=p50_path, mode='lines',
                    line=dict(color='#00e5ff', width=2.8),
                    name='Trajectoire Médiane (50%)'
                ))

                fig_paths.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14,18,26,0.7)",
                    xaxis_title="Calendrier de Projection (Jours Ouvrés)",
                    yaxis_title="Prix Projeté ($)",
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(9,13,22,0.8)")
                )
                st.plotly_chart(fig_paths, use_container_width=True)

            with c_sim2:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=final_prices[final_prices >= s0], nbinsx=35,
                    marker_color='#00e676', opacity=0.75, name="Gain (> S₀)"
                ))
                fig_hist.add_trace(go.Histogram(
                    x=final_prices[final_prices < s0], nbinsx=35,
                    marker_color='#ff5252', opacity=0.75, name="Perte (< S₀)"
                ))
                fig_hist.add_vline(x=s0, line_color="#ffffff", line_dash="dash", annotation_text="S₀")
                fig_hist.add_vline(x=p50_end, line_color="#00e5ff", line_width=2, annotation_text="Médiane")
                fig_hist.update_layout(
                    barmode='overlay',
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14,18,26,0.7)",
                    xaxis_title="Distribution des Prix à Terme ($)",
                    yaxis_title="Fréquence",
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------------------------------------
    # ONGLET 3 : VALUE-AT-RISK (VaR) & CVaR
    # -------------------------------------------------------------
    with tabs[2]:
        st.markdown("#### Value-at-Risk (Cornish-Fisher) & Expected Shortfall (CVaR)")
        if len(hist_prices) > 30:
            returns = log_ret.values
            confidence_level = st.slider("Niveau de confiance (1 - α)", 0.90, 0.99, 0.95, 0.01)
            alpha = 1.0 - confidence_level
            
            s = float(skew(returns))
            k = float(kurtosis(returns))
            z_norm = norm.ppf(alpha)
            z_cf = (z_norm + (1/6)*(z_norm**2 - 1)*s + (1/24)*(z_norm**3 - 3*z_norm)*k - (1/36)*(2*z_norm**3 - 5*z_norm)*(s**2))
            
            var_cf_daily = -(mu_daily + z_cf * sigma_daily)
            var_hist_daily = -np.percentile(returns, alpha * 100)
            cvar_hist_daily = -returns[returns <= -var_hist_daily].mean()
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Asymétrie (Skewness)", f"{s:.3f}")
            r2.metric("Kurtosis Excessif", f"{k:.3f}")
            r3.metric(f"VaR Cornish-Fisher ({confidence_level*100:.0f}%)", f"-{var_cf_daily*100:.2f}%", f"-${s0 * var_cf_daily:.2f}")
            r4.metric(f"CVaR / Expected Shortfall", f"-{cvar_hist_daily*100:.2f}%", f"-${s0 * cvar_hist_daily:.2f}")
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(
                x=returns * 100, nbinsx=80,
                marker_color='#00e5ff', opacity=0.65, name="Rendements Journaliers"
            ))
            fig_var.add_vline(x=-var_cf_daily * 100, line_color="#ffab00", line_width=2, annotation_text=f"VaR CF (-{var_cf_daily*100:.2f}%)")
            fig_var.add_vline(x=-cvar_hist_daily * 100, line_color="#ff5252", line_width=2, line_dash="dash", annotation_text=f"CVaR (-{cvar_hist_daily*100:.2f}%)")
            fig_var.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,18,26,0.7)",
                xaxis_title="Rendement Journalier (%)",
                yaxis_title="Occurrences",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_var, use_container_width=True)

    # -------------------------------------------------------------
    # ONGLET 4 : CORRÉLATION FACTORIELLE & BENCHMARK
    # -------------------------------------------------------------
    with tabs[3]:
        st.markdown("#### Dynamique de Corrélation & Bêta de Marché")
        s_bench, _, _ = fetch_historical_series(macro_ticker, years=history_years)
        df_pair = pd.DataFrame({primary_ticker: hist_prices, macro_ticker: s_bench}).dropna()
        
        if len(df_pair) > 30:
            df_rebased = (df_pair / df_pair.iloc[0]) * 100
            returns_pair = df_pair.pct_change().dropna()
            
            corr_returns = returns_pair[primary_ticker].corr(returns_pair[macro_ticker])
            cov_matrix = np.cov(returns_pair[primary_ticker], returns_pair[macro_ticker])
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            rolling_corr = returns_pair[primary_ticker].rolling(60).corr(returns_pair[macro_ticker])
            
            b1, b2, b3 = st.columns(3)
            b1.metric("Corrélation des Rendements (Pearson)", f"{corr_returns:.3f}")
            b2.metric(f"Bêta Sensibilité ({macro_ticker})", f"{beta:.2f}")
            b3.metric("Dernière Date Synchronisée", df_pair.index[-1].strftime("%d/%m/%Y"))
            
            c_cor1, c_cor2 = st.columns(2)
            with c_cor1:
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(
                    x=df_rebased.index, y=df_rebased[primary_ticker],
                    name=f"{primary_ticker} (Base 100)", line=dict(color='#00e5ff', width=2)
                ))
                fig_corr.add_trace(go.Scatter(
                    x=df_rebased.index, y=df_rebased[macro_ticker],
                    name=f"{macro_ticker} (Base 100)", line=dict(color='#00e676', width=2)
                ))
                fig_corr.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14,18,26,0.7)",
                    title="Performance Comparée (Base 100)",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
            with c_cor2:
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(
                    x=rolling_corr.index, y=rolling_corr,
                    mode='lines', fill='tozeroy',
                    fillcolor='rgba(0, 229, 255, 0.12)',
                    line=dict(color='#00e5ff', width=1.8),
                    name="Corrélation Glissante 60j"
                ))
                fig_roll.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(14,18,26,0.7)",
                    title="Corrélation Glissante (Fenêtre de 60 Jours Ouvrés)",
                    yaxis=dict(range=[-1, 1]),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_roll, use_container_width=True)
