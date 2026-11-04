import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, t as student_t, skew, kurtosis
import datetime

# Configuration page
st.set_page_config(
    page_title="QuantTerminal Pro | Valuation & Stochastic Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Etat session
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

# Styles CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #00e5ff !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 20% 10%, #0e1e38 0%, #080d1a 45%, #05070e 100%);
        color: #e2e8f0;
    }
    
    /* Barre laterale */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1324 0%, #070b14 55%, #050811 100%) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.25);
        box-shadow: 6px 0 28px rgba(0, 0, 0, 0.6);
    }

    .sidebar-brand {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.14), rgba(0, 230, 118, 0.08));
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 16px;
        text-align: center;
    }

    .sidebar-section {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #00e5ff;
        padding: 8px 12px;
        margin: 18px 0 10px 0;
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.16) 0%, rgba(124, 77, 255, 0.06) 100%);
        border-left: 3px solid #00e676;
        border-radius: 0 6px 6px 0;
    }

    /* Champs saisie barre laterale */
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: rgba(14, 23, 42, 0.9) !important;
        border: 1px solid rgba(0, 229, 255, 0.28) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        transition: all 0.2s ease;
    }

    section[data-testid="stSidebar"] div[data-baseweb="input"] > div:focus-within,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
        border-color: #00e676 !important;
        box-shadow: 0 0 12px rgba(0, 230, 118, 0.3) !important;
    }

    section[data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.16) 0%, rgba(0, 230, 118, 0.12) 100%) !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 0.5rem 0.9rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        border-color: #00e676 !important;
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.28) 0%, rgba(0, 229, 255, 0.24) 100%) !important;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.3) !important;
        color: #ffffff !important;
    }

    /* Tags multiselect */
    span[data-baseweb="tag"] {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.22), rgba(0, 229, 255, 0.22)) !important;
        border: 1px solid #00e676 !important;
        color: #00e676 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    span[data-baseweb="tag"] span {
        color: #e2f8ee !important;
    }

    /* Curseurs */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #00e5ff !important;
        border: 2px solid #00e676 !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.9) !important;
    }
    div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {
        color: #94a3b8 !important;
    }

    /* Onglets navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(11, 18, 32, 0.9);
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0, 229, 255, 0.22);
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 999px !important;
        color: #94a3b8;
        background-color: transparent;
        border: none !important;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.3), rgba(0, 229, 255, 0.3)) !important;
        border: 1px solid #00e5ff !important;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.3);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Cartes KPI */
    .kpi-card {
        background: linear-gradient(145deg, rgba(16, 26, 46, 0.92), rgba(9, 14, 26, 0.96));
        border: 1px solid rgba(0, 229, 255, 0.26);
        border-top: 2px solid #00e5ff;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
        padding: 18px 22px;
        border-radius: 14px;
        height: 100%;
    }
    .kpi-card-alt {
        border-top: 2px solid #00e676;
    }
    .kpi-card-warn {
        border-top: 2px solid #ffab00;
    }
    .kpi-card-risk {
        border-top: 2px solid #ff5252;
    }
    .kpi-title {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .kpi-sub-pos {
        font-size: 0.82rem;
        color: #00e676;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub-neg {
        font-size: 0.82rem;
        color: #ff5252;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub-neutral {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Panneaux principaux */
    .glass-panel {
        background: linear-gradient(160deg, rgba(13, 21, 38, 0.88), rgba(8, 13, 24, 0.94));
        border: 1px solid rgba(0, 229, 255, 0.28);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        margin-bottom: 16px;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(15, 28, 52, 0.94) 0%, rgba(9, 15, 28, 0.98) 100%);
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-left: 4px solid #00e676;
        border-radius: 16px;
        padding: 34px 38px;
        margin-bottom: 24px;
    }

    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 230, 118, 0.14);
        border: 1px solid rgba(0, 230, 118, 0.55);
        color: #00e676;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
""", unsafe_allow_html=True)

# Donnees secours
INSTITUTIONAL_FALLBACK_DB = {
    "NVDA":  {"name": "NVIDIA Corporation",      "shares_b": 24.30, "pe": 38.4, "f_pe": 27.5, "ev_ebitda": 32.1, "ps": 21.8, "pb": 36.5},
    "MSFT":  {"name": "Microsoft Corporation",   "shares_b": 7.43,  "pe": 31.2, "f_pe": 26.4, "ev_ebitda": 22.8, "ps": 12.4, "pb": 11.2},
    "GOOGL": {"name": "Alphabet Inc.",           "shares_b": 12.25, "pe": 22.4, "f_pe": 19.1, "ev_ebitda": 15.6, "ps": 6.4,  "pb": 6.8},
    "AMZN":  {"name": "Amazon.com, Inc.",        "shares_b": 10.50, "pe": 36.8, "f_pe": 28.2, "ev_ebitda": 18.4, "ps": 3.6,  "pb": 8.1},
    "AAPL":  {"name": "Apple Inc.",              "shares_b": 15.12, "pe": 30.5, "f_pe": 26.8, "ev_ebitda": 23.4, "ps": 8.9,  "pb": 44.2},
    "META":  {"name": "Meta Platforms, Inc.",    "shares_b": 2.53,  "pe": 25.4, "f_pe": 21.6, "ev_ebitda": 16.2, "ps": 9.1,  "pb": 8.9},
    "TSLA":  {"name": "Tesla, Inc.",             "shares_b": 3.19,  "pe": 68.5, "f_pe": 52.0, "ev_ebitda": 41.2, "ps": 8.4,  "pb": 12.3},
    "AMD":   {"name": "Advanced Micro Devices",  "shares_b": 1.62,  "pe": 42.1, "f_pe": 24.8, "ev_ebitda": 26.5, "ps": 9.8,  "pb": 4.5},
    "AVGO":  {"name": "Broadcom Inc.",           "shares_b": 4.67,  "pe": 39.2, "f_pe": 28.1, "ev_ebitda": 25.4, "ps": 15.6, "pb": 11.8},
    "ORCL":  {"name": "Oracle Corporation",      "shares_b": 2.77,  "pe": 34.0, "f_pe": 24.5, "ev_ebitda": 20.1, "ps": 8.5,  "pb": 38.0},
    "JPM":   {"name": "JPMorgan Chase & Co.",    "shares_b": 2.82,  "pe": 13.2, "f_pe": 12.5, "ev_ebitda": 10.4, "ps": 4.1,  "pb": 1.9},
    "GS":    {"name": "Goldman Sachs Group",     "shares_b": 0.31,  "pe": 14.8, "f_pe": 12.9, "ev_ebitda": 11.2, "ps": 3.2,  "pb": 1.5},
    "MS":    {"name": "Morgan Stanley",          "shares_b": 1.61,  "pe": 15.6, "f_pe": 13.8, "ev_ebitda": 11.8, "ps": 3.4,  "pb": 1.8},
    "BLK":   {"name": "BlackRock, Inc.",         "shares_b": 0.15,  "pe": 22.8, "f_pe": 19.5, "ev_ebitda": 16.9, "ps": 7.2,  "pb": 3.4},
    "V":     {"name": "Visa Inc.",               "shares_b": 1.95,  "pe": 29.4, "f_pe": 25.1, "ev_ebitda": 21.8, "ps": 15.8, "pb": 14.6},
    "MC.PA": {"name": "LVMH Moet Hennessy",      "shares_b": 0.50,  "pe": 23.5, "f_pe": 21.0, "ev_ebitda": 14.2, "ps": 4.1,  "pb": 5.2},
    "OR.PA": {"name": "L'Oreal S.A.",            "shares_b": 0.53,  "pe": 31.0, "f_pe": 27.8, "ev_ebitda": 20.5, "ps": 5.1,  "pb": 6.9},
    "AIR.PA":{"name": "Airbus SE",               "shares_b": 0.79,  "pe": 26.4, "f_pe": 20.2, "ev_ebitda": 13.8, "ps": 1.8,  "pb": 6.1},
    "TTE.PA":{"name": "TotalEnergies SE",        "shares_b": 2.28,  "pe": 8.4,  "f_pe": 7.9,  "ev_ebitda": 5.1,  "ps": 0.7,  "pb": 1.2},
    "BNP.PA":{"name": "BNP Paribas SA",          "shares_b": 1.13,  "pe": 7.6,  "f_pe": 6.9,  "ev_ebitda": 6.2,  "ps": 1.5,  "pb": 0.65},
    "QQQ":   {"name": "Invesco QQQ (Nasdaq 100)","shares_b": 0.62,  "pe": 30.8, "f_pe": 25.4, "ev_ebitda": 21.5, "ps": 6.8,  "pb": 8.4},
    "SPY":   {"name": "SPDR S&P 500 ETF Trust",  "shares_b": 0.98,  "pe": 24.6, "f_pe": 21.1, "ev_ebitda": 16.8, "ps": 3.1,  "pb": 4.8},
    "DIA":   {"name": "SPDR Dow Jones ETF",      "shares_b": 0.09,  "pe": 21.4, "f_pe": 18.9, "ev_ebitda": 14.9, "ps": 2.7,  "pb": 5.1},
    "IWM":   {"name": "iShares Russell 2000 ETF","shares_b": 0.31,  "pe": 18.2, "f_pe": 16.4, "ev_ebitda": 12.8, "ps": 1.4,  "pb": 2.1},
}

# Mapping indices
INDEX_PROXY_MAP = {
    "NQ=F": ("QQQ", "Invesco QQQ (Proxy Nasdaq 100)"),
    "^NDX": ("QQQ", "Invesco QQQ (Proxy Nasdaq 100)"),
    "^IXIC": ("QQQ", "Invesco QQQ (Proxy Nasdaq)"),
    "ES=F": ("SPY", "SPDR S&P 500 ETF (Proxy S&P 500)"),
    "^GSPC": ("SPY", "SPDR S&P 500 ETF (Proxy S&P 500)"),
    "YM=F": ("DIA", "SPDR Dow Jones ETF (Proxy Dow 30)"),
    "RTY=F": ("IWM", "iShares Russell 2000 ETF")
}

DEFAULT_UNIVERSE = list(INSTITUTIONAL_FALLBACK_DB.keys()) + ["NQ=F", "ES=F"]

# Series temporelles
@st.cache_data(ttl=180, show_spinner=False)
def fetch_historical_series(ticker: str, years: int = 5):
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=years * 365 + 15)
    end_date = now + datetime.timedelta(days=1)
    
    try:
        t_obj = yf.Ticker(ticker)
        data = t_obj.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=True)
        
        if data.empty or 'Close' not in data.columns:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty:
                return pd.Series(dtype=float), None, "INDISPONIBLE"
            prices = data['Close'].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data['Close']
        else:
            prices = data['Close']
            
        prices = prices.dropna()
        if prices.empty:
            return pd.Series(dtype=float), None, "INDISPONIBLE"

        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        status_mode = "CLOTURE OFFICIELLE"
        
        # Flux intraday
        try:
            intraday = t_obj.history(period="5d", interval="5m", prepost=True)
            if not intraday.empty and 'Close' in intraday.columns:
                last_intra_val = float(intraday['Close'].dropna().iloc[-1])
                last_intra_ts = pd.to_datetime(intraday.index[-1]).tz_localize(None)
                last_daily_date = prices.index[-1].date()
                
                if last_intra_ts.date() > last_daily_date:
                    new_point = pd.Series([last_intra_val], index=[pd.Timestamp(last_intra_ts.date())])
                    prices = pd.concat([prices, new_point])
                    status_mode = "TEMPS REEL / PRE-MARCHE"
                elif last_intra_ts.date() == last_daily_date:
                    prices.iloc[-1] = last_intra_val
                    if last_intra_ts.date() == now.date():
                        status_mode = "TEMPS REEL / INTRADAY"
        except Exception:
            pass

        last_dt = prices.index[-1]
        return pd.Series(prices.values.ravel(), index=prices.index, name=ticker), last_dt, status_mode
    except Exception:
        return pd.Series(dtype=float), None, "INDISPONIBLE"

# Extraction fondamentaux
@st.cache_data(ttl=600, show_spinner=False)
def fetch_fundamentals_bulletproof(tickers: tuple):
    rows = []
    for t in tickers:
        eff_t = INDEX_PROXY_MAP[t][0] if t in INDEX_PROXY_MAP else t
        fb = INSTITUTIONAL_FALLBACK_DB.get(eff_t, {})
        
        # Prix historique
        s_hist, _, _ = fetch_historical_series(eff_t, years=1)
        live_price = float(s_hist.iloc[-1]) if not s_hist.empty else np.nan
        
        name = fb.get("name", eff_t)
        mcap_b = (live_price * fb["shares_b"]) if (pd.notnull(live_price) and "shares_b" in fb) else np.nan
        pe = fb.get("pe", np.nan)
        f_pe = fb.get("f_pe", np.nan)
        ev_ebitda = fb.get("ev_ebitda", np.nan)
        ps = fb.get("ps", np.nan)
        pb = fb.get("pb", np.nan)

        # Enrichissement API
        try:
            info = yf.Ticker(eff_t).info
            if info and isinstance(info, dict):
                name = info.get('shortName') or info.get('longName') or name
                if info.get('marketCap'):
                    mcap_b = info['marketCap'] / 1e9
                if info.get('trailingPE'):
                    pe = float(info['trailingPE'])
                if info.get('forwardPE'):
                    f_pe = float(info['forwardPE'])
                if info.get('enterpriseToEbitda'):
                    ev_ebitda = float(info['enterpriseToEbitda'])
                if info.get('priceToSalesTrailing12Months'):
                    ps = float(info['priceToSalesTrailing12Months'])
                if info.get('priceToBook'):
                    pb = float(info['priceToBook'])
        except Exception:
            pass

        # Secours capitalisation
        if pd.isnull(mcap_b):
            try:
                fi = yf.Ticker(eff_t).fast_info
                if hasattr(fi, 'market_cap') and fi.market_cap:
                    mcap_b = fi.market_cap / 1e9
            except Exception:
                pass

        rows.append({
            'Ticker': t,
            'Nom': name,
            'Prix ($)': live_price,
            'Capitalisation ($B)': mcap_b,
            'P/E (Trailing)': pe,
            'Forward P/E': f_pe,
            'EV/EBITDA': ev_ebitda,
            'P/S': ps,
            'P/B': pb
        })
    return pd.DataFrame(rows)

# Recherche actifs
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


# Page accueil
if st.session_state.page == "home":
    now_str = datetime.datetime.now().strftime("%d %b %Y — %H:%M")
    st.markdown(f"""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <span style="color: #00e5ff; font-family: 'JetBrains Mono'; font-size: 0.82rem; letter-spacing: 0.12em;">
                PLATEFORME QUANTITATIVE INSTITUTIONNELLE
            </span>
            <span class="live-badge">SYNCHRONISATION : {now_str}</span>
        </div>
        <h1 style="font-size: 2.4rem; font-weight: 800; color: #ffffff; margin: 14px 0 10px 0;">
            QuantValuation & Stochastic Risk Terminal
        </h1>
        <p style="font-size: 1.02rem; color: #94a3b8; max-width: 820px; line-height: 1.6;">
            Analyse fondamentale multi-ratios (P/E Trailing vs Forward P/E), cones de probabilite Monte Carlo 
            et mesure du risque extreme (VaR Cornish-Fisher) connectes aux flux de marche en direct.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### Lancer une analyse d'actif")
        home_query = st.text_input("Symbole ou nom d'entreprise :", value=st.session_state.selected_ticker)
        res = search_securities(home_query)
        if res:
            sel = st.selectbox("Resultats :", options=res, format_func=lambda x: x[1])
            target_sym = sel[0]
        else:
            target_sym = home_query.upper().strip() or "NVDA"
            
        if st.button("Ouvrir le Terminal Quantitatif", type="primary", use_container_width=True):
            go_to_terminal(ticker=target_sym)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### Univers Sectoriels Preconfigures")
        if st.button("Big Tech & IA (NVDA, MSFT, GOOGL, AMZN, AAPL, META)", use_container_width=True):
            go_to_terminal("NVDA", ["MSFT", "GOOGL", "AMZN", "AAPL", "META"])
            st.rerun()
        if st.button("Contrats Futures & Indices US (NQ=F, QQQ, SPY, DIA, IWM)", use_container_width=True):
            go_to_terminal("NQ=F", ["QQQ", "SPY", "DIA", "IWM"])
            st.rerun()
        if st.button("Banques Wall Street (JPM, GS, MS, BLK, V)", use_container_width=True):
            go_to_terminal("JPM", ["GS", "MS", "BLK", "V"])
            st.rerun()
        if st.button("Leaders CAC 40 (MC.PA, OR.PA, AIR.PA, TTE.PA, BNP.PA)", use_container_width=True):
            go_to_terminal("MC.PA", ["OR.PA", "AIR.PA", "TTE.PA", "BNP.PA"])
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# Terminal quantitatif
else:
    # En-tete barre laterale
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#00e5ff; letter-spacing:0.15em; font-weight:700;">
            QUANT TERMINAL v2.5
        </div>
        <div style="font-size:0.72rem; color:#94a3b8; margin-top:2px;">
            CONSOLE DE PARAMETRAGE
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        if st.button("Accueil", use_container_width=True):
            go_to_home()
            st.rerun()
    with col_sb2:
        if st.button("Actualiser", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Section actif
    st.sidebar.markdown('<div class="sidebar-section">01 / Univers & Actifs</div>', unsafe_allow_html=True)
    search_term = st.sidebar.text_input(
        "Recherche d'actif (NVDA, NQ=F, AAPL)",
        value=st.session_state.selected_ticker
    )
    search_results = search_securities(search_term)

    if search_results:
        selected_tuple = st.sidebar.selectbox(
            "Resultats disponibles :",
            options=search_results,
            format_func=lambda x: x[1]
        )
        primary_ticker = selected_tuple[0]
    else:
        primary_ticker = search_term.upper().strip() if search_term else "NVDA"

    st.session_state.selected_ticker = primary_ticker

    available_options = sorted(list(set(DEFAULT_UNIVERSE + st.session_state.selected_peers + [primary_ticker])))
    default_peers = [p for p in st.session_state.selected_peers if p != primary_ticker]
    
    peer_list = st.sidebar.multiselect(
        "Actifs comparables :",
        options=[x for x in available_options if x != primary_ticker],
        default=default_peers[:5]
    )

    # Section stochastique
    st.sidebar.markdown('<div class="sidebar-section">02 / Modele Stochastique</div>', unsafe_allow_html=True)
    stoch_model = st.sidebar.selectbox(
        "Loi de distribution",
        ["GBM Classique (Loi Normale)", "Queues Epaisses (t-Student, df=5)"]
    )
    n_simulations = st.sidebar.slider("Trajectoires Monte Carlo", 500, 10000, 2500, 500)
    forecast_days = st.sidebar.slider("Horizon (jours ouvres)", 21, 504, 252, 21)
    history_years = st.sidebar.selectbox("Profondeur historique (ans)", [1, 3, 5, 10], index=2)

    # Section macro
    st.sidebar.markdown('<div class="sidebar-section">03 / Facteur Macro</div>', unsafe_allow_html=True)
    macro_ticker = st.sidebar.text_input("Indice de reference", value="^GSPC")

    # Chargement donnees
    hist_prices, last_quote_dt, status_mode = fetch_historical_series(primary_ticker, years=history_years)
    all_tickers = tuple([primary_ticker] + [p for p in peer_list if p != primary_ticker])
    df_metrics = fetch_fundamentals_bulletproof(all_tickers)

    if last_quote_dt is not None:
        date_str = last_quote_dt.strftime("%d/%m/%Y")
        badge_text = f"DERNIERE COTATION : {date_str} ({status_mode})"
    else:
        date_str = "N/A"
        badge_text = "FLUX HORS LIGNE"

    # En-tete principal
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:16px; flex-wrap:wrap; gap:12px;">
        <div>
            <div style="font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; color: #00e5ff; font-family:'JetBrains Mono';">
                PLATEFORME QUANTITATIVE INSTITUTIONNELLE
            </div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #ffffff; margin-top: 2px;">
                ANALYSE D'ACTIF : {primary_ticker}
            </div>
        </div>
        <div class="live-badge">{badge_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Calculs globaux
    if len(hist_prices) > 30:
        s0 = float(hist_prices.iloc[-1])
        s_prev = float(hist_prices.iloc[-2])
        daily_chg = ((s0 - s_prev) / s_prev) * 100
        
        log_ret = np.log(hist_prices / hist_prices.shift(1)).dropna()
        mu_daily = float(log_ret.mean())
        sigma_daily = float(log_ret.std())
        mu_ann = mu_daily * 252
        sigma_ann = sigma_daily * np.sqrt(252)
        
        s_val = float(skew(log_ret.values))
        k_val = float(kurtosis(log_ret.values))
        z95 = norm.ppf(0.05)
        z_cf95 = z95 + (1/6)*(z95**2 - 1)*s_val + (1/24)*(z95**3 - 3*z95)*k_val - (1/36)*(2*z95**3 - 5*z95)*(s_val**2)
        var95_pct = -(mu_daily + z_cf95 * sigma_daily) * 100

        pe_median = df_metrics['P/E (Trailing)'].median()
        total_mcap_t = df_metrics['Capitalisation ($B)'].sum() / 1000.0

        # Bandeau KPI
        chg_class = "kpi-sub-pos" if daily_chg >= 0 else "kpi-sub-neg"
        chg_sign = "+" if daily_chg >= 0 else ""
        
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Cours Actuel ({date_str})</div>
                <div class="kpi-value">
                    ${s0:,.2f}
                    <span class="{chg_class}">{chg_sign}{daily_chg:.2f}%</span>
                </div>
                <div class="kpi-sub-neutral">Capitalisation groupe : ${total_mcap_t:.2f}T</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card kpi-card-alt">
                <div class="kpi-title">P/E Median Sectoriel</div>
                <div class="kpi-value">{pe_median:.1f}x</div>
                <div class="kpi-sub-neutral">Sur {len(all_tickers)} actifs compares</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card kpi-card-warn">
                <div class="kpi-title">Volatilite Annualisee</div>
                <div class="kpi-value">{sigma_ann * 100:.1f}%</div>
                <div class="kpi-sub-neutral">Profondeur : {history_years} ans</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card kpi-card-risk">
                <div class="kpi-title">VaR 95% (Cornish-Fisher 1j)</div>
                <div class="kpi-value">
                    -{var95_pct:.2f}%
                    <span class="kpi-sub-neg">-${s0 * (var95_pct/100):,.2f}</span>
                </div>
                <div class="kpi-sub-neutral">Ajustee Skewness & Kurtosis</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "1. Multiples & Valorisation",
        "2. Mouvement Brownien & Monte Carlo",
        "3. Gestion des Risques (VaR & CVaR)",
        "4. Analyse Factorielle & Correlations"
    ])

    # Generateur Monte Carlo
    def build_monte_carlo_figure(height=420):
        if "t-Student" in stoch_model:
            df_t = 5
            raw_shocks = student_t.rvs(df_t, size=(forecast_days, n_simulations))
            shocks = raw_shocks / np.sqrt(df_t / (df_t - 2))
        else:
            shocks = np.random.normal(0, 1, (forecast_days, n_simulations))
            
        drift = (mu_daily - 0.5 * (sigma_daily ** 2))
        daily_returns_sim = drift + sigma_daily * shocks
        price_trajectories = s0 * np.exp(np.vstack([np.zeros((1, n_simulations)), np.cumsum(daily_returns_sim, axis=0)]))
        
        future_dates = pd.bdate_range(start=last_quote_dt, periods=forecast_days + 1)
        p5_path = np.percentile(price_trajectories, 5, axis=1)
        p25_path = np.percentile(price_trajectories, 25, axis=1)
        p50_path = np.percentile(price_trajectories, 50, axis=1)
        p75_path = np.percentile(price_trajectories, 75, axis=1)
        p95_path = np.percentile(price_trajectories, 95, axis=1)

        fig = go.Figure()
        # Intervalle 5-95
        fig.add_trace(go.Scatter(
            x=future_dates, y=p95_path, mode='lines',
            line=dict(color='rgba(0, 229, 255, 0.5)', width=1.2, dash='dot'), name='95e Centile'
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=p5_path, mode='lines',
            fill='tonexty', fillcolor='rgba(0, 229, 255, 0.09)',
            line=dict(color='rgba(0, 229, 255, 0.35)', width=1.2, dash='dot'), name='Cone 5% - 95%'
        ))
        # Intervalle 25-75
        fig.add_trace(go.Scatter(
            x=future_dates, y=p75_path, mode='lines',
            line=dict(color='rgba(0, 230, 118, 0.55)', width=1.2), name='75e Centile'
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=p25_path, mode='lines',
            fill='tonexty', fillcolor='rgba(0, 230, 118, 0.18)',
            line=dict(color='rgba(0, 230, 118, 0.55)', width=1.2), name='Cone 25% - 75%'
        ))
        # Echantillon trajectoires
        for j in range(min(n_simulations, 45)):
            fig.add_trace(go.Scatter(
                x=future_dates, y=price_trajectories[:, j], mode='lines',
                line=dict(color='rgba(0, 229, 255, 0.12)', width=0.8),
                showlegend=False, hoverinfo='skip'
            ))
        # Mediane
        fig.add_trace(go.Scatter(
            x=future_dates, y=p50_path, mode='lines',
            line=dict(color='#00e676', width=2.8), name='Mediane (50%)'
        ))
        fig.update_layout(
            height=height,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,17,29,0.5)",
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Prix ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        return fig, price_trajectories, future_dates

    # Onglet 1 : Multiples
    with tabs[0]:
        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("#### Valorisation Relative Sectorielle")
            
            st.dataframe(
                df_metrics[['Ticker', 'Nom', 'Capitalisation ($B)', 'P/E (Trailing)', 'Forward P/E', 'EV/EBITDA']].set_index('Ticker'),
                column_config={
                    "Nom": st.column_config.TextColumn("Nom"),
                    "Capitalisation ($B)": st.column_config.NumberColumn("Capitalisation ($B)", format="%.2fB"),
                    "P/E (Trailing)": st.column_config.NumberColumn("P/E (Trailing)", format="%.2fx"),
                    "Forward P/E": st.column_config.NumberColumn("Forward P/E", format="%.2fx"),
                    "EV/EBITDA": st.column_config.NumberColumn("EV/EBITDA", format="%.1fx"),
                },
                use_container_width=True
            )

            ratio_choice = st.selectbox(
                "Multiple de valorisation a analyser :",
                ["P/E (Trailing) vs Forward P/E (Groupe)", "P/E (Trailing)", "Forward P/E", "EV/EBITDA", "P/S", "P/B"]
            )

            fig_bar = go.Figure()
            if "Groupe" in ratio_choice:
                med_pe = df_metrics['P/E (Trailing)'].median()
                fig_bar.add_trace(go.Bar(
                    x=df_metrics['Ticker'],
                    y=df_metrics['P/E (Trailing)'],
                    name="P/E Trailing",
                    marker_color="#00e5ff",
                    text=df_metrics['P/E (Trailing)'].apply(lambda x: f"{x:.1f}x"),
                    textposition="outside"
                ))
                fig_bar.add_trace(go.Bar(
                    x=df_metrics['Ticker'],
                    y=df_metrics['Forward P/E'],
                    name="Forward P/E",
                    marker_color="#00e676",
                    text=df_metrics['Forward P/E'].apply(lambda x: f"{x:.1f}x"),
                    textposition="outside"
                ))
                fig_bar.add_hline(
                    y=med_pe,
                    line=dict(color="#ffab00", dash="dash", width=1.8),
                    annotation_text=f"Mediane : {med_pe:.1f}x",
                    annotation_position="top right"
                )
            else:
                med_val = df_metrics[ratio_choice].median()
                fig_bar.add_trace(go.Bar(
                    x=df_metrics['Ticker'],
                    y=df_metrics[ratio_choice],
                    name=ratio_choice,
                    marker_color="#00e5ff",
                    text=df_metrics[ratio_choice].apply(lambda x: f"{x:.1f}x"),
                    textposition="outside"
                ))
                fig_bar.add_hline(
                    y=med_val,
                    line=dict(color="#ffab00", dash="dash", width=1.8),
                    annotation_text=f"Mediane : {med_val:.1f}x",
                    annotation_position="top right"
                )

            y_max = max(df_metrics['P/E (Trailing)'].max(), df_metrics['Forward P/E'].max()) * 1.22
            fig_bar.update_layout(
                height=290,
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,17,29,0.5)",
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(range=[0, y_max], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("#### Simulation Monte Carlo & Trajectoires")
            if len(hist_prices) > 30:
                fig_mc_preview, price_trajectories, future_dates = build_monte_carlo_figure(height=535)
                st.plotly_chart(fig_mc_preview, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Onglet 2 : Monte Carlo
    with tabs[1]:
        if len(hist_prices) > 30:
            fig_mc_full, price_trajectories, future_dates = build_monte_carlo_figure(height=440)
            final_prices = price_trajectories[-1, :]
            p50_end = float(np.median(final_prices))

            c1, c2 = st.columns([3, 1.8])
            with c1:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown("#### Cone de Projection Stochastique")
                st.plotly_chart(fig_mc_full, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown(f"#### Distribution au {future_dates[-1].strftime('%d/%m/%Y')}")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=final_prices[final_prices >= s0], nbinsx=35,
                    marker_color='#00e676', opacity=0.75, name="Haussier (> S0)"
                ))
                fig_hist.add_trace(go.Histogram(
                    x=final_prices[final_prices < s0], nbinsx=35,
                    marker_color='#ff5252', opacity=0.75, name="Baissier (< S0)"
                ))
                fig_hist.add_vline(x=s0, line_color="#ffffff", line_dash="dash", annotation_text="S0")
                fig_hist.add_vline(x=p50_end, line_color="#00e5ff", line_width=2, annotation_text=f"Mediane (${p50_end:.0f})")
                fig_hist.update_layout(
                    height=440,
                    barmode='overlay',
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,17,29,0.5)",
                    xaxis_title="Prix Final ($)",
                    yaxis_title="Frequence",
                    margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # Onglet 3 : Risques
    with tabs[2]:
        if len(hist_prices) > 30:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            returns = log_ret.values
            confidence_level = st.slider("Niveau de confiance statistique (1 - alpha)", 0.90, 0.99, 0.95, 0.01)
            alpha = 1.0 - confidence_level
            
            z_norm = norm.ppf(alpha)
            z_cf = (z_norm + (1/6)*(z_norm**2 - 1)*s_val + (1/24)*(z_norm**3 - 3*z_norm)*k_val - (1/36)*(2*z_norm**3 - 5*z_norm)*(s_val**2))
            
            var_cf_daily = -(mu_daily + z_cf * sigma_daily)
            var_hist_daily = -np.percentile(returns, alpha * 100)
            cvar_hist_daily = -returns[returns <= -var_hist_daily].mean()
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(
                x=returns * 100, nbinsx=80,
                marker_color='#00e5ff', opacity=0.65, name="Rendements Journaliers"
            ))
            fig_var.add_vline(x=-var_cf_daily * 100, line_color="#ffab00", line_width=2.2, annotation_text=f"VaR CF (-{var_cf_daily*100:.2f}%)")
            fig_var.add_vline(x=-cvar_hist_daily * 100, line_color="#ff5252", line_width=2.2, line_dash="dash", annotation_text=f"CVaR (-{cvar_hist_daily*100:.2f}%)")
            fig_var.update_layout(
                height=420,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,17,29,0.5)",
                xaxis_title="Rendement Journalier (%)",
                yaxis_title="Occurrences",
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_var, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Onglet 4 : Correlations
    with tabs[3]:
        s_bench, _, _ = fetch_historical_series(macro_ticker, years=history_years)
        df_pair = pd.DataFrame({primary_ticker: hist_prices, macro_ticker: s_bench}).dropna()
        
        if len(df_pair) > 30:
            df_rebased = (df_pair / df_pair.iloc[0]) * 100
            returns_pair = df_pair.pct_change().dropna()
            rolling_corr = returns_pair[primary_ticker].rolling(60).corr(returns_pair[macro_ticker])
            
            c_cor1, c_cor2 = st.columns(2)
            with c_cor1:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
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
                    height=400,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,17,29,0.5)",
                    title="Performance Comparee (Base 100)",
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c_cor2:
                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(
                    x=rolling_corr.index, y=rolling_corr,
                    mode='lines', fill='tozeroy',
                    fillcolor='rgba(0, 230, 118, 0.14)',
                    line=dict(color='#00e676', width=1.8),
                    name="Correlation Glissante 60j"
                ))
                fig_roll.update_layout(
                    height=400,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(11,17,29,0.5)",
                    title="Correlation Glissante 60 Jours",
                    yaxis=dict(range=[-1, 1]),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_roll, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
