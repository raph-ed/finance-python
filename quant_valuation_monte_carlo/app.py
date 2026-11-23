import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import kurtosis, norm, skew, t as student_t
import streamlit as st
import yfinance as yf

# Configuration page
st.set_page_config(
    page_title="QuantTerminal | Valuation & Risk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Etat session
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "NVDA"
if "selected_peers" not in st.session_state:
    st.session_state.selected_peers = "MSFT, GOOGL, AMZN, AAPL, META"

# Univers sectoriels
PRESETS = [
    ("Big Tech & IA", "NVDA", "MSFT, GOOGL, AMZN, AAPL, META"),
    ("Stockage & Semi-conducteurs", "SNDK", "WDC, MU, NVDA, AMD, AVGO"),
    ("Indices & Futures US", "QQQ", "SPY, DIA, IWM, NQ=F"),
    ("Banques Wall Street", "JPM", "GS, MS, BLK, BAC, V"),
    ("Leaders CAC 40", "MC.PA", "OR.PA, AIR.PA, TTE.PA, BNP.PA"),
]

# Theme CSS temporaire (refonte visuelle prevue a l'etape suivante)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp {
    background: radial-gradient(circle at 18% 0%, #101827 0%, #090d14 55%, #070a0f 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] {
    background-color: #0c111b;
    border-right: 1px solid #1e293b;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    border: 1px solid #263348 !important;
    border-radius: 6px !important;
    color: #f1f5f9 !important;
}
div[data-baseweb="input"] input {
    color: #f1f5f9 !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.35) !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background-color: #3b82f6 !important;
    border-color: #3b82f6 !important;
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #93c5fd !important;
    font-family: 'JetBrains Mono', monospace;
}

div.stButton > button {
    background-color: #111827;
    color: #e2e8f0;
    border: 1px solid #263348;
    border-radius: 6px;
    padding: 0.55rem 1rem;
    font-weight: 500;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
div.stButton > button:hover {
    background-color: #172133;
    border-color: #3b82f6;
    color: #ffffff;
}
div.stButton > button[kind="primary"] {
    background-color: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #1d4ed8;
    border-color: #1d4ed8;
}

div[data-testid="metric-container"] {
    background-color: #0f1623;
    border: 1px solid #1e293b;
    padding: 14px 16px;
    border-radius: 6px;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.3rem;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    color: #94a3b8;
    background: transparent;
    border: none;
    font-weight: 500;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background-color: #131c2e !important;
    border-bottom: 2px solid #3b82f6 !important;
}

.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #0b111e 100%);
    border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 28px 32px;
    margin-bottom: 28px;
}
.hero-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    color: #60a5fa;
    text-transform: uppercase;
}
.hero-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #94a3b8;
    background: #111827;
    border: 1px solid #263348;
    padding: 4px 10px;
    border-radius: 999px;
}
.hero-title {
    font-size: 1.95rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 8px 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    max-width: 780px;
}

.term-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding-bottom: 14px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 18px;
}
.term-symbol {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f8fafc;
}
.term-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    color: #93c5fd;
    margin-left: 14px;
}

/* Encadre de lecture dynamique */
.signal-card {
    background-color: #0f1623;
    border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 18px;
}
.signal-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.signal-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
}
.signal-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #94a3b8;
}
.signal-text {
    font-size: 0.94rem;
    line-height: 1.55;
    color: #e2e8f0;
    margin: 0;
}
.cond-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.cond-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 4px 9px;
    border-radius: 4px;
    background: #111827;
    border: 1px solid #263348;
}
</style>
""", unsafe_allow_html=True)

COLORS = {
    "bg": "#090d14",
    "panel": "#0f1623",
    "grid": "#1e293b",
    "accent": "#3b82f6",
    "median": "#f59e0b",
    "up": "#10b981",
    "down": "#ef4444",
    "muted": "#94a3b8",
}

def currency_symbol(ticker: str) -> str:
    return "€" if ticker.endswith((".PA", ".DE", ".AS", ".MI", ".MC", ".BR")) else "$"

def fmt_pnl(val: float, cur: str) -> str:
    sign = "+" if val >= 0 else "-"
    return f"{sign}{cur}{abs(val):,.0f}"

def render_signal_card(badge: str, accent: str, text: str, right_meta: str = "", chips: list = None):
    chips_html = ""
    if chips:
        items = []
        for label, passed in chips:
            if passed is None:
                c_color, state = "#64748b", "N/A"
            elif passed:
                c_color, state = COLORS["up"], "OK"
            else:
                c_color, state = COLORS["down"], "NON"
            items.append(f"<span class='cond-chip' style='border-color:{c_color}55;color:#cbd5e1;'><span style='color:{c_color};font-weight:600;'>{state}</span> · {label}</span>")
        chips_html = f"<div class='cond-grid'>{''.join(items)}</div>"

    st.markdown(f"""
    <div class="signal-card" style="border-left-color: {accent};">
        <div class="signal-head">
            <span class="signal-badge" style="color: {accent}; background: {accent}1a; border: 1px solid {accent}55;">{badge}</span>
            <span class="signal-score">{right_meta}</span>
        </div>
        <p class="signal-text">{text}</p>
        {chips_html}
    </div>
    """, unsafe_allow_html=True)

def style_fig(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="Inter, sans-serif", size=12, color="#cbd5e1"),
        margin=dict(l=20, r=20, t=34, b=20),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor="#334155"),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor="#334155"),
    )
    if kwargs:
        fig.update_layout(**kwargs)
    return fig

def batch_paths(paths_2d: np.ndarray, time_steps: np.ndarray):
    n = paths_2d.shape[1]
    if n == 0:
        return [], []
    x = np.tile(np.append(time_steps, np.nan), n)
    y = np.vstack([paths_2d, np.full((1, n), np.nan)]).T.ravel()
    return x, y

# Statistiques de risque
def compute_risk_stats(returns: np.ndarray, conf: float = 0.95) -> dict:
    alpha = 1.0 - conf
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    s_val = float(skew(returns))
    k_val = float(kurtosis(returns))
    z = float(norm.ppf(alpha))
    z_cf = (
        z
        + (z**2 - 1) * s_val / 6
        + (z**3 - 3 * z) * k_val / 24
        - (2 * z**3 - 5 * z) * (s_val**2) / 36
    )
    var_cf = max(-(mu + z_cf * sigma), 1e-6)
    var_hist = float(-np.percentile(returns, alpha * 100))
    tail = returns[returns <= -var_hist]
    cvar_hist = float(-tail.mean()) if len(tail) else max(var_hist, 1e-6)
    return {
        "skew": s_val,
        "kurt": k_val,
        "var_cf": var_cf,
        "var_hist": var_hist,
        "cvar": max(cvar_hist, 1e-6),
    }

# Calcul centralise Timing & Sizing
def compute_timing_sizing(hist_prices: pd.Series, log_ret: pd.Series, capital: float, risk_pct: float, max_pos_pct: float) -> dict:
    s0 = float(hist_prices.iloc[-1])
    mm50 = hist_prices.rolling(50).mean()
    mm200 = hist_prices.rolling(200).mean()

    has_mm50 = mm50.dropna().shape[0] >= 1
    has_mm200 = mm200.dropna().shape[0] >= 21

    above_mm50 = bool(s0 >= float(mm50.iloc[-1])) if has_mm50 else None
    if has_mm200:
        mm200_now = float(mm200.iloc[-1])
        dist_mm200 = (s0 / mm200_now - 1) * 100
        mm200_slope = (mm200_now / float(mm200.iloc[-21]) - 1) * 100
        above_mm200 = bool(s0 >= mm200_now)
        slope_pos = bool(mm200_slope >= 0)
        trend_label = "Haussière" if (above_mm200 and slope_pos) else ("Transition" if (above_mm200 or slope_pos) else "Baissière")
    else:
        dist_mm200, mm200_slope, above_mm200, slope_pos = np.nan, np.nan, None, None
        trend_label = "Historique < 200j"

    if len(hist_prices) >= 253:
        mom_12_1 = (float(hist_prices.iloc[-22]) / float(hist_prices.iloc[-253]) - 1) * 100
        mom_pos = bool(mom_12_1 > 0)
    else:
        mom_12_1, mom_pos = np.nan, None

    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    long_win = min(252, max(60, len(log_ret) // 2))
    vol_long = log_ret.rolling(long_win).std() * np.sqrt(252)
    vol_ratio_series = (vol_20 / vol_long).dropna()
    vol_ratio = float(vol_ratio_series.iloc[-1]) if not vol_ratio_series.empty else 1.0
    vol_ok = bool(vol_ratio <= 1.25)

    if vol_ratio > 1.35:
        vol_regime = "Stress"
    elif vol_ratio < 0.85:
        vol_regime = "Calme"
    else:
        vol_regime = "Normal"

    # Sizing par CVaR 95%
    cvar_95 = compute_risk_stats(log_ret.values, conf=0.95)["cvar"]
    raw_weight = (risk_pct / 100.0) / cvar_95
    regime_mult = (1.0 / vol_ratio) if vol_ratio > 1.35 else 1.0
    adj_weight = raw_weight * regime_mult
    final_weight = min(adj_weight, max_pos_pct / 100.0)

    target_amount = capital * final_weight
    n_shares = int(target_amount // s0) if s0 > 0 else 0
    real_amount = n_shares * s0
    invest_base = real_amount if n_shares > 0 else target_amount
    tail_loss = invest_base * cvar_95

    # Evaluation des conditions (score dynamique)
    conditions = [
        ("Prix > MM200", above_mm200),
        ("Pente MM200 > 0", slope_pos),
        ("Prix > MM50", above_mm50),
        ("Momentum 12-1M > 0", mom_pos),
        ("Volatilité maîtrisée", vol_ok),
    ]
    valid_conds = [c[1] for c in conditions if c[1] is not None]
    score = sum(1 for v in valid_conds if v)
    total = len(valid_conds)
    ratio_ok = (score / total) if total > 0 else 0.0

    # Classification en 5 etats
    if vol_ratio > 1.50 or score == 0:
        status = "Zone de stress — Aversion au risque"
        accent = COLORS["down"]
        tranches = "Attendre accalmie (ou 3+ tranches)"
        narrative = (
            f"Le marché est en phase de stress ou de baisse marquée ({score}/{total} critères réunis, volatilité à {vol_ratio:.2f}x sa normale). "
            f"Le modèle suggère de rester à l'écart tant que le titre ne stabilise pas sa volatilité ou ne repasse pas au-dessus de sa MM50."
        )
    elif ratio_ok >= 0.85:
        status = "Fenêtre d'entrée favorable"
        accent = COLORS["up"]
        tranches = "1 à 2 tranches"
        narrative = (
            f"La quasi-totalité des signaux sont alignés ({score}/{total} critères réunis) : tendance porteuse, momentum positif et volatilité sous contrôle ({vol_ratio:.2f}x). "
            f"Le cadre technique est propre pour engager jusqu'à {final_weight * 100:.1f}% du portefeuille en 1 à 2 tranches."
        )
    elif ratio_ok >= 0.65:
        status = "Configuration intéressante — Entrée progressive"
        accent = "#34d399"
        tranches = "2 tranches"
        narrative = (
            f"La configuration est constructive ({score}/{total} critères réunis) mais un élément reste à surveiller. "
            f"Un déploiement progressif en 2 tranches jusqu'à {final_weight * 100:.1f}% permet de participer au mouvement tout en lissant le point d'entrée."
        )
    elif ratio_ok >= 0.40:
        status = "Phase indécise — Consolidation"
        accent = COLORS["median"]
        tranches = "2 à 3 tranches (ou surveillance)"
        narrative = (
            f"Les signaux sont partagés ({score}/{total} critères réunis) : l'actif hésite entre reprise et essoufflement. "
            f"Mieux vaut garder le titre en watchlist ou n'initier qu'une première demi-tranche en attendant une direction plus franche."
        )
    else:
        status = "Attente — Configuration fragile"
        accent = "#f97316"
        tranches = "Observation / 3 tranches max"
        narrative = (
            f"Trop peu de critères sont validés ({score}/{total}) : la dynamique reste dégradée et le risque d'entrer trop tôt est élevé. "
            f"Le modèle recommande la patience jusqu'au retour au-dessus de la MM50 ou d'un momentum positif."
        )

    return {
        "mm50": mm50,
        "mm200": mm200,
        "dist_mm200": dist_mm200,
        "mm200_slope": mm200_slope,
        "trend_label": trend_label,
        "mom_12_1": mom_12_1,
        "vol_ratio": vol_ratio,
        "vol_regime": vol_regime,
        "vol_ratio_series": vol_ratio_series,
        "long_win": long_win,
        "cvar_95": cvar_95,
        "raw_weight": raw_weight,
        "regime_mult": regime_mult,
        "adj_weight": adj_weight,
        "final_weight": final_weight,
        "target_amount": target_amount,
        "n_shares": n_shares,
        "real_amount": real_amount,
        "invest_base": invest_base,
        "tail_loss": tail_loss,
        "conditions": conditions,
        "score": score,
        "total": total,
        "status": status,
        "accent": accent,
        "tranches": tranches,
        "narrative": narrative,
    }

# Recherche priorisee
@st.cache_data(ttl=1800, show_spinner=False)
def search_securities(query: str):
    if not query or len(query.strip()) < 2:
        return []
    try:
        quotes = yf.Search(query.strip(), max_results=12).quotes
        preferred_exch = {"NMS", "NYQ", "NGM", "NAS", "PAR", "PCX", "BTS"}
        type_rank = {"EQUITY": 0, "ETF": 1, "INDEX": 2, "FUTURE": 3}

        def sort_key(q):
            return (
                type_rank.get(q.get("quoteType", ""), 4),
                0 if q.get("exchange", "") in preferred_exch else 1,
            )

        quotes = sorted([q for q in quotes if q.get("symbol")], key=sort_key)
        return [
            (
                q["symbol"],
                f"{q['symbol']} — {q.get('shortname') or q.get('longname') or q['symbol']} [{q.get('exchange', '')} | {q.get('quoteType', '')}]",
            )
            for q in quotes
        ]
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close(ticker: str, years: int = 5) -> pd.Series:
    start = datetime.datetime.now() - datetime.timedelta(days=years * 365 + 45)
    data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if data.empty:
        return pd.Series(dtype=float)
    prices = data["Close"] if "Close" in data else data.iloc[:, 0]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.dropna()
    return pd.Series(prices.values.ravel(), index=prices.index, name=ticker)

# Fondamentaux anti-split
def split_safe_ratio(raw_val, num, den):
    calc = (num / den) if (num and den and den > 0) else np.nan
    if pd.notnull(calc):
        if pd.isnull(raw_val) or raw_val <= 0 or not (0.65 <= raw_val / calc <= 1.5):
            return calc, True
    return (raw_val if (raw_val is not None and raw_val > 0) else np.nan), False

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(tickers: tuple):
    rows, adjusted_tickers = [], []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mcap = info.get("marketCap")
            net_inc = info.get("netIncomeToCommon")
            rev = info.get("totalRevenue")
            ev = info.get("enterpriseValue")
            ebitda = info.get("ebitda")

            pe_val, pe_fixed = split_safe_ratio(info.get("trailingPE"), mcap, net_inc)
            ps_val, _ = split_safe_ratio(info.get("priceToSalesTrailing12Months"), mcap, rev)
            ev_ebitda, _ = split_safe_ratio(info.get("enterpriseToEbitda"), ev, ebitda)

            fwd_pe = info.get("forwardPE", np.nan)
            if pe_fixed and pd.notnull(fwd_pe) and pd.notnull(pe_val) and fwd_pe < pe_val * 0.25:
                fwd_pe = np.nan
            if pe_fixed:
                adjusted_tickers.append(t)

            rows.append({
                "Ticker": t,
                "Nom": info.get("shortName", t),
                "Cap. ($B)": mcap / 1e9 if mcap else np.nan,
                "P/E": pe_val,
                "Fwd P/E": fwd_pe,
                "EV/EBITDA": ev_ebitda,
                "P/S": ps_val,
                "P/B": info.get("priceToBook", np.nan),
            })
        except Exception:
            rows.append({"Ticker": t, "Nom": t})
    return pd.DataFrame(rows), adjusted_tickers

# Vue Accueil
def render_home():
    now_str = datetime.datetime.now().strftime("%d %b %Y — %H:%M")
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-top">
            <span class="hero-eyebrow">Quant Valuation & Risk Suite</span>
            <span class="hero-pill">Flux Yahoo Finance · {now_str}</span>
        </div>
        <h1 class="hero-title">QuantTerminal</h1>
        <p class="hero-sub">
            Valorisation par multiples sectoriels corrigés des splits, Timing & Sizing par CVaR,
            simulations Monte Carlo (GBM / Student-t) et analyse factorielle face au benchmark.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.05, 1], gap="large")

    with col_left:
        st.markdown("#### Lancer une analyse")
        query = st.text_input("Symbole ou nom d'entreprise", value=st.session_state.selected_ticker, placeholder="ex: SNDK, NVDA, AAPL, LVMH...")
        results = search_securities(query)

        if results:
            chosen = st.selectbox("Résultats correspondants", options=results, format_func=lambda x: x[1])[0]
        else:
            chosen = query.upper().strip() or "NVDA"
            st.caption(f"Symbole direct : **{chosen}**")

        if st.button("Ouvrir le Terminal", type="primary", use_container_width=True):
            st.session_state.selected_ticker = chosen
            st.session_state.page = "terminal"
            st.rerun()

    with col_right:
        st.markdown("#### Univers sectoriels")
        for label, ticker, peers in PRESETS:
            if st.button(f"{label}  ({ticker}, {peers})", use_container_width=True):
                st.session_state.selected_ticker = ticker
                st.session_state.selected_peers = peers
                st.session_state.page = "terminal"
                st.rerun()

# Vue Terminal
def render_terminal():
    with st.sidebar:
        if st.button("← Accueil", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

        st.divider()
        st.markdown("**Univers**")
        search_term = st.text_input("Actif", value=st.session_state.selected_ticker)
        results = search_securities(search_term)
        if results:
            primary_ticker = st.selectbox("Sélection", options=results, format_func=lambda x: x[1])[0]
        else:
            primary_ticker = search_term.upper().strip() or st.session_state.selected_ticker

        st.session_state.selected_ticker = primary_ticker
        peers_input = st.text_input("Comparables", value=st.session_state.selected_peers)
        st.session_state.selected_peers = peers_input
        peer_list = [p.strip().upper() for p in peers_input.split(",") if p.strip()]

        st.divider()
        st.markdown("**Portefeuille (Sizing)**")
        cur = currency_symbol(primary_ticker)
        portfolio_capital = st.number_input(f"Capital total ({cur})", min_value=500.0, max_value=10_000_000.0, value=10_000.0, step=500.0)
        risk_budget_pct = st.slider("Risque max / jour extrême (%)", 0.1, 2.0, 0.5, 0.1)
        max_pos_pct = st.slider("Plafond par ligne (%)", 5, 35, 15, 1)

        st.divider()
        st.markdown("**Modélisation**")
        dist_mode = st.radio("Distribution des chocs", ["Normale (GBM)", "Student-t (Queues épaisses)"], index=0)
        n_simulations = st.slider("Trajectoires", 500, 10000, 2500, 500)
        forecast_days = st.slider("Horizon (jours ouvrés)", 21, 504, 252, 21)
        history_years = st.selectbox("Historique (années)", [1, 3, 5, 10], index=2)
        macro_ticker = st.text_input("Benchmark", value="^GSPC")

    hist_prices = fetch_close(primary_ticker, years=history_years)
    has_history = len(hist_prices) > 30

    if has_history:
        log_ret = np.log(hist_prices / hist_prices.shift(1)).dropna()
        drift_daily = float(log_ret.mean())
        sigma_daily = float(log_ret.std())
        mu_ann = (drift_daily + 0.5 * sigma_daily**2) * 252
        sigma_ann = sigma_daily * np.sqrt(252)
        s0 = float(hist_prices.iloc[-1])
        ts = compute_timing_sizing(hist_prices, log_ret, portfolio_capital, risk_budget_pct, max_pos_pct)
    else:
        ts = None

    price_html = f"<span class='term-price'>{cur}{s0:,.2f}</span>" if has_history else ""
    st.markdown(
        f"<div class='term-header'><div><span class='term-symbol'>{primary_ticker}</span>{price_html}</div>"
        f"<span style='font-family:JetBrains Mono;font-size:0.8rem;color:#64748b;'>Historique {history_years}A · Benchmark {macro_ticker}</span></div>",
        unsafe_allow_html=True,
    )

    tab_val, tab_ts, tab_mc, tab_risk, tab_bench = st.tabs([
        "Multiples",
        "Timing & Sizing",
        "Monte Carlo",
        "Risque (VaR/CVaR)",
        "Benchmark",
    ])

    # Onglet 1 : Multiples
    with tab_val:
        all_tickers = tuple([primary_ticker] + [p for p in peer_list if p != primary_ticker])
        with st.spinner("Chargement des ratios..."):
            df_metrics, adjusted = fetch_fundamentals(all_tickers)

        ratio_cols = ["P/E", "Fwd P/E", "EV/EBITDA", "P/S", "P/B"]
        if adjusted:
            st.caption(f"Recalcul anti-split appliqué via Market Cap / Net Income pour : {', '.join(adjusted)}.")

        if not df_metrics.empty and "P/E" in df_metrics.columns:
            col_ctrl, col_chart = st.columns([0.45, 1.55], gap="large")
            with col_ctrl:
                selected_ratio = st.selectbox("Multiple à comparer", ratio_cols)
                valid_vals = df_metrics[selected_ratio].dropna()
                med_val = float(valid_vals.median()) if not valid_vals.empty else np.nan
                target_series = df_metrics.loc[df_metrics["Ticker"] == primary_ticker, selected_ratio]
                target_val = float(target_series.values[0]) if (len(target_series) and pd.notnull(target_series.values[0])) else np.nan

                if pd.notnull(med_val):
                    st.metric("Médiane du groupe", f"{med_val:.2f}x")
                if pd.notnull(target_val) and pd.notnull(med_val) and med_val > 0:
                    spread = (target_val / med_val - 1) * 100
                    st.metric(f"{primary_ticker} vs Médiane", f"{target_val:.2f}x", f"{spread:+.1f}%", delta_color="inverse")

            # Lecture dynamique valorisation
            if pd.notnull(target_val) and pd.notnull(med_val) and med_val > 0:
                spread = (target_val / med_val - 1) * 100
                pe_row = df_metrics.loc[df_metrics["Ticker"] == primary_ticker]
                fwd_note = ""
                if not pe_row.empty and pd.notnull(pe_row["P/E"].values[0]) and pd.notnull(pe_row["Fwd P/E"].values[0]):
                    pe_t, pe_f = float(pe_row["P/E"].values[0]), float(pe_row["Fwd P/E"].values[0])
                    if pe_f < pe_t * 0.9:
                        fwd_note = f" Le Forward P/E ({pe_f:.1f}x) est nettement sous le P/E actuel ({pe_t:.1f}x), signe d'une croissance attendue des bénéfices."
                    elif pe_f > pe_t * 1.1:
                        fwd_note = f" Attention : le Forward P/E ({pe_f:.1f}x) dépasse le P/E actuel ({pe_t:.1f}x), ce qui reflète un ralentissement anticipé."

                if spread <= -15:
                    v_badge, v_col = "Décote sectorielle", COLORS["up"]
                    v_txt = f"{primary_ticker} se traite {abs(spread):.1f}% sous la médiane du groupe sur le {selected_ratio} ({target_val:.1f}x vs {med_val:.1f}x). À croiser avec la qualité fondamentale pour écarter un piège de valeur.{fwd_note}"
                elif spread >= 15:
                    v_badge, v_col = "Prime sectorielle", COLORS["down"]
                    v_txt = f"{primary_ticker} affiche une prime de {spread:+.1f}% face à ses comparables sur le {selected_ratio} ({target_val:.1f}x vs {med_val:.1f}x). Le marché intègre déjà des attentes élevées.{fwd_note}"
                else:
                    v_badge, v_col = "Valorisation en ligne", COLORS["accent"]
                    v_txt = f"{primary_ticker} est valorisé proche de la médiane sectorielle sur le {selected_ratio} ({target_val:.1f}x vs {med_val:.1f}x, écart {spread:+.1f}%).{fwd_note}"

                render_signal_card(v_badge, v_col, v_txt, right_meta=f"Multiple : {selected_ratio}")

            with col_chart:
                df_plot = df_metrics.dropna(subset=[selected_ratio])
                if not df_plot.empty:
                    max_y = float(df_plot[selected_ratio].max())
                    colors = [
                        (COLORS["up"] if v <= med_val else COLORS["down"]) if t == primary_ticker else COLORS["accent"]
                        for t, v in zip(df_plot["Ticker"], df_plot[selected_ratio])
                    ]
                    fig_bar = go.Figure(go.Bar(
                        x=df_plot["Ticker"],
                        y=df_plot[selected_ratio],
                        marker_color=colors,
                        text=df_plot[selected_ratio].map(lambda x: f"{x:.1f}x"),
                        textposition="outside",
                    ))
                    fig_bar.add_hline(
                        y=med_val,
                        line=dict(color=COLORS["median"], dash="dash", width=1.6),
                        annotation_text=f" Médiane {med_val:.1f}x ",
                        annotation_position="top left",
                        annotation=dict(bgcolor="rgba(15,22,35,0.9)", font=dict(color=COLORS["median"])),
                    )
                    st.plotly_chart(
                        style_fig(fig_bar, yaxis_title=selected_ratio, yaxis_range=[0, max_y * 1.22]),
                        use_container_width=True,
                    )

            fmt_num = lambda x: f"{x:.2f}" if pd.notnull(x) else "—"
            formatters = {"Cap. ($B)": lambda x: f"{x:.1f}B" if pd.notnull(x) else "—", **{c: fmt_num for c in ratio_cols}}
            st.dataframe(df_metrics.set_index("Ticker").style.format(formatters), use_container_width=True)

    # Onglet 2 : Timing & Sizing
    with tab_ts:
        if has_history and ts is not None:
            render_signal_card(
                badge=ts["status"],
                accent=ts["accent"],
                text=ts["narrative"],
                right_meta=f"Score modèle : {ts['score']} / {ts['total']} conditions",
                chips=ts["conditions"],
            )

            st.markdown("#### Dimensionnement de position")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Poids max conseillé", f"{ts['final_weight'] * 100:.1f}%", f"Brut CVaR : {ts['raw_weight'] * 100:.1f}%")
            s2.metric("Montant cible", f"{cur}{ts['target_amount']:,.0f}")
            s3.metric("Actions (entier)", f"{ts['n_shares']}", f"{cur}{ts['real_amount']:,.0f} engagés")
            s4.metric("Perte extrême 1j (CVaR)", f"-{cur}{ts['tail_loss']:,.0f}", f"-{(ts['tail_loss'] / portfolio_capital) * 100:.2f}% portef.")
            s5.metric("Cadence d'entrée", ts["tranches"])

            st.markdown("#### Indicateurs de régime")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Tendance (MM200)", ts["trend_label"], f"{ts['dist_mm200']:+.1f}% vs MM200" if pd.notnull(ts["dist_mm200"]) else "N/A")
            t2.metric("Pente MM200 (20j)", f"{ts['mm200_slope']:+.2f}%" if pd.notnull(ts["mm200_slope"]) else "—")
            t3.metric("Momentum 12-1M", f"{ts['mom_12_1']:+.1f}%" if pd.notnull(ts["mom_12_1"]) else "Historique < 1A")
            t4.metric("Régime Volatilité", ts["vol_regime"], f"Ratio 20j/{ts['long_win']}j : {ts['vol_ratio']:.2f}x")

            g1, g2 = st.columns([1.45, 1])
            with g1:
                plot_slice = hist_prices.tail(504)
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice, name=primary_ticker, line=dict(color="#f8fafc", width=1.8)))
                fig_trend.add_trace(go.Scatter(x=plot_slice.index, y=ts["mm50"].loc[plot_slice.index], name="MM50", line=dict(color=COLORS["accent"], width=1.3)))
                fig_trend.add_trace(go.Scatter(x=plot_slice.index, y=ts["mm200"].loc[plot_slice.index], name="MM200", line=dict(color=COLORS["median"], width=1.6)))
                st.plotly_chart(
                    style_fig(fig_trend, yaxis_title=f"Prix ({cur})",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)),
                    use_container_width=True,
                )

            with g2:
                vr_slice = ts["vol_ratio_series"].tail(504)
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=vr_slice.index, y=vr_slice, name="Vol 20j / Vol longue",
                    line=dict(color=COLORS["accent"], width=1.6), fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
                ))
                fig_vol.add_hline(y=1.0, line=dict(color=COLORS["muted"], dash="dot", width=1))
                fig_vol.add_hline(y=1.35, line=dict(color=COLORS["down"], dash="dash", width=1.4),
                                  annotation_text="Seuil Stress (1.35x)", annotation_position="top left")
                st.plotly_chart(style_fig(fig_vol, yaxis_title="Ratio de volatilité", showlegend=False), use_container_width=True)

            constraint = (
                f"Plafonné par votre limite max par ligne ({max_pos_pct}%)."
                if ts["adj_weight"] > max_pos_pct / 100.0
                else (
                    f"Réduit par le régime de stress (facteur {ts['regime_mult']:.2f}x)."
                    if ts["vol_ratio"] > 1.35
                    else f"Contraint directement par la CVaR 95% ({ts['cvar_95'] * 100:.2f}% / jour)."
                )
            )
            st.caption(f"Contrainte active : {constraint}")

            with st.expander("Méthodologie Timing & Sizing"):
                st.latex(r"w_{\text{CVaR}} = \frac{\text{Budget de risque}_{1\text{j}}}{\text{CVaR}_{95\%}}, \qquad M_{12-1} = \frac{P_{t-21}}{P_{t-252}} - 1")
                st.caption("Le diagnostic combine 5 filtres objectifs ; il s'agit d'une lecture quantitative du régime de marché, pas d'un conseil en investissement personnalisé.")
        else:
            st.warning(f"Historique insuffisant pour {primary_ticker}.")

    # Onglet 3 : Monte Carlo
    with tab_mc:
        if has_history and ts is not None:
            if dist_mode.startswith("Student"):
                df_t, loc_t, scale_t = student_t.fit(log_ret.values)
                df_t = max(float(df_t), 2.5)
                daily_sim = student_t.rvs(df_t, loc=loc_t, scale=scale_t, size=(forecast_days, n_simulations))
            else:
                shocks = np.random.normal(0, 1, (forecast_days, n_simulations))
                daily_sim = drift_daily + sigma_daily * shocks

            cum_log = np.vstack([np.zeros((1, n_simulations)), np.cumsum(daily_sim, axis=0)])
            paths = s0 * np.exp(cum_log)
            final_prices = paths[-1]

            p5, p50, p95 = np.percentile(final_prices, [5, 50, 95])
            n_pos = int(np.sum(final_prices > s0))
            n_neg = int(n_simulations - n_pos)
            prob_bull = (n_pos / n_simulations) * 100
            prob_bear = 100.0 - prob_bull

            # Plus-values / moins-values sur la position dimensionnee
            invest_base = ts["invest_base"]
            pos_label = f"{ts['n_shares']} actions ({cur}{invest_base:,.0f})" if ts["n_shares"] > 0 else f"position cible ({cur}{invest_base:,.0f})"
            pnl_p5 = invest_base * (p5 / s0 - 1)
            pnl_p50 = invest_base * (p50 / s0 - 1)
            pnl_p95 = invest_base * (p95 / s0 - 1)
            avg_gain = float(np.mean(final_prices[final_prices > s0] / s0 - 1) * invest_base) if n_pos > 0 else 0.0
            avg_loss = float(np.mean(final_prices[final_prices <= s0] / s0 - 1) * invest_base) if n_neg > 0 else 0.0
            rr_ratio = abs(p95 - s0) / max(abs(s0 - p5), 1e-6)

            # Lecture dynamique Monte Carlo
            if prob_bull >= 60 and pnl_p50 > 0:
                mc_badge, mc_col = "Asymétrie favorable", COLORS["up"]
            elif prob_bull >= 48:
                mc_badge, mc_col = "Distribution équilibrée", COLORS["median"]
            else:
                mc_badge, mc_col = "Biais baissier", COLORS["down"]

            mc_text = (
                f"Sur {n_simulations:,} scénarios à {forecast_days} jours ouvrés ({dist_mode}), "
                f"<strong>{n_pos:,} trajectoires terminent en hausse ({prob_bull:.1f}%)</strong> contre "
                f"<strong>{n_neg:,} en baisse ({prob_bear:.1f}%)</strong>. "
                f"Sur votre {pos_label}, la plus-value médiane ressort à <strong>{fmt_pnl(pnl_p50, cur)} ({(p50 / s0 - 1) * 100:+.1f}%)</strong>, "
                f"avec un scénario défavorable P5 à <strong>{fmt_pnl(pnl_p5, cur)}</strong> et un scénario haut P95 à <strong>{fmt_pnl(pnl_p95, cur)}</strong> "
                f"(ratio potentiel haut / baisse P5 : {rr_ratio:.2f}x)."
            )
            render_signal_card(mc_badge, mc_col, mc_text, right_meta=f"Base P&L : {pos_label}")

            # Ligne 1 : Trajectoires & parametres
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Trajectoires positives", f"{n_pos:,}", f"{prob_bull:.1f}% des scénarios")
            m2.metric("Trajectoires négatives", f"{n_neg:,}", f"-{prob_bear:.1f}% des scénarios", delta_color="inverse")
            m3.metric("Prix médian (P50)", f"{cur}{p50:.2f}", f"{(p50 / s0 - 1) * 100:+.1f}%")
            m4.metric("Drift annuel (μ)", f"{mu_ann * 100:+.1f}%")
            m5.metric("Volatilité (σ)", f"{sigma_ann * 100:.1f}%")

            # Ligne 2 : Plus-values / moins-values estimees
            p_c1, p_c2, p_c3, p_c4 = st.columns(4)
            p_c1.metric("P&L Médian (P50)", fmt_pnl(pnl_p50, cur), f"{(p50 / s0 - 1) * 100:+.1f}%")
            p_c2.metric("Scénario bas (P5)", fmt_pnl(pnl_p5, cur), f"{(p5 / s0 - 1) * 100:+.1f}%")
            p_c3.metric("Scénario haut (P95)", fmt_pnl(pnl_p95, cur), f"{(p95 / s0 - 1) * 100:+.1f}%")
            p_c4.metric("Gain moyen / Perte moy.", f"{fmt_pnl(avg_gain, cur)}", f"vs {fmt_pnl(avg_loss, cur)}")

            c1, c2 = st.columns([3, 2])
            with c1:
                t_axis = np.arange(forecast_days + 1, dtype=float)
                sample = paths[:, :min(n_simulations, 120)]
                bull_mask = sample[-1] >= s0

                fig_paths = go.Figure()
                for mask, rgba in [(bull_mask, "rgba(16,185,129,0.12)"), (~bull_mask, "rgba(239,68,68,0.12)")]:
                    bx, by = batch_paths(sample[:, mask], t_axis)
                    fig_paths.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color=rgba, width=1),
                                                   showlegend=False, hoverinfo="skip"))

                for q, label, color, dash in [
                    (95, "P95", COLORS["up"], "dash"),
                    (50, "Médiane", COLORS["median"], "solid"),
                    (5, "P5", COLORS["down"], "dash"),
                ]:
                    fig_paths.add_trace(go.Scatter(
                        x=t_axis, y=np.percentile(paths, q, axis=1), mode="lines",
                        line=dict(color=color, width=2 if q == 50 else 1.4, dash=dash), name=label,
                    ))

                st.plotly_chart(
                    style_fig(fig_paths, xaxis_title="Jours ouvrés", yaxis_title=f"Prix ({cur})",
                              legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(15,22,35,0.85)")),
                    use_container_width=True,
                )

            with c2:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=final_prices[final_prices >= s0], nbinsx=30, marker_color=COLORS["up"], opacity=0.7))
                fig_hist.add_trace(go.Histogram(x=final_prices[final_prices < s0], nbinsx=30, marker_color=COLORS["down"], opacity=0.7))
                fig_hist.add_vline(x=s0, line_color=COLORS["muted"], line_dash="dash", line_width=1)
                fig_hist.add_vline(x=p50, line_color=COLORS["median"], line_width=2, annotation_text=f"P50 {cur}{p50:.1f}")
                st.plotly_chart(style_fig(fig_hist, barmode="overlay", xaxis_title=f"Prix terminal ({cur})", showlegend=False),
                                use_container_width=True)

            with st.expander("Méthodologie stochastique"):
                st.latex(r"S_t = S_0 \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma W_t\right)")
                st.caption("Ces projections reflètent la distribution historique sur la fenêtre sélectionnée et ne garantissent pas les performances futures.")
        else:
            st.warning(f"Historique insuffisant pour {primary_ticker}.")

    # Onglet 4 : Risque
    with tab_risk:
        if has_history:
            returns = log_ret.values
            conf = st.slider("Niveau de confiance", 0.90, 0.99, 0.95, 0.01)
            r_stats = compute_risk_stats(returns, conf=conf)

            running_max = hist_prices.cummax()
            drawdown = (hist_prices - running_max) / running_max
            max_dd = float(drawdown.min())

            tail_badge = "Queues épaisses marquées" if r_stats["kurt"] > 3 else "Distribution modérée"
            tail_col = COLORS["down"] if r_stats["kurt"] > 3 else COLORS["accent"]
            risk_txt = (
                f"Dans les {(1 - conf) * 100:.0f}% pires séances, la perte moyenne quotidienne (CVaR) atteint "
                f"<strong>{r_stats['cvar'] * 100:.2f}%</strong> ({cur}{s0 * r_stats['cvar']:.2f}/action). "
                f"Sur les {history_years} dernières années, le pire creux historique (Max Drawdown) a été de <strong>{max_dd * 100:.1f}%</strong>."
            )
            render_signal_card(tail_badge, tail_col, risk_txt, right_meta=f"Excess Kurtosis : {r_stats['kurt']:.2f}")

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Skewness", f"{r_stats['skew']:.3f}")
            r2.metric("Excess Kurtosis", f"{r_stats['kurt']:.3f}")
            r3.metric(f"VaR CF {conf * 100:.0f}% (1j)", f"{r_stats['var_cf'] * 100:.2f}%", f"-{cur}{s0 * r_stats['var_cf']:.2f}")
            r4.metric(f"CVaR {conf * 100:.0f}% (1j)", f"{r_stats['cvar'] * 100:.2f}%", f"-{cur}{s0 * r_stats['cvar']:.2f}")
            r5.metric("Max Drawdown", f"{max_dd * 100:.1f}%")

            fig_var = go.Figure(go.Histogram(x=returns * 100, nbinsx=80, marker_color=COLORS["accent"], opacity=0.65))
            fig_var.add_vline(x=-r_stats["var_cf"] * 100, line_color=COLORS["down"], line_width=2, annotation_text=f"VaR {r_stats['var_cf'] * 100:.2f}%")
            fig_var.add_vline(x=-r_stats["cvar"] * 100, line_color="#fb7185", line_width=2, line_dash="dash", annotation_text=f"CVaR {r_stats['cvar'] * 100:.2f}%")
            st.plotly_chart(style_fig(fig_var, xaxis_title="Rendement journalier (%)", yaxis_title="Fréquence"), use_container_width=True)

            with st.expander("Formule Cornish-Fisher"):
                st.latex(r"z_c = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2")

    # Onglet 5 : Benchmark
    with tab_bench:
        if has_history:
            s_bench = fetch_close(macro_ticker, years=history_years)
            df_pair = pd.DataFrame({primary_ticker: hist_prices, macro_ticker: s_bench}).dropna()

            if not df_pair.empty:
                df_rebased = (df_pair / df_pair.iloc[0]) * 100
                ret_pair = df_pair.pct_change().dropna()
                corr_ret = float(ret_pair[primary_ticker].corr(ret_pair[macro_ticker]))
                rolling_corr = ret_pair[primary_ticker].rolling(63).corr(ret_pair[macro_ticker]).dropna()
                corr_63 = float(rolling_corr.iloc[-1]) if not rolling_corr.empty else np.nan
                var_b = float(ret_pair[macro_ticker].var())
                beta = float(ret_pair[primary_ticker].cov(ret_pair[macro_ticker]) / var_b) if var_b > 0 else np.nan

                beta_badge = "Sensibilité amplifiée" if beta > 1.25 else ("Profil défensif" if beta < 0.8 else "Sensibilité proche du marché")
                beta_col = COLORS["median"] if beta > 1.25 else COLORS["accent"]
                bench_txt = (
                    f"Avec un Bêta de <strong>{beta:.2f}</strong> face à {macro_ticker}, une variation de ±1% de l'indice s'accompagne en moyenne de "
                    f"<strong>±{beta:.2f}%</strong> sur {primary_ticker} (corrélation des rendements : {corr_ret:.2f})."
                )
                render_signal_card(beta_badge, beta_col, bench_txt, right_meta=f"Benchmark : {macro_ticker}")

                b1, b2, b3 = st.columns(3)
                b1.metric("Corrélation rendements", f"{corr_ret:.3f}")
                b2.metric("Corrélation 63j (3M)", f"{corr_63:.3f}" if pd.notnull(corr_63) else "—")
                b3.metric(f"Bêta vs {macro_ticker}", f"{beta:.2f}")

                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(x=df_rebased.index, y=df_rebased[primary_ticker], name=primary_ticker, line=dict(color=COLORS["up"], width=1.6)))
                fig_corr.add_trace(go.Scatter(x=df_rebased.index, y=df_rebased[macro_ticker], name=macro_ticker, line=dict(color=COLORS["median"], width=1.6)))
                st.plotly_chart(style_fig(fig_corr, yaxis_title="Base 100"), use_container_width=True)

if st.session_state.page == "home":
    render_home()
else:
    render_terminal()

