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

# Theme CSS
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
    font-size: 1.35rem;
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

# Statistiques de risque partagees
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
            fmt_num = lambda x: f"{x:.2f}" if pd.notnull(x) else "—"
            formatters = {"Cap. ($B)": lambda x: f"{x:.1f}B" if pd.notnull(x) else "—", **{c: fmt_num for c in ratio_cols}}
            st.dataframe(df_metrics.set_index("Ticker").style.format(formatters), use_container_width=True)

            col_ctrl, col_chart = st.columns([0.45, 1.55], gap="large")
            with col_ctrl:
                selected_ratio = st.selectbox("Multiple à comparer", ratio_cols)
                valid_vals = df_metrics[selected_ratio].dropna()
                if not valid_vals.empty:
                    med_val = float(valid_vals.median())
                    target_val = df_metrics.loc[df_metrics["Ticker"] == primary_ticker, selected_ratio].values
                    st.metric("Médiane du groupe", f"{med_val:.2f}x")
                    if len(target_val) and pd.notnull(target_val[0]):
                        spread = (target_val[0] / med_val - 1) * 100
                        st.metric(f"{primary_ticker} vs Médiane", f"{target_val[0]:.2f}x", f"{spread:+.1f}%", delta_color="inverse")

            with col_chart:
                df_plot = df_metrics.dropna(subset=[selected_ratio])
                if not df_plot.empty:
                    med_val = float(df_plot[selected_ratio].median())
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

    # Onglet 2 : Timing & Sizing
    with tab_ts:
        if has_history:
            mm50 = hist_prices.rolling(50).mean()
            mm200 = hist_prices.rolling(200).mean()
            has_mm200 = mm200.dropna().shape[0] >= 21

            if has_mm200:
                mm200_now = float(mm200.iloc[-1])
                dist_mm200 = (s0 / mm200_now - 1) * 100
                mm200_slope = (mm200_now / float(mm200.iloc[-21]) - 1) * 100
                trend_label = "Haussière" if (s0 >= mm200_now and mm200_slope >= 0) else (
                    "Transition" if (s0 >= mm200_now or mm200_slope >= 0) else "Baissière"
                )
            else:
                mm200_now, dist_mm200, mm200_slope, trend_label = np.nan, np.nan, np.nan, "Historique < 200j"

            # Momentum 12-1 mois
            if len(hist_prices) >= 253:
                mom_12_1 = (float(hist_prices.iloc[-22]) / float(hist_prices.iloc[-253]) - 1) * 100
            else:
                mom_12_1 = np.nan

            # Regime de volatilite (20j vs 252j)
            vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
            long_win = min(252, max(60, len(log_ret) // 2))
            vol_long = log_ret.rolling(long_win).std() * np.sqrt(252)
            vol_ratio_series = (vol_20 / vol_long).dropna()
            vol_ratio = float(vol_ratio_series.iloc[-1]) if not vol_ratio_series.empty else 1.0

            if vol_ratio > 1.35:
                vol_regime = "Stress"
            elif vol_ratio < 0.85:
                vol_regime = "Calme"
            else:
                vol_regime = "Normal"

            # Sizing par CVaR 95% + ajustement de regime
            risk_95 = compute_risk_stats(log_ret.values, conf=0.95)
            cvar_95 = risk_95["cvar"]
            raw_weight = (risk_budget_pct / 100.0) / cvar_95
            regime_mult = (1.0 / vol_ratio) if vol_ratio > 1.35 else 1.0
            adj_weight = raw_weight * regime_mult
            final_weight = min(adj_weight, max_pos_pct / 100.0)

            target_amount = portfolio_capital * final_weight
            n_shares = int(target_amount // s0) if s0 > 0 else 0
            real_amount = n_shares * s0
            tail_loss_est = real_amount * cvar_95

            # Plan d'entree
            if trend_label == "Haussière" and vol_regime != "Stress":
                tranches = "1 à 2 tranches"
            elif trend_label == "Baissière" or vol_regime == "Stress":
                tranches = "3 tranches espacées (prudence)"
            else:
                tranches = "2 tranches"

            # Ligne 1 : Sizing
            st.markdown("#### Dimensionnement de position")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Poids max conseillé", f"{final_weight * 100:.1f}%", f"Brut CVaR : {raw_weight * 100:.1f}%")
            s2.metric("Montant cible", f"{cur}{target_amount:,.0f}")
            s3.metric("Actions (entier)", f"{n_shares}", f"{cur}{real_amount:,.0f} engagés")
            s4.metric("Perte extrême 1j (CVaR)", f"-{cur}{tail_loss_est:,.0f}", f"-{(tail_loss_est / portfolio_capital) * 100:.2f}% portef.")
            s5.metric("Entrée suggérée", tranches)

            # Ligne 2 : Timing
            st.markdown("#### Régime & Timing")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Tendance (MM200)", trend_label, f"{dist_mm200:+.1f}% vs MM200" if pd.notnull(dist_mm200) else "N/A")
            t2.metric("Pente MM200 (20j)", f"{mm200_slope:+.2f}%" if pd.notnull(mm200_slope) else "—")
            t3.metric("Momentum 12-1M", f"{mom_12_1:+.1f}%" if pd.notnull(mom_12_1) else "Historique < 1A")
            t4.metric("Régime Volatilité", vol_regime, f"Ratio 20j/{long_win}j : {vol_ratio:.2f}x")

            # Graphiques Timing
            g1, g2 = st.columns([1.45, 1])
            with g1:
                plot_slice = hist_prices.tail(504)
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=plot_slice.index, y=plot_slice, name=primary_ticker, line=dict(color="#f8fafc", width=1.8)))
                fig_trend.add_trace(go.Scatter(x=plot_slice.index, y=mm50.loc[plot_slice.index], name="MM50", line=dict(color=COLORS["accent"], width=1.3)))
                fig_trend.add_trace(go.Scatter(x=plot_slice.index, y=mm200.loc[plot_slice.index], name="MM200", line=dict(color=COLORS["median"], width=1.6)))
                st.plotly_chart(
                    style_fig(fig_trend, yaxis_title=f"Prix ({cur})",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)),
                    use_container_width=True,
                )

            with g2:
                vr_slice = vol_ratio_series.tail(504)
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=vr_slice.index, y=vr_slice, name="Vol 20j / Vol longue",
                    line=dict(color=COLORS["accent"], width=1.6), fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
                ))
                fig_vol.add_hline(y=1.0, line=dict(color=COLORS["muted"], dash="dot", width=1))
                fig_vol.add_hline(y=1.35, line=dict(color=COLORS["down"], dash="dash", width=1.4),
                                  annotation_text="Seuil Stress (1.35x)", annotation_position="top left")
                st.plotly_chart(
                    style_fig(fig_vol, yaxis_title="Ratio de volatilité", showlegend=False),
                    use_container_width=True,
                )

            # Synthese factuelle des contraintes
            constraint = (
                f"Plafonné par votre limite max par ligne ({max_pos_pct}%)."
                if adj_weight > max_pos_pct / 100.0
                else (
                    f"Réduit par le régime de stress (facteur {regime_mult:.2f}x)."
                    if vol_ratio > 1.35
                    else f"Contraint directement par la CVaR 95% ({cvar_95 * 100:.2f}% / jour)."
                )
            )
            st.caption(f"Contrainte active : {constraint}")

            with st.expander("Méthodologie Timing & Sizing"):
                st.latex(r"w_{\text{CVaR}} = \frac{\text{Budget de risque}_{1\text{j}}}{\text{CVaR}_{95\%}}, \qquad M_{12-1} = \frac{P_{t-21}}{P_{t-252}} - 1")
                st.caption("Lorsque le ratio de volatilité 20j / 252j dépasse 1.35, le poids brut est divisé par ce ratio afin d'amortir les régimes de panique.")
        else:
            st.warning(f"Historique insuffisant pour {primary_ticker}.")

    # Onglet 3 : Monte Carlo
    with tab_mc:
        if has_history:
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
            prob_bull = float(np.mean(final_prices > s0) * 100)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Drift annuel (μ)", f"{mu_ann * 100:+.1f}%")
            m2.metric("Volatilité (σ)", f"{sigma_ann * 100:.1f}%")
            m3.metric("Médiane (P50)", f"{cur}{p50:.2f}", f"{(p50 / s0 - 1) * 100:+.1f}%")
            m4.metric("Intervalle 90%", f"{cur}{p5:.0f} – {cur}{p95:.0f}")
            m5.metric("Prob. hausse", f"{prob_bull:.1f}%")

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
