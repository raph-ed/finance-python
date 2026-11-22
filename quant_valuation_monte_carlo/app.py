import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, t as student_t, skew, kurtosis
import datetime

# Configuration page
st.set_page_config(
    page_title="Quant Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme sobre
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp {
    background-color: #0b0e11;
    color: #d1d5db;
}
section[data-testid="stSidebar"] {
    background-color: #11151c;
    border-right: 1px solid #1f2630;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.35rem;
    font-weight: 600;
}
div[data-testid="metric-container"] {
    background-color: #11151c;
    border: 1px solid #1f2630;
    padding: 12px 16px;
    border-radius: 3px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #1f2630;
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    color: #8b949e;
    background: transparent;
    border: none;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    color: #f3f4f6 !important;
    background-color: #161b22 !important;
    border-bottom: 2px solid #f59e0b !important;
}
.ticker-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid #1f2630;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

COLORS = {
    "bg": "#0b0e11",
    "panel": "#11151c",
    "grid": "#1c222b",
    "accent": "#3b82f6",
    "median": "#f59e0b",
    "up": "#10b981",
    "down": "#ef4444",
}

def style_fig(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(family="Inter, sans-serif", size=12, color="#c9d1d9"),
        margin=dict(l=16, r=16, t=24, b=16),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor="#2a323d"),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor="#2a323d"),
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

# Donnees marche
@st.cache_data(ttl=1800, show_spinner=False)
def search_securities(query: str):
    if not query or len(query.strip()) < 2:
        return []
    try:
        quotes = yf.Search(query, max_results=10).quotes
        return [
            (
                q.get("symbol", ""),
                f"{q.get('symbol', '')} — {q.get('shortname') or q.get('longname') or q.get('symbol', '')} [{q.get('exchange', '')}]",
            )
            for q in quotes if q.get("symbol")
        ]
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_close(ticker: str, years: int = 5) -> pd.Series:
    start = datetime.datetime.now() - datetime.timedelta(days=years * 365 + 30)
    data = yf.download(ticker, start=start, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    prices = data["Close"] if "Close" in data else data.iloc[:, 0]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices = prices.dropna()
    return pd.Series(prices.values.ravel(), index=prices.index, name=ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(tickers: tuple) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap = info.get("marketCap")
            rows.append({
                "Ticker": t,
                "Nom": info.get("shortName", t),
                "Cap. ($B)": cap / 1e9 if cap else np.nan,
                "P/E": info.get("trailingPE", np.nan),
                "Fwd P/E": info.get("forwardPE", np.nan),
                "EV/EBITDA": info.get("enterpriseToEbitda", np.nan),
                "P/S": info.get("priceToSalesTrailing12Months", np.nan),
                "P/B": info.get("priceToBook", np.nan),
            })
        except Exception:
            rows.append({"Ticker": t, "Nom": t})
    return pd.DataFrame(rows)

# Sidebar
st.sidebar.markdown("**Paramètres**")
search_term = st.sidebar.text_input("Actif (ticker ou nom)", value="NVDA")
search_results = search_securities(search_term)

if search_results:
    primary_ticker = st.sidebar.selectbox(
        "Sélection", options=search_results, format_func=lambda x: x[1]
    )[0]
else:
    primary_ticker = search_term.upper().strip() or "QQQ"

default_peers = "SPY, IWM, DIA" if any(k in primary_ticker for k in ("QQQ", "IXIC")) else "MSFT, GOOGL, AMZN, AAPL"
peers_input = st.sidebar.text_input("Comparables", value=default_peers)
peer_list = [p.strip().upper() for p in peers_input.split(",") if p.strip()]

st.sidebar.divider()
n_simulations = st.sidebar.slider("Simulations Monte Carlo", 500, 10000, 2500, 500)
forecast_days = st.sidebar.slider("Horizon (jours ouvrés)", 21, 504, 252, 21)
history_years = st.sidebar.selectbox("Historique (années)", [1, 3, 5, 10], index=2)
macro_ticker = st.sidebar.text_input("Benchmark", value="^GSPC")

# Chargement centralise
hist_prices = fetch_close(primary_ticker, years=history_years)
has_history = len(hist_prices) > 30

if has_history:
    log_ret = np.log(hist_prices / hist_prices.shift(1)).dropna()
    drift_daily = float(log_ret.mean())  # deja egal a mu - 0.5 * sigma^2
    sigma_daily = float(log_ret.std())
    mu_ann = (drift_daily + 0.5 * sigma_daily**2) * 252
    sigma_ann = sigma_daily * np.sqrt(252)
    s0 = float(hist_prices.iloc[-1])

# En-tete compact
price_badge = f"<span style='font-family:JetBrains Mono;color:#9ca3af;font-size:1.1rem;'>${s0:,.2f}</span>" if has_history else ""
st.markdown(
    f"<div class='ticker-header'><span style='font-size:1.4rem;font-weight:600;color:#fff;'>{primary_ticker}</span>{price_badge}</div>",
    unsafe_allow_html=True,
)

tab_val, tab_mc, tab_risk, tab_bench = st.tabs(["Multiples", "Monte Carlo", "Risque (VaR/CVaR)", "Benchmark"])

# Onglet 1 : Multiples
with tab_val:
    all_tickers = tuple([primary_ticker] + [p for p in peer_list if p != primary_ticker])
    df_metrics = fetch_fundamentals(all_tickers)
    ratio_cols = ["P/E", "Fwd P/E", "EV/EBITDA", "P/S", "P/B"]

    if not df_metrics.empty and "P/E" in df_metrics.columns:
        col_tbl, col_chart = st.columns([1.1, 1])
        fmt_num = lambda x: f"{x:.2f}" if pd.notnull(x) else "—"
        formatters = {"Cap. ($B)": lambda x: f"{x:.1f}B" if pd.notnull(x) else "—", **{c: fmt_num for c in ratio_cols}}

        with col_tbl:
            st.dataframe(df_metrics.set_index("Ticker").style.format(formatters), use_container_width=True)
            selected_ratio = st.selectbox("Multiple affiché", ratio_cols)

        with col_chart:
            df_plot = df_metrics.dropna(subset=[selected_ratio])
            if not df_plot.empty:
                med_val = df_plot[selected_ratio].median()
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
                fig_bar.add_hline(y=med_val, line=dict(color=COLORS["median"], dash="dash", width=1.5),
                                  annotation_text=f"Médiane {med_val:.1f}x", annotation_position="top right")
                st.plotly_chart(style_fig(fig_bar, yaxis_title=selected_ratio), use_container_width=True)

# Onglet 2 : Monte Carlo
with tab_mc:
    if has_history:
        shocks = np.random.normal(0, 1, (forecast_days, n_simulations))
        cum_log = np.vstack([np.zeros((1, n_simulations)), np.cumsum(drift_daily + sigma_daily * shocks, axis=0)])
        paths = s0 * np.exp(cum_log)
        final_prices = paths[-1]
        p5, p50, p95 = np.percentile(final_prices, [5, 50, 95])
        prob_bull = float(np.mean(final_prices > s0) * 100)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Drift annuel (μ)", f"{mu_ann * 100:+.1f}%")
        m2.metric("Volatilité (σ)", f"{sigma_ann * 100:.1f}%")
        m3.metric("Médiane (P50)", f"${p50:.2f}", f"{(p50 / s0 - 1) * 100:+.1f}%")
        m4.metric("Intervalle 90%", f"${p5:.0f} – ${p95:.0f}")
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
                style_fig(fig_paths, xaxis_title="Jours ouvrés", yaxis_title="Prix ($)",
                          legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(17,21,28,0.8)")),
                use_container_width=True,
            )

        with c2:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=final_prices[final_prices >= s0], nbinsx=30, marker_color=COLORS["up"], opacity=0.7))
            fig_hist.add_trace(go.Histogram(x=final_prices[final_prices < s0], nbinsx=30, marker_color=COLORS["down"], opacity=0.7))
            fig_hist.add_vline(x=s0, line_color="#9ca3af", line_dash="dash", line_width=1)
            fig_hist.add_vline(x=p50, line_color=COLORS["median"], line_width=2, annotation_text=f"P50 ${p50:.1f}")
            st.plotly_chart(style_fig(fig_hist, barmode="overlay", xaxis_title="Prix terminal ($)", showlegend=False),
                            use_container_width=True)

        with st.expander("Méthodologie GBM"):
            st.latex(r"S_t = S_0 \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma W_t\right)")
    else:
        st.warning(f"Historique insuffisant pour {primary_ticker}.")

# Onglet 3 : VaR & CVaR
with tab_risk:
    if has_history:
        returns = log_ret.values
        conf = st.slider("Niveau de confiance", 0.90, 0.99, 0.95, 0.01)
        alpha = 1.0 - conf
        s_val, k_val = float(skew(returns)), float(kurtosis(returns))
        z = norm.ppf(alpha)
        z_cf = z + (z**2 - 1) * s_val / 6 + (z**3 - 3 * z) * k_val / 24 - (2 * z**3 - 5 * z) * (s_val**2) / 36

        var_cf = -(drift_daily + z_cf * sigma_daily)
        var_hist = float(-np.percentile(returns, alpha * 100))
        tail = returns[returns <= -var_hist]
        cvar_hist = float(-tail.mean()) if len(tail) else var_hist

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Skewness", f"{s_val:.3f}")
        r2.metric("Excess Kurtosis", f"{k_val:.3f}")
        r3.metric(f"VaR CF {conf * 100:.0f}% (1j)", f"{var_cf * 100:.2f}%", f"-${s0 * var_cf:.2f}")
        r4.metric(f"CVaR {conf * 100:.0f}% (1j)", f"{cvar_hist * 100:.2f}%", f"-${s0 * cvar_hist:.2f}")

        fig_var = go.Figure(go.Histogram(x=returns * 100, nbinsx=80, marker_color=COLORS["accent"], opacity=0.65))
        fig_var.add_vline(x=-var_cf * 100, line_color=COLORS["down"], line_width=2, annotation_text=f"VaR {var_cf * 100:.2f}%")
        fig_var.add_vline(x=-cvar_hist * 100, line_color="#fb7185", line_width=2, line_dash="dash", annotation_text=f"CVaR {cvar_hist * 100:.2f}%")
        st.plotly_chart(style_fig(fig_var, xaxis_title="Rendement journalier (%)", yaxis_title="Fréquence"), use_container_width=True)

        with st.expander("Formule Cornish-Fisher"):
            st.latex(r"z_c = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2")

# Onglet 4 : Benchmark
with tab_bench:
    if has_history:
        s_bench = fetch_close(macro_ticker, years=history_years)
        df_pair = pd.DataFrame({primary_ticker: hist_prices, macro_ticker: s_bench}).dropna()

        if not df_pair.empty:
            df_rebased = (df_pair / df_pair.iloc[0]) * 100
            ret_pair = df_pair.pct_change().dropna()
            corr_ret = ret_pair[primary_ticker].corr(ret_pair[macro_ticker])
            corr_lvl = df_pair[primary_ticker].corr(df_pair[macro_ticker])
            var_b = float(ret_pair[macro_ticker].var())
            beta = float(ret_pair[primary_ticker].cov(ret_pair[macro_ticker]) / var_b) if var_b > 0 else np.nan

            b1, b2, b3 = st.columns(3)
            b1.metric("Corrélation rendements", f"{corr_ret:.3f}")
            b2.metric("Corrélation niveaux", f"{corr_lvl:.3f}")
            b3.metric(f"Bêta vs {macro_ticker}", f"{beta:.2f}")

            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=df_rebased.index, y=df_rebased[primary_ticker], name=primary_ticker, line=dict(color=COLORS["up"], width=1.6)))
            fig_corr.add_trace(go.Scatter(x=df_rebased.index, y=df_rebased[macro_ticker], name=macro_ticker, line=dict(color=COLORS["median"], width=1.6)))
            st.plotly_chart(style_fig(fig_corr, yaxis_title="Base 100"), use_container_width=True)

