import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import sys

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=10 * 365)
    end_date = now + datetime.timedelta(days=1)

    print("=" * 68)
    print(f"  ANALYSE QUANTITATIVE TEMPS RÉEL : {ticker}")
    print("=" * 68)

    t_obj = yf.Ticker(ticker)
    data = t_obj.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=True)
    prices = data['Close'].dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # Injection du flux intraday du jour (ex: 4 novembre) si disponible
    is_live = False
    try:
        intra = t_obj.history(period="5d", interval="15m")
        if not intra.empty:
            last_val = float(intra['Close'].dropna().iloc[-1])
            last_ts = pd.to_datetime(intra.index[-1]).tz_localize(None)
            if last_ts.date() > prices.index[-1].date():
                prices = pd.concat([prices, pd.Series([last_val], index=[pd.Timestamp(last_ts.date())])])
                is_live = True
            elif last_ts.date() == prices.index[-1].date():
                prices.iloc[-1] = last_val
                is_live = (last_ts.date() == now.date())
    except Exception:
        pass

    last_date_str = prices.index[-1].strftime("%d/%m/%Y")
    s0 = float(prices.iloc[-1])
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu_daily = float(log_ret.mean())
    sigma_daily = float(log_ret.std())

    mu_ann = mu_daily * 252
    sigma_ann = sigma_daily * np.sqrt(252)

    print(f"[-] Dernière cotation intégrée : {last_date_str} ({'LIVE INTRADAY' if is_live else 'CLÔTURE'})")
    print(f"[-] Prix actuel (S0)           : ${s0:.2f}")
    print(f"[-] Rendement annuel moyen (μ) : {mu_ann*100:.2f}%")
    print(f"[-] Volatilité annuelle (σ)    : {sigma_ann*100:.2f}%")

    n_sim = 10000
    days = 252
    drift = (mu_daily - 0.5 * (sigma_daily ** 2))
    shocks = np.random.normal(0, 1, (days, n_sim))
    log_paths = np.vstack([np.zeros((1, n_sim)), np.cumsum(drift + sigma_daily * shocks, axis=0)])
    paths = s0 * np.exp(log_paths)

    final = paths[-1, :]
    p5, p50, p95 = np.percentile(final, [5, 50, 95])
    prob_gain = (np.sum(final > s0) / n_sim) * 100

    print("\n" + "=" * 68)
    print(f"  SIMULATION MONTE CARLO ({n_sim:,} chemins à partir du {last_date_str})")
    print("=" * 68)
    print(f"[-] Trajectoire Médiane (50%)  : ${p50:.2f} ({((p50-s0)/s0)*100:+.2f}%)")
    print(f"[-] Scénario Baissier (5%)     : ${p5:.2f} ({((p5-s0)/s0)*100:+.2f}%)")
    print(f"[-] Scénario Haussier (95%)    : ${p95:.2f} ({((p95-s0)/s0)*100:+.2f}%)")
    print(f"[-] Probabilité de gain à 1 an : {prob_gain:.1f}%")
    print(f"[-] Value-at-Risk (95% VaR)    : -${s0 - p5:.2f}")
    print("=" * 68)

if __name__ == "__main__":
    main()
