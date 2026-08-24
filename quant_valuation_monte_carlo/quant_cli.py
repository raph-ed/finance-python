import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import sys

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print("=" * 65)
    print(f"  ANALYSE QUANTITATIVE RAPIDE : {ticker}")
    print("=" * 65)
    
    # 1. Fetch data
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=10*365)
    print(f"[*] Téléchargement de l'historique sur 10 ans pour {ticker}...")
    
    data = yf.download(ticker, start=start_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'] if 'Close' in data else data.iloc[:, 0]
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    prices = pd.Series(prices.dropna().values.ravel(), index=prices.dropna().index)
    
    s0 = float(prices.iloc[-1])
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu_daily = log_ret.mean()
    sigma_daily = log_ret.std()
    
    mu_ann = mu_daily * 252
    sigma_ann = sigma_daily * np.sqrt(252)
    
    print(f"[-] Prix actuel (S0)           : ${s0:.2f}")
    print(f"[-] Rendement annuel moyen (μ) : {mu_ann*100:.2f}%")
    print(f"[-] Volatilité annuelle (σ)    : {sigma_ann*100:.2f}%")
    
    # 2. Monte Carlo
    n_sim = 10000
    days = 252
    drift = (mu_daily - 0.5 * (sigma_daily ** 2))
    shocks = np.random.normal(0, 1, (days, n_sim))
    log_paths = np.vstack([np.zeros((1, n_sim)), np.cumsum(drift + sigma_daily * shocks, axis=0)])
    paths = s0 * np.exp(log_paths)
    
    final = paths[-1, :]
    p5 = np.percentile(final, 5)
    p50 = np.percentile(final, 50)
    p95 = np.percentile(final, 95)
    prob_gain = (np.sum(final > s0) / n_sim) * 100
    
    print("\n" + "=" * 65)
    print(f"  RÉSULTATS DE LA SIMULATION MONTE CARLO ({n_sim:,} chemins à 1 an)")
    print("=" * 65)
    print(f"[-] Trajectoire la plus probable (Médiane 50%) : ${p50:.2f} ({((p50-s0)/s0)*100:+.2f}%)")
    print(f"[-] Scénario baissier critique (Percentile 5%)   : ${p5:.2f} ({((p5-s0)/s0)*100:+.2f}%)")
    print(f"[-] Scénario haussier optimiste (Percentile 95%) : ${p95:.2f} ({((p95-s0)/s0)*100:+.2f}%)")
    print(f"[-] Probabilité de gain à 1 an                   : {prob_gain:.1f}%")
    print(f"[-] Value-at-Risk (95% VaR)                      : -${s0 - p5:.2f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
