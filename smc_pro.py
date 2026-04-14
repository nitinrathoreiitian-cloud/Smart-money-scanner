import yfinance as yf
import pandas as pd
import numpy as np
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────
@dataclass
class Analysis:
    ticker: str
    price: float
    score: float
    signal: str
    confidence: float
    setup_type: str
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    reasons: list = field(default_factory=list)

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = np.abs(df["High"] - df["Close"].shift())
    lc = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    df = df.copy()
    tr = atr(df)
    plus_dm = np.where((df["High"].diff() > df["Low"].diff()) & (df["High"].diff() > 0), df["High"].diff(), 0)
    minus_dm = np.where((df["Low"].diff() > df["High"].diff()) & (df["Low"].diff() > 0), df["Low"].diff(), 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / (tr + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / (tr + 1e-10))
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    return dx.rolling(period).mean()

def cmf(df, period=20):
    range_ = (df["High"] - df["Low"]).replace(0, 1e-10)
    mfv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / range_
    mfv = mfv * df["Volume"]
    return mfv.rolling(period).sum() / (df["Volume"].rolling(period).sum() + 1e-10)

def obv(df):
    return (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

# ─────────────────────────────────────────────
# SMC LOGIC
# ─────────────────────────────────────────────
def detect_bos(df):
    if len(df) < 22: return False
    recent_high = df["High"].rolling(20).max()
    return df["Close"].iloc[-1] > recent_high.iloc[-2]

def detect_choch(df):
    if len(df) < 22: return False
    recent_low = df["Low"].rolling(20).min()
    return df["Close"].iloc[-1] < recent_low.iloc[-2]

def liquidity_sweep(df):
    if len(df) < 6: return False
    prev_high = df["High"].iloc[-5:-1].max()
    return df["High"].iloc[-1] > prev_high and df["Close"].iloc[-1] < prev_high

# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────
def analyze_ticker(ticker, df):
    try:
        df = df.dropna()
        if len(df) < 100: return None

        price = df["Close"].iloc[-1]
        
        # Technicals
        df["RSI"] = rsi(df["Close"])
        df["ATR"] = atr(df)
        df["ADX"] = adx(df)
        df["CMF"] = cmf(df)
        df["OBV"] = obv(df)

        rsi_v, atr_v, adx_v, cmf_v = df["RSI"].iloc[-1], df["ATR"].iloc[-1], df["ADX"].iloc[-1], df["CMF"].iloc[-1]
        obv_trend = df["OBV"].iloc[-1] - df["OBV"].iloc[-10]

        ema20 = df["Close"].ewm(span=20).mean().iloc[-1]
        ema50 = df["Close"].ewm(span=50).mean().iloc[-1]
        ema200 = df["Close"].ewm(span=200).mean().iloc[-1]
        
        vol, avg_vol = df["Volume"].iloc[-1], df["Volume"].rolling(20).mean().iloc[-1]
        bos, sweep, choch = detect_bos(df), liquidity_sweep(df), detect_choch(df)

        score, reasons = 0, []
        if ema20 > ema50 > ema200: score += 15; reasons.append("Bullish Trend")
        if bos: score += 20; reasons.append("BOS Breakout")
        if sweep: score += 15; reasons.append("Liquidity Sweep")
        if rsi_v < 45: score += 10; reasons.append("Pullback Zone")
        if adx_v > 25: score += 10; reasons.append("Strong Momentum")
        if cmf_v > 0: score += 10; reasons.append("Accumulation")
        if obv_trend > 0: score += 10; reasons.append("OBV Rising")
        if vol > avg_vol * 1.3: score += 10; reasons.append("Volume Surge")

        signal = "STRONG BUY" if score >= 70 else "BUY" if score >= 50 else "WATCH"
        
        # Risk Management
        stop_loss = price - (atr_v * 1.5)
        risk = max(price - stop_loss, 0.01)
        t1, t2, t3 = price + risk, price + (risk * 2), price + (risk * 3)

        confidence = sum([20 for cond in [adx_v > 25, vol > avg_vol, cmf_v > 0, obv_trend > 0, bos] if cond])

        return Analysis(ticker, round(price,2), min(100, score), signal, confidence, 
                        "Trend Continuation" if bos else "Reversal" if sweep else "Mixed",
                        round(stop_loss,2), round(t1,2), round(t2,2), round(t3,2), reasons)
    except Exception:
        return None

# ─────────────────────────────────────────────
# SCANNER RUNNER
# ─────────────────────────────────────────────
def run_scan(tickers):
    print(f"--- Downloading data for {len(tickers)} symbols ---")
    
    # NO SESSION PASSED: Let yfinance handle impersonation internally
    data = yf.download(tickers, period="1y", group_by="ticker", progress=False)

    results = []
    
    def process(t):
        try:
            if len(tickers) > 1:
                if t not in data.columns.get_level_values(0): return None
                df_ticker = data[t]
            else:
                df_ticker = data
            return analyze_ticker(t, df_ticker)
        except: return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process, t) for t in tickers]
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:10]

# ─────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Clean list of 100 Tickers
    ticker_list = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "SQ",
        "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "PYPL", "AVGO", "ORCL",
        "ADBE", "CRM", "INTC", "CSCO", "TSM", "ASML", "BABA", "COST", "WMT", "DIS",
        "XOM", "CVX", "UNH", "PFE", "ABBV", "LLY", "MRK", "TMO", "PEP", "KO",
        "PG", "NKE", "MCD", "SBUX", "AMT", "PLTR", "SNOW", "UBER", "ABNB", "SHOP",
        "NET", "CRWD", "DDOG", "ZS", "PANW", "OKTA", "SE", "MELI", "RIVN", "LCID",
        "DKNG", "HOOD", "COIN", "MARA", "RIOT", "U", "AI", "SOFI", "AFRM", "UPST",
        "MRNA", "BNTX", "ZM", "DOCU", "PINS", "SNAP", "ROKU", "TTD", "ENPH", "SEDG",
        "F", "GM", "TM", "HMC", "CAT", "DE", "BA", "LMT", "RTX", "GE",
        "AAL", "DAL", "UAL", "LUV", "BKNG", "EXPE", "T", "VZ", "TMUS", "IBM","SNDK",
    ]

    start_time = time.time()
    picks = run_scan(ticker_list)

    print(f"\n🚀 TOP 10 SMART MONEY PICKS (Scan Time: {round(time.time()-start_time, 2)}s)")
    for r in picks:
        print(f"\n{r.ticker} — {r.signal} (Score: {r.score}%)")
        print(f"  Entry: ${r.price} | SL: ${r.stop_loss} | T1: ${r.target_1}")
        print(f"  Setup: {r.setup_type} | Reasons: {', '.join(r.reasons)}")
