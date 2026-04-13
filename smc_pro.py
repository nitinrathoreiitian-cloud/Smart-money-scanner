# smart_money_production.py

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
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = np.abs(df["High"] - df["Close"].shift())
    lc = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx(df, period=14):
    df = df.copy()
    df["TR"] = atr(df)
    df["+DM"] = np.where(df["High"].diff() > df["Low"].diff(), df["High"].diff(), 0)
    df["-DM"] = np.where(df["Low"].diff() > df["High"].diff(), df["Low"].diff(), 0)

    plus_di = 100 * (df["+DM"].rolling(period).mean() / df["TR"])
    minus_di = 100 * (df["-DM"].rolling(period).mean() / df["TR"])
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

def cmf(df, period=20):
    range_ = (df["High"] - df["Low"]).replace(0, 1e-10)
    mfv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / range_
    mfv = mfv * df["Volume"]
    return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum()

def obv(df):
    direction = np.sign(df["Close"].diff())
    return (direction * df["Volume"]).cumsum()

# ─────────────────────────────────────────────
# SMC LOGIC
# ─────────────────────────────────────────────
def detect_bos(df):
    recent_high = df["High"].rolling(20).max()
    return df["Close"].iloc[-1] > recent_high.iloc[-2]

def detect_choch(df):
    recent_low = df["Low"].rolling(20).min()
    return df["Close"].iloc[-1] < recent_low.iloc[-2]

def liquidity_sweep(df):
    prev_high = df["High"].iloc[-5:-1].max()
    return df["High"].iloc[-1] > prev_high and df["Close"].iloc[-1] < prev_high

# ─────────────────────────────────────────────
# ANALYSIS FUNCTION
# ─────────────────────────────────────────────
def analyze_ticker(ticker, df):
    try:
        df = df.dropna()
        if len(df) < 100:
            return None

        price = df["Close"].iloc[-1]

        # Indicators
        df["RSI"] = rsi(df["Close"])
        df["ATR"] = atr(df)
        df["ADX"] = adx(df)
        df["CMF"] = cmf(df)
        df["OBV"] = obv(df)

        rsi_v = df["RSI"].iloc[-1]
        atr_v = df["ATR"].iloc[-1]
        adx_v = df["ADX"].iloc[-1]
        cmf_v = df["CMF"].iloc[-1]
        obv_trend = df["OBV"].iloc[-1] - df["OBV"].iloc[-10]

        ema20 = df["Close"].ewm(span=20).mean().iloc[-1]
        ema50 = df["Close"].ewm(span=50).mean().iloc[-1]
        ema200 = df["Close"].ewm(span=200).mean().iloc[-1]

        vol = df["Volume"].iloc[-1]
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]

        # SMC
        bos = detect_bos(df)
        choch = detect_choch(df)
        sweep = liquidity_sweep(df)

        # SCORE
        score = 0
        reasons = []

        if ema20 > ema50 > ema200:
            score += 15
            reasons.append("EMA bullish")

        if bos:
            score += 20
            reasons.append("BOS breakout")

        if sweep:
            score += 15
            reasons.append("Liquidity sweep")

        if rsi_v < 40:
            score += 10
            reasons.append("RSI oversold")

        if adx_v > 25:
            score += 10
            reasons.append("Strong trend")

        if cmf_v > 0:
            score += 10
            reasons.append("Money inflow")

        if obv_trend > 0:
            score += 10
            reasons.append("OBV rising")

        if vol > avg_vol * 1.5:
            score += 10
            reasons.append("Volume surge")

        score = min(100, score)

        # SIGNAL
        if score >= 70:
            signal = "STRONG BUY"
        elif score >= 50:
            signal = "BUY"
        else:
            signal = "WATCH"

        # TRADE PLAN
        stop_loss = price - (atr_v * 1.5)
        risk = price - stop_loss

        t1 = price + risk * 1
        t2 = price + risk * 2
        t3 = price + risk * 3

        # SETUP TYPE
        if bos and ema20 > ema50:
            setup = "Trend Continuation"
        elif sweep:
            setup = "Reversal"
        elif choch:
            setup = "Early Reversal"
        else:
            setup = "Mixed"

        # CONFIDENCE
        confidence = 0
        if adx_v > 25: confidence += 20
        if vol > avg_vol: confidence += 20
        if cmf_v > 0: confidence += 20
        if obv_trend > 0: confidence += 20
        if bos: confidence += 20

        return Analysis(
            ticker, round(price,2), score, signal, confidence, setup,
            round(stop_loss,2), round(t1,2), round(t2,2), round(t3,2),
            reasons
        )

    except:
        return None

# ─────────────────────────────────────────────
# RUN SCANNER
# ─────────────────────────────────────────────
def run_scan(tickers):

    print("Downloading data...")
    data = yf.download(tickers, period="1y", group_by="ticker", progress=False)

    results = []

    def process(t):
        try:
            if t not in data:
                return None
            return analyze_ticker(t, data[t])
        except:
            return None

    print("Running parallel scan...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process, t) for t in tickers]

        for f in as_completed(futures):
            res = f.result()
            if res and res.score >= 40:
                results.append(res)

    results.sort(key=lambda x: x.score, reverse=True)

    return results[:10]

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # REALISTIC 200 STOCK LIST (sample)
    tickers = [
        "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD","NFLX","SHOP",
        "JPM","BAC","GS","MS","WFC","C","V","MA","PYPL","SQ"
    ] * 10  # 200 total loops

    tickers = list(set(tickers))  # remove duplicates

    start = time.time()

    results = run_scan(tickers)

    print("\n🔥 TOP 10 PICKS\n")

    for r in results:
        print("====================================")
        print(f"{r.ticker} — {r.signal} (Score: {r.score})")
        print(f"Setup: {r.setup_type} | Confidence: {r.confidence}%")
        print(f"Entry: {r.price} | SL: {r.stop_loss}")
        print(f"T1: {r.target_1} | T2: {r.target_2} | T3: {r.target_3}")
        print("Reasons:", ", ".join(r.reasons))

    print(f"\n⏱ Completed in {round(time.time()-start,2)} seconds")
