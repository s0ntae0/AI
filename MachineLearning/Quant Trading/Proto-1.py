"""
Binance BTC/USDT OHLCV -> Features -> LightGBM -> Buy/Sell -> Simple Backtest

주의:
- 이 코드는 "연습용"이다. 실거래는 슬리피지, 주문체결, 레버리지, 리스크 관리 반드시 추가.
- 데이터 누수 방지 위해: 피처는 현재/과거만, 라벨은 미래(shift -1).
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd

# pip install ccxt lightgbm scikit-learn
import ccxt
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# 1) 데이터 로드 (Binance OHLCV)
# -----------------------------
def fetch_ohlcv_binance(symbol="BTC/USDT", timeframe="1h", limit=1000, since_ms=None):
    """
    limit: 최대 1000(바이낸스 기준). 더 길게 받으려면 루프.
    since_ms: 시작 시각(ms). None이면 최근 limit개.
    """
    ex = ccxt.binance({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df


def fetch_many(symbol="BTC/USDT", timeframe="1h", days=365, batch_limit=1000):
    """
    days 만큼의 1h 캔들을 최대한 가져오기.
    바이낸스는 한 번에 limit=1000이라 반복 호출.
    """
    ex = ccxt.binance({"enableRateLimit": True})
    now_ms = ex.milliseconds()
    since_ms = now_ms - int(days * 24 * 60 * 60 * 1000)

    all_rows = []
    cur_since = since_ms

    while True:
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur_since, limit=batch_limit)
        if not data:
            break

        all_rows.extend(data)

        last_ts = data[-1][0]
        # 다음 호출은 마지막 캔들 다음으로
        cur_since = last_ts + 1

        # 너무 자주 치지 않게 레이트리밋 약간 여유
        time.sleep(ex.rateLimit / 1000)

        # 종료 조건: 마지막 캔들이 현재에 거의 도달하면 끝
        if last_ts >= now_ms - 2 * 60 * 60 * 1000:
            break

        # 안전장치: 무한루프 방지
        if len(all_rows) > days * 24 + 5000:
            break

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df


# -----------------------------
# 2) Feature Engineering
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # returns
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_3"] = d["close"].pct_change(3)
    d["ret_6"] = d["close"].pct_change(6)
    d["ret_12"] = d["close"].pct_change(12)
    d["ret_24"] = d["close"].pct_change(24)

    # volatility
    d["vol_24"] = d["ret_1"].rolling(24).std()
    d["vol_72"] = d["ret_1"].rolling(72).std()

    # moving averages & momentum
    d["ma_10"] = d["close"].rolling(10).mean()
    d["ma_30"] = d["close"].rolling(30).mean()
    d["ma_ratio_10_30"] = d["ma_10"] / (d["ma_30"] + 1e-12)

    # range features
    d["hl_range"] = (d["high"] - d["low"]) / (d["close"] + 1e-12)
    d["oc_range"] = (d["close"] - d["open"]) / (d["open"] + 1e-12)

    # volume features
    d["vol_z_48"] = (d["volume"] - d["volume"].rolling(48).mean()) / (d["volume"].rolling(48).std() + 1e-12)

    # RSI
    d["rsi_14"] = rsi(d["close"], 14)

    # cleanup
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d


# -----------------------------
# 3) Labeling (BUY=1 / SELL=0)
# -----------------------------
def make_labels(df_feat: pd.DataFrame, horizon: int = 1, threshold: float = 0.0):
    """
    horizon 캔들 뒤 수익률로 라벨:
      - future_return > threshold -> BUY(1)
      - else -> SELL(0)
    threshold를 조금 올리면 노이즈 줄어듦(예: 0.0005)
    """
    d = df_feat.copy()
    d["future_ret"] = d["close"].pct_change(horizon).shift(-horizon)
    d["y"] = (d["future_ret"] > threshold).astype(int)
    d = d.dropna()
    return d


# -----------------------------
# 4) Train / Test split (time series)
# -----------------------------
def time_split(df: pd.DataFrame, train_ratio: float = 0.8):
    n = len(df)
    cut = int(n * train_ratio)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test


# -----------------------------
# 5) Backtest
# -----------------------------
def simple_backtest(test_df: pd.DataFrame, y_pred: np.ndarray, fee_bps: float = 2.0):
    """
    전략:
      - 예측이 1이면 다음 캔들 롱, 0이면 다음 캔들 숏
      - 1캔들 보유 후 청산(단순)
    fee_bps: 왕복 수수료를 대충 반영하고 싶으면 bps 늘려라 (예: 4~10 bps)
    """
    fee = fee_bps / 10000.0

    # 포지션: BUY=+1, SELL=-1
    pos = np.where(y_pred == 1, 1.0, -1.0)

    # 다음 캔들 수익률을 "실현 수익"으로(누수 주의: 여기서는 test에서 미래값 사용 OK)
    # test_df는 이미 future_ret 컬럼이 있음
    raw_pnl = pos * test_df["future_ret"].values

    # 거래 비용: 매 캔들 진입/청산 가정 -> 대략 fee 차감
    pnl = raw_pnl - fee

    equity = (1 + pnl).cumprod()
    total_return = equity[-1] - 1

    # 간단 샤프(연환산): 1h 기준이면 연환산 팩터는 sqrt(24*365)
    sharpe = np.mean(pnl) / (np.std(pnl) + 1e-12) * np.sqrt(24 * 365)

    # MDD
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1
    mdd = dd.min()

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "equity_curve": equity,
        "pnl": pnl
    }


# -----------------------------
# 6) Main
# -----------------------------
def main():
    symbol = "BTC/USDT"
    timeframe = "1h"
    days = 365 * 2          # 연습: 최근 2년치 1h
    horizon = 1             # 다음 1캔들 수익률로 라벨
    threshold = 0.0         # 노이즈 줄이려면 0.0005 같은 값도 추천
    fee_bps = 2.0           # 대충 왕복 2bps 가정(원하면 6~10bps로 현실화)

    print(f"Fetching {symbol} {timeframe} for ~{days} days...")
    df = fetch_many(symbol=symbol, timeframe=timeframe, days=days)
    print("raw:", df.shape, df.index.min(), "->", df.index.max())

    df_feat = add_features(df)
    df_lab = make_labels(df_feat, horizon=horizon, threshold=threshold)

    # 학습에 쓸 feature 컬럼
    feature_cols = [
        "ret_1","ret_3","ret_6","ret_12","ret_24",
        "vol_24","vol_72",
        "ma_ratio_10_30",
        "hl_range","oc_range",
        "vol_z_48",
        "rsi_14",
    ]

    train, test = time_split(df_lab, train_ratio=0.8)
    X_train, y_train = train[feature_cols], train["y"]
    X_test, y_test = test[feature_cols], test["y"]

    model = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )

    print("Training LightGBM...")
    model.fit(X_train, y_train)

    # 예측
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # 백테스트
    bt = simple_backtest(test, y_pred, fee_bps=fee_bps)
    print("\n=== Simple Backtest (1-candle hold) ===")
    print(f"Total Return: {bt['total_return']*100:.2f}%")
    print(f"Sharpe (rough): {bt['sharpe']:.2f}")
    print(f"Max Drawdown: {bt['mdd']*100:.2f}%")

    # 최신 캔들 기준 “지금 신호” 출력(마지막 row는 미래수익률 없어서 df_feat 기준)
    last_row = df_feat.iloc[-1:][feature_cols]
    last_proba = model.predict_proba(last_row)[:, 1][0]
    last_signal = "BUY" if last_proba >= 0.5 else "SELL"
    print("\n=== Latest Signal ===")
    print("time:", df_feat.index[-1])
    print(f"proba(BUY): {last_proba:.4f} -> {last_signal}")

if __name__ == "__main__":
    main()
