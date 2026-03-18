import yfinance as yf
import pandas as pd

df = yf.download('^NSEI', period='2d', interval='1m', progress=False)

def check(entry_time, entry_price, direction, sl, tp):
    t = pd.to_datetime(entry_time).tz_localize(None)
    df.index = df.index.tz_localize(None)
    sub = df[df.index >= t]
    if sub.empty: return "NO DATA"
    hit_sl = False; hit_tp = False; exit_time = None; pnl = 0
    for idx, row in sub.iterrows():
        high = float(row['High'].iloc[0]) if isinstance(row['High'], pd.DataFrame) or isinstance(row['High'], pd.Series) else float(row['High'])
        low = float(row['Low'].iloc[0]) if isinstance(row['Low'], pd.DataFrame) or isinstance(row['Low'], pd.Series) else float(row['Low'])
        if direction == 'CE':
            if low <= sl: hit_sl = True; exit_time = idx; pnl = sl - entry_price; break
            if high >= tp: hit_tp = True; exit_time = idx; pnl = tp - entry_price; break
        else:
            if high >= sl: hit_sl = True; exit_time = idx; pnl = entry_price - sl; break
            if low <= tp: hit_tp = True; exit_time = idx; pnl = entry_price - tp; break
    if hit_tp: return f"Target Hit! Profit: {pnl:.2f} at {exit_time}"
    if hit_sl: return f"SL Hit! Loss: {pnl:.2f} at {exit_time}"
    return "Still Open"

print("Trade 1 (12:26):", check('2026-02-20 12:26:12', 25648.199, 'CE', 25615.932, 25696.599))
print("Trade 2 (12:58):", check('2026-02-20 12:58:07', 25577.849, 'PE', 25612.148, 25526.401))
print("Trade 3 (13:08):", check('2026-02-20 13:08:27', 25570.949, 'PE', 25604.998, 25519.874))
