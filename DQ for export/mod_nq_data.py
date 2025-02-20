import pandas as pd
from tqdm import tqdm

df = pd.read_csv('data/glbx-mdp3-20100606-20241222.ohlcv-1m.csv')
# df['Date'] = pd.to_datetime(df['Date'])

df.pop('rtype')
df.pop('publisher_id')
df.pop('instrument_id')
df['Date'] = pd.to_datetime(df['ts_event'])
df['Open'] = df['open']
df['High'] = df['high']
df['Low'] = df['low']
df['Close'] = df['close']
df['Adj Close'] = df['close']
df['Volume'] = df['volume']
df["Symbol"] = df['symbol']
df.pop('open')
df.pop('high')
df.pop('low')
df.pop('close')
df.pop('volume')
df.pop('ts_event')
df.pop('symbol')

print(df)
grouped = df.groupby(pd.Grouper(key="Date", freq='W-THU'))
for name, group in tqdm(grouped):
    week_start = name.strftime('%Y-%m-%d')
    filename = f"data/NQ1-1m-DATA-FRI-THU/week_{week_start}.csv"
    syms = group['Symbol'].tolist()
    best_sym = ""
    sym_count = 0
    for i in set(syms):
        if syms.count(i) > sym_count:
            sym_count = syms.count(i)
            best_sym = i
    mask = group['Symbol'] == best_sym
    group = group[mask]
    group.to_csv(filename, index=False)



# df.to_csv('data/nq_1m_max_ALLLLL_DATA.csv', index = False)