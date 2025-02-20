import pandas as pd

def inv_data(csv_file):
    csvwrite = csv_file.replace(".csv", "") + "_INVERSE.csv"
    df = pd.read_csv(csv_file)

    df2 = df.copy()
    ind = list(df.index)
    po = df2["Open"][ind[0]]

    llist = df2["Low"]    .tolist()
    hlist = df2["High"]   .tolist()
    olist = df2["Open"]   .tolist()
    clist = df2["Close"]  .tolist()

    llistf = []
    hlistf = []
    olistf = []
    clistf = []

    for i in range(len(ind)):
        hlistf.append(po - (llist   [ind[i]] - po))
        llistf.append(po - (hlist   [ind[i]] - po))
        olistf.append(po - (olist   [ind[i]] - po))
        clistf.append(po - (clist   [ind[i]] - po))

    df2['High']  = llistf
    df2['Low']   = hlistf
    df2['Open']  = olistf
    df2['Close'] = clistf

    df2.to_csv(csvwrite)


def inv_from_pd(df):
    df2 = df.copy()
    ind = list(df.index)
    po = df2["Open"][ind[0]]

    llist = df2["Low"]    .tolist()
    hlist = df2["High"]   .tolist()
    olist = df2["Open"]   .tolist()
    clist = df2["Close"]  .tolist()

    llistf = []
    hlistf = []
    olistf = []
    clistf = []

    for i in range(len(ind)):
        hlistf.append(po - (llist   [ind[i]] - po))
        llistf.append(po - (hlist   [ind[i]] - po))
        olistf.append(po - (olist   [ind[i]] - po))
        clistf.append(po - (clist   [ind[i]] - po))

    df2['High']  = llistf
    df2['Low']   = hlistf
    df2['Open']  = olistf
    df2['Close'] = clistf

    return df2


# inv_data("data/NQ1-1m-DATA-BY-WEEK/week_2024-11-16.csv")
# inv_data("data/NQ1-1m-DATA-BY-WEEK/week_2024-11-23.csv")
# inv_data("data/NQ1-1m-DATA-BY-WEEK/week_2024-11-30.csv")
# inv_data("data/NQ1-1m-DATA-BY-WEEK/week_2024-12-07.csv")
# inv_data("data/NQ1-1m-DATA-BY-WEEK/week_2024-12-14.csv")

names = ['data/NQ1-1m-DATA-BY-WEEK/week_2024-11-09.csv', 'data/NQ1-1m-DATA-BY-WEEK/week_2024-11-02.csv',
         'data/NQ1-1m-DATA-BY-WEEK/week_2024-10-26.csv', 'data/NQ1-1m-DATA-BY-WEEK/week_2024-10-19.csv',
         'data/NQ1-1m-DATA-BY-WEEK/week_2024-10-12.csv', 'data/NQ1-1m-DATA-BY-WEEK/week_2024-10-05.csv',
         'data/NQ1-1m-DATA-BY-WEEK/week_2024-09-28.csv', 'data/NQ1-1m-DATA-BY-WEEK/week_2024-09-21.csv',
         'data/NQ1-1m-DATA-BY-WEEK/week_2024-09-14.csv', 'data/NQ1-1m-DATA-BY-WEEK/week_2024-09-07.csv',
         'data/NQ1-1m-DATA-BY-WEEK/week_2024-08-31.csv', 'data/NQ1-1m-DATA-BY-WEEK/week_2024-08-24.csv',
         'data/NQ1-1m-DATA-BY-WEEK/week_2024-08-17.csv', 'data/NQ1-1m-DATA-BY-WEEK/week_2024-08-10.csv',
         'data/NQ1-1m-DATA-BY-WEEK/week_2024-08-03.csv', 'data/NQ1-1m-DATA-BY-WEEK/week_2024-07-27.csv',]

if __name__ == "__main__":
    inv_data("data/NQ1-1m-DATA-BY-WEEK/week_2025-02-01.csv")
    # for i in names:
    #     inv_data(i)