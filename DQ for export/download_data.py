import yfinance as yf
import pandas as pd


file_name = "NQ=F_2024-12-09_to_2024-12-14_1m.csv"
interval = "1m"


file_name = "data/" + file_name

data1 = yf.download("NQ=F", interval=interval, start = "2024-11-25", end = "2024-11-30")
data2 = yf.download("NQ=F", interval=interval, start = "2024-12-02", end = "2024-12-07")
data3 = yf.download("NQ=F", interval=interval, start = "2024-12-09", end = "2024-12-14")
data4 = yf.download("NQ=F", interval=interval, start = "2024-12-16", end = "2024-12-21")


# data = pd.concat([data1, data2])
# data = data3
data1.to_csv("NQ=F_2024-11-25_to_2024-11-30_1m.csv", encoding = 'utf-8', index = True, index_label = "Date", header = True)
data2.to_csv("NQ=F_2024-12-02_to_2024-12-07_1m.csv", encoding = 'utf-8', index = True, index_label = "Date", header = True)
data3.to_csv("NQ=F_2024-12-09_to_2024-12-14_1m.csv", encoding = 'utf-8', index = True, index_label = "Date", header = True)
data4.to_csv("NQ=F_2024-12-16_to_2024-12-21_1m.csv", encoding = 'utf-8', index = True, index_label = "Date", header = True)