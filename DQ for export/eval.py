import pandas as pd

import altair as alt

from tqdm import tqdm
import device_vars as dv
import coloredlogs

from trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
from trading_bot.methods import evaluate_model_speed, evaluate_model
from trading_bot.agent import Agent

def visualize(df, history, title="trading session"):
    # add history to dataframe
    position = [history[0][0][3]] + [x[0][3] for x in history]
    actions = ['HOLD'] + [x[1] for x in history]
    df['position'] = position
    df['action'] = actions

    action2 = [x[1] for x in history]


    # specify y-axis scale for stock prices
    scale = alt.Scale(domain=(min(min(df['actual']), min(df['position'])) - 50, max(max(df['actual']), max(df['position'])) + 50), clamp=True)

    # plot a line chart for stock positions
    actual = alt.Chart(df).mark_line(
        color='green',
        opacity=0.5
    ).encode(
        x='date:T',
        y=alt.Y('position', axis=alt.Axis(format='$.2f', title='Price'), scale=scale)
    ).interactive(
        bind_y=False
    )
    
    # plot the BUY and SELL actions as points
    points = alt.Chart(df).transform_filter(
        alt.datum.action != 'HOLD'
    ).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('position', axis=alt.Axis(format='$.2f', title='Price'), scale=scale),
        color='action'
    ).interactive(bind_y=False)

    # merge the two charts
    chart = alt.layer(actual, points, title=title).properties(height=300, width=1000)
    
    return chart

def test_d(model_name, test_stock, pdd = True):
    if pdd:
        print(model_name.split("/")[-1], test_stock.split("/")[-1])
    window_size = 30
    debug = False

    model_name = model_name.replace('models/', '')

    from trading_bot.agent import Agent

    agent = Agent(window_size, pretrained=True, model_name=model_name)

    agent.learning_rate = agent.learning_rate

    # read csv into dataframe
    df = pd.read_csv(test_stock)
    # filter out the desired features
    df = df[['Date', 'Adj Close']]
    # rename feature column names
    df = df.rename(columns={'Adj Close': 'actual', 'Date': 'date'})
    # convert dates from object to DateTime type
    dates = df['date']
    dates = pd.to_datetime(dates, infer_datetime_format=True)
    df['date'] = dates

    import logging
    import coloredlogs

    from trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
    from trading_bot.methods import evaluate_model_speed, evaluate_model, evaluate_model01, evaluate_model01_1
    from eval_and_train import main

    coloredlogs.install() # level='DEBUG'
    switch_k_backend_device()

    test_data = get_stock_data(test_stock)

    test_data = test_data[:int(len(test_data)/5)] # validation data
    # test_data = test_data[int(len(test_data)/5):int(len(test_data)/5) * 2] # train data
    # test_data = test_data[:int(len(test_data)/5) * 2] # ALL TRAIN DATA
    # test_data = test_data[int(len(test_data)/5) * 2:] # non-train data
    # test_data = test_data[int(len(test_data)/5) * 2:int(len(test_data)/5) * 3] # 1w data



    initial_offset = test_data[1][3] - test_data[0][3]

    test_result, history, wins, losses, total_trades, mwls, final_score = evaluate_model01(agent, test_data, window_size, debug, window_size)
    total_profit = test_result
    ec = mwls[3]
    if pdd and (wins + losses) != 0:
        show_eval_result(model_name, test_result, initial_offset)
        print("TOTAL PROFIT: ", total_profit)
        print("WIN RATE: ", wins / (wins + losses))
        print("WINS: ", wins)
        print("LOSSES: ", losses)
        print("TOTAL TRADES: ", total_trades)
        print("LONGEST LOSS STREAK: ", mwls[1])
        print("LONGEST WIN STREAK: ", mwls[0])
        print("MAX DRAWDOWN: ", mwls[2])
        print("Longs: ", mwls[4])
        print("Shorts: ", mwls[5])

    
    import matplotlib.pyplot as plt

    ys = ec.copy()
    ys = ys
    xs = [x for x in range(len(ys))]

    tp = [450 for x in range(len(ys))]
    sl = [-150 for x in range(len(ys))]
    cost_per_trade = [1.25 * x for x in range(len(ys))]

    peak = 0
    max_dd = 200
    max_dd_l = []
    for i in ec:
        peak = max(peak, i)
        max_dd_l.append(peak - max_dd)

    plt.title("Week {} Equity Curve".format(test_stock))
    plt.plot(xs, tp)
    plt.plot(xs, sl)
    plt.plot(xs, ys)
    plt.plot(xs, cost_per_trade)
    plt.plot(xs, max_dd_l)
    plt.show()
    # Make sure to close the plt object once done
    plt.close()

    return wins / (wins + losses) if (wins + losses) != 0 else -1

def test_d_plt(model_name, test_stock, image_save_dir, net_pnl_f, pdd = True):
    if pdd:
        print(model_name.split("/")[-1], test_stock.split("/")[-1])
    window_size = 30
    debug = False

    model_name = model_name.replace('models/', '')

    from trading_bot.agent import Agent

    agent = Agent(window_size, pretrained=True, model_name=model_name)

    agent.learning_rate = agent.learning_rate

    # read csv into dataframe
    df = pd.read_csv(test_stock)
    # filter out the desired features
    df = df[['Date', 'Adj Close']]
    # rename feature column names
    df = df.rename(columns={'Adj Close': 'actual', 'Date': 'date'})
    # convert dates from object to DateTime type
    dates = df['date']
    dates = pd.to_datetime(dates, infer_datetime_format=True)
    df['date'] = dates

    import logging
    import coloredlogs

    from trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
    from trading_bot.methods import evaluate_model_speed, evaluate_model, evaluate_model01, evaluate_model01_1
    from eval_and_train import main

    coloredlogs.install() # level='DEBUG'
    switch_k_backend_device()

    test_data = get_stock_data(test_stock)

    test_data = test_data[:int(len(test_data)/5)] # validation data
    # test_data = test_data[int(len(test_data)/5):int(len(test_data)/5) * 2] # train data
    # test_data = test_data[:int(len(test_data)/5) * 2] # ALL TRAIN DATA
    # test_data = test_data[int(len(test_data)/5) * 2:] # non-train data
    # test_data = test_data[int(len(test_data)/5) * 2:int(len(test_data)/5) * 3] # 1w data



    initial_offset = test_data[1][3] - test_data[0][3]

    test_result, history, wins, losses, total_trades, mwls, final_score = evaluate_model01(agent, test_data, window_size, debug, window_size)
    total_profit = test_result
    ec = mwls[3]
    if pdd and (wins + losses) != 0:
        show_eval_result(model_name, test_result, initial_offset)
        print("TOTAL PROFIT: ", total_profit)
        print("WIN RATE: ", wins / (wins + losses))
        print("WINS: ", wins)
        print("LOSSES: ", losses)
        print("TOTAL TRADES: ", total_trades)
        print("LONGEST LOSS STREAK: ", mwls[1])
        print("LONGEST WIN STREAK: ", mwls[0])
        print("MAX DRAWDOWN: ", mwls[2])
        print("Longs: ", mwls[4])
        print("Shorts: ", mwls[5])

    
    import matplotlib.pyplot as plt

    ys = ec.copy()
    ys = ys
    xs = [x for x in range(len(ys))]

    tp = [450 for x in range(len(ys))]
    sl = [-150 for x in range(len(ys))]
    cost_per_trade = [1.25 * x for x in range(len(ys))]

    peak = 0
    max_dd = 200
    max_dd_l = []
    prf = ec[-1] if len(ec) != 0 else 0
    vald = True

    for i in ec:
        peak = max(peak, i)
        max_dd_l.append(peak - max_dd)
        if vald and i < max_dd_l[-1]:
            prf = i
            vald = False

    plt.title("Week {} Equity Curve".format(test_stock))
    plt.plot(xs, tp)
    plt.plot(xs, sl)
    plt.plot(xs, ys)
    plt.plot(xs, cost_per_trade)
    plt.plot(xs, max_dd_l)
    
    dxd = image_save_dir + test_stock.split("/")[-1].split(".")[0] + ".png"

    if len(xs) != 0:
        plt.savefig(dxd)

    with open(net_pnl_f, "a") as f:
        f.write(str(dxd) + ": " + str(prf) + "\n")
    

    # Make sure to close the plt object once done
    plt.close()

    return wins / (wins + losses) if (wins + losses) != 0 else -1


def test_d_comb(model_names, test_stock, pdd = True):
    # if pdd:
    #     print(model_name.split("/")[-1], test_stock.split("/")[-1])
    window_size = 30
    debug = False

    model_names = [i.replace('models/', '') for i in model_names]
    # model_name = model_name.replace('models/', '')

    from trading_bot.agent import Agent

    # agent = Agent(window_size, pretrained=True, model_name=model_name)

    agents = [Agent(window_size, pretrained=True, model_name=i) for i in model_names]

    # read csv into dataframe
    df = pd.read_csv(test_stock)
    # filter out the desired features
    df = df[['Date', 'Adj Close']]
    # rename feature column names
    df = df.rename(columns={'Adj Close': 'actual', 'Date': 'date'})
    # convert dates from object to DateTime type
    dates = df['date']
    dates = pd.to_datetime(dates, infer_datetime_format=True)
    df['date'] = dates

    import logging
    import coloredlogs

    from trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
    from trading_bot.methods import evaluate_model01, evaluate_model01_1
    from eval_and_train import main

    coloredlogs.install() # level='DEBUG'
    switch_k_backend_device()

    test_data = get_stock_data(test_stock)

    # test_data = test_data[:int(len(test_data)/5)] # validation data
    # test_data = test_data[int(len(test_data)/5):int(len(test_data)/5) * 2] # train data
    # test_data = test_data[:int(len(test_data)/5) * 2] # ALL TRAIN DATA
    # test_data = test_data[int(len(test_data)/5) * 2:] # non-train data
    # test_data = test_data[int(len(test_data)/5) * 2:int(len(test_data)/5) * 3] # 1w data



    initial_offset = test_data[1][3] - test_data[0][3]

    test_result, history, wins, losses, total_trades, mwls, final_score = evaluate_model01_1(agents, test_data, window_size, debug, window_size)
    
    
    
    total_profit = test_result
    ec = mwls[3]
    if pdd and (wins + losses) != 0:
        show_eval_result(model_name, test_result, initial_offset)
        print("TOTAL PROFIT: ", total_profit)
        print("WIN RATE: ", wins / (wins + losses))
        print("WINS: ", wins)
        print("LOSSES: ", losses)
        print("TOTAL TRADES: ", total_trades)
        print("LONGEST LOSS STREAK: ", mwls[1])
        print("LONGEST WIN STREAK: ", mwls[0])
        print("MAX DRAWDOWN: ", mwls[2])
        print("Longs: ", mwls[4])
        print("Shorts: ", mwls[5])

    
    import matplotlib.pyplot as plt

    ys = ec.copy()
    ys = ys
    xs = [x for x in range(len(ys))]

    tp = [450 for x in range(len(ys))]
    sl = [-150 for x in range(len(ys))]
    cost_per_trade = [1.25 * x for x in range(len(ys))]

    peak = 0
    max_dd = 200
    max_dd_l = []
    for i in ec:
        peak = max(peak, i)
        max_dd_l.append(peak - max_dd)

    plt.title("Week {} Equity Curve".format(test_stock))
    plt.plot(xs, tp)
    plt.plot(xs, sl)
    plt.plot(xs, ys)
    plt.plot(xs, max_dd_l)
    plt.plot(xs, cost_per_trade)
    plt.show()
    # Make sure to close the plt object once done
    plt.close()

    return wins / (wins + losses) if (wins + losses) != 0 else -1

def test_d_comb_plt(model_names, test_stock, image_save_dir, net_pnl_f, pdd = True):
    # if pdd:
    #     print(model_name.split("/")[-1], test_stock.split("/")[-1])
    window_size = 30
    debug = False

    model_names = [i.replace('models/', '') for i in model_names]
    # model_name = model_name.replace('models/', '')

    from trading_bot.agent import Agent

    # agent = Agent(window_size, pretrained=True, model_name=model_name)

    agents = [Agent(window_size, pretrained=True, model_name=i) for i in model_names]

    # read csv into dataframe
    df = pd.read_csv(test_stock)
    # filter out the desired features
    df = df[['Date', 'Adj Close']]
    # rename feature column names
    df = df.rename(columns={'Adj Close': 'actual', 'Date': 'date'})
    # convert dates from object to DateTime type
    dates = df['date']
    dates = pd.to_datetime(dates, infer_datetime_format=True)
    df['date'] = dates

    import logging
    import coloredlogs

    from trading_bot.utils import show_eval_result, switch_k_backend_device, get_stock_data
    from trading_bot.methods import evaluate_model01, evaluate_model01_1
    from eval_and_train import main

    coloredlogs.install() # level='DEBUG'
    switch_k_backend_device()

    test_data = get_stock_data(test_stock)

    test_data = test_data[:int(len(test_data)/5)] # 1 day of data
    # test_data = test_data[int(len(test_data)/5):int(len(test_data)/5) * 2] # train data
    # test_data = test_data[:int(len(test_data)/5) * 2] # ALL TRAIN DATA
    # test_data = test_data[int(len(test_data)/5) * 2:] # non-train data
    # test_data = test_data[int(len(test_data)/5) * 2:int(len(test_data)/5) * 3] # 1w data



    initial_offset = test_data[1][3] - test_data[0][3]

    test_result, history, wins, losses, total_trades, mwls, final_score = evaluate_model01_1(agents, test_data, window_size, debug, window_size)
    
    
    
    total_profit = test_result
    ec = mwls[3]
    if pdd and (wins + losses) != 0:
        show_eval_result(model_name, test_result, initial_offset)
        print("TOTAL PROFIT: ", total_profit)
        print("WIN RATE: ", wins / (wins + losses))
        print("WINS: ", wins)
        print("LOSSES: ", losses)
        print("TOTAL TRADES: ", total_trades)
        print("LONGEST LOSS STREAK: ", mwls[1])
        print("LONGEST WIN STREAK: ", mwls[0])
        print("MAX DRAWDOWN: ", mwls[2])
        print("Longs: ", mwls[4])
        print("Shorts: ", mwls[5])

    
    import matplotlib.pyplot as plt

    ys = ec.copy()
    ys = ys
    xs = [x for x in range(len(ys))]

    tp = [450 for x in range(len(ys))]
    sl = [-150 for x in range(len(ys))]
    cost_per_trade = [1.25 * x for x in range(len(ys))]

    peak = 0
    max_dd = 200
    max_dd_l = []
    prf = ec[-1] if len(ec) != 0 else 0
    vald = True

    for i in ec:
        peak = max(peak, i)
        max_dd_l.append(peak - max_dd)
        if vald and i < max_dd_l[-1]:
            prf = i
            vald = False

    plt.title("Week {} Equity Curve".format(test_stock))
    plt.plot(xs, tp)
    plt.plot(xs, sl)
    plt.plot(xs, ys)
    plt.plot(xs, max_dd_l)
    plt.plot(xs, cost_per_trade)

    dxd = image_save_dir + test_stock.split("/")[-1].split(".")[0] + ".png"

    plt.savefig(dxd)

    with open(net_pnl_f, "a") as f:
        f.write(str(dxd) + ": " + str(prf) + "\n")
    
    # # Make sure to close the plt object once done
    plt.close()

    return wins / (wins + losses) if (wins + losses) != 0 else -1




alt.data_transformers.enable('csv')

model_name = "models/week_2025-01-11_XL_MAS_FINAL_EP"
test_stock = "data/NQ1-1m-DATA-BY-WEEK/week_2025-01-11.csv" # _INVERSE


a = ["BY", "WEEK", "by", "week"]
b = ["TUE", "MON", "tue", "mon"]
c = ["WED", "TUE", "wed", "tue"]
d = ["THU", "WED", "thu", "wed"]
e = ["FRI", "THU", "fri", "thu"]

sel = e

data_dir = f"data/NQ1-1m-DATA-{sel[0]}-{sel[1]}/"
mod_dir = f"models/seeded-{sel[2]}-{sel[3]}/"
image_save_dir = f"save_imgs/seeded-{sel[2]}-{sel[3]}-run/"

pnlf = image_save_dir + "net_pnls.txt"


from pathlib import Path
from datetime import datetime, timedelta
mn = []
ddls = {}

for i in Path(mod_dir).iterdir():
    # if "2024-12-07" in str(i): # "week_2025-02-08_" in str(i) and 
    print(i)
    mod_name = str(i).replace("models\\", "") # .replace("\\", "/")
    print("\n\n", mod_name, "\n\n")
    weekn = mod_name.split("_")[1]
    do = datetime.strptime(weekn, "%Y-%m-%d") + timedelta(weeks=1)
    dd = do.strftime("%Y-%m-%d")

    test_stock1 = data_dir + "week_" + dd + ".csv"
    b = test_d_plt(mod_name, test_stock1, image_save_dir, pnlf, False)

    if weekn not in ddls.keys():
        test_stock1 = data_dir + "week_" + dd  + ".csv"
        ddls[weekn] = [test_stock1, []]

    ddls[weekn][1].append(mod_name)

        

        # test_stock1 = "data/NQ1-1m-DATA-BY-WEEK/week_" + dd  + ".csv"
        # b = test_d(mod_name, test_stock1, False)
        # mn.append(mod_name)
        # # test_stock2 = "data/NQ1-1m-DATA-BY-WEEK/week_" + dd + "_INVERSE.csv"

# ll = []
# for i in tqdm(list(ddls.keys())):
#     ll.append(test_d_comb_plt(ddls[i][1], ddls[i][0], image_save_dir, pnlf, False))
    

# a = test_d_comb(mn, test_stock1, image_save_dir, True)
#         # b = test_d(mod_name, test_stock2, False)
        
"Seeded: 98 -> 128 -> 256, Weeklys: 64 -> 128 -> 256, and modified attention for seeded"

raise



window_size = 30
debug = False

model_name = model_name.replace('models/', '')

agent = Agent(window_size, pretrained=True, model_name=model_name)

agent.learning_rate = agent.learning_rate

# read csv into dataframe
df = pd.read_csv(test_stock)
# filter out the desired features
df = df[['Date', 'Adj Close']]
# rename feature column names
df = df.rename(columns={'Adj Close': 'actual', 'Date': 'date'})
# convert dates from object to DateTime type
dates = df['date']
dates = pd.to_datetime(dates, infer_datetime_format=True)
df['date'] = dates

coloredlogs.install() # level='DEBUG'
switch_k_backend_device()

test_data = get_stock_data(test_stock)

test_data = get_stock_data(test_stock)

# test_data = test_data[:int(len(test_data)/5)] # validation data
# test_data = test_data[int(len(test_data)/5):int(len(test_data)/5) * 2] # train data
test_data = test_data[int(len(test_data)/5) * 2:] # non-train data
# test_data = test_data[int(len(test_data)/5) * 2:int(len(test_data)/5) * 3] # 1w data

initial_offset = test_data[1][3] - test_data[0][3]

test_result, history, wins, losses, total_trades, mwls, final_score = evaluate_model_speed(agent, test_data, window_size, debug, window_size)
total_profit = test_result
ec = mwls[3]
show_eval_result(model_name, test_result, initial_offset)
print("FINAL REWARD: ", final_score)
print("TOTAL PROFIT: ", total_profit)
print("WIN RATE: ", wins / (wins + losses))
print("WINS: ", wins)
print("LOSSES: ", losses)
print("TOTAL TRADES: ", total_trades)
print("LONGEST LOSS STREAK: ", mwls[1])
print("LONGEST WIN STREAK: ", mwls[0])
print("MAX DRAWDOWN: ", mwls[2])
print("Longs: ", mwls[4])
print("Shorts: ", mwls[5])


def visualize(df, history, title="trading session"):
    # add history to dataframe
    position = [history[0][0][3]] + [x[0][3] for x in history]
    actions = ['HOLD'] + [x[1] for x in history]
    df['position'] = position
    df['action'] = actions

    action2 = [x[1] for x in history]


    # specify y-axis scale for stock prices
    scale = alt.Scale(domain=(min(min(df['actual']), min(df['position'])) - 50, max(max(df['actual']), max(df['position'])) + 50), clamp=True)

    # plot a line chart for stock positions
    actual = alt.Chart(df).mark_line(
        color='green',
        opacity=0.5
    ).encode(
        x='date:T',
        y=alt.Y('position', axis=alt.Axis(format='$.2f', title='Price'), scale=scale)
    ).interactive(
        bind_y=False
    )
    
    # plot the BUY and SELL actions as points
    points = alt.Chart(df).transform_filter(
        alt.datum.action != 'HOLD'
    ).mark_point(
        filled=True
    ).encode(
        x=alt.X('date:T', axis=alt.Axis(title='Date')),
        y=alt.Y('position', axis=alt.Axis(format='$.2f', title='Price'), scale=scale),
        color='action'
    ).interactive(bind_y=False)

    # merge the two charts
    chart = alt.layer(actual, points, title=title).properties(height=300, width=1000)
    
    return chart

def get_ec(data_name, window_size, model_name, debug):
    model_name = model_name.replace('models/', '')
    test_data = get_stock_data(data_name)
    agent2 = Agent(window_size, pretrained=True, model_name=model_name)
    test_result, history, wins, losses, total_trades, mwls = evaluate_model(agent2, test_data, window_size, debug, window_size)
    return mwls[3]


# TESTED EQUITY CURVE

import matplotlib.pyplot as plt

ys = ec.copy()
ys = ys
xs = [x for x in range(len(ys))]

tp = [450 for x in range(len(ys))]
sl = [-150 for x in range(len(ys))]
cost_per_trade = [1.25 * x for x in range(len(ys))]

plt.title("Week {} Equity Curve".format(test_stock))
plt.plot(xs, tp)
plt.plot(xs, sl)
plt.plot(xs, ys)
plt.plot(xs, cost_per_trade)
plt.show()
# Make sure to close the plt object once done
plt.close()