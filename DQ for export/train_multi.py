from train import main, main2
from collections import deque
from threading import Thread
from trading_bot.utils import get_stock_data
from inversedata import inv_data
from subprocess import Popen, PIPE
from pathlib import Path
import device_vars as dv
import datetime as dt

from subprocess import Popen, CREATE_NEW_CONSOLE
from time import sleep
from trading_bot.agent import Agent

def threading_func(qq, m_num = None):
    mname = "_XL_MAS"

    # fname = qq.pop()
    # verf_name = None # emtpy data is here: 'data/EMPTY.csv'
    # csv_write = fname.replace(".csv", "") + "_INVERSE.csv"
    # model_name = modfold + fname.split("/")[-1].replace(".csv", "") + str(mname)

    if not dv.use_cmd:
        while True:
            fname, find = qq.pop()
            modfold = dv.save_fold[find]
            verf_name = fname # emtpy data is here: 'data/EMPTY.csv'
            csv_write = fname.replace(".csv", "") + "_INVERSE.csv"
            model_name = modfold + fname.split("/")[-1].replace(".csv", "") + str(mname) + str(m_num)

            ds = fname.split("/")[-1].replace(".csv", "").split("_")[-1]
            verf_name = "/".join(fname.split("/")[:-1]) + "/" + "week_" + (dt.datetime.strptime(ds, "%Y-%m-%d") - dt.timedelta(weeks=1)).strftime('%Y-%m-%d') + ".csv"

            main2(fname, verf_name, 30, 32, 1, train_stock2 = csv_write, model_name = model_name) # 30 if dv.train_1_day else 50


    else:
        while True:
            fname, find = qq.pop()
            modfold = dv.save_fold[find]
            verf_name = None # emtpy data is here: 'data/EMPTY.csv'
            csv_write = fname.replace(".csv", "") + "_INVERSE.csv"
            model_name = modfold + fname.split("/")[-1].replace(".csv", "") + str(mname) + str(m_num)
            
            ds = fname.split("/")[-1].replace(".csv", "").split("_")[-1]
            verf_name = "/".join(fname.split("/")[:-1]) + "/" + "week_" + (dt.datetime.strptime(ds, "%Y-%m-%d") - dt.timedelta(weeks=1)).strftime('%Y-%m-%d') + ".csv"

            il = [dv.python_path, 
                dv.train_f_path, 
                f'{fname}',  # train data 1
                f'{verf_name}', # val data
                f'{csv_write}', # train data 2
                r'--strategy=t-dqn', # strategy
                r'--window-size=30', # window size
                r'--batch-size=32', # batch size
                f'--episode-count={1}', # episode count # 30 if dv.train_1_day else 50
                f'--model-name={model_name}', # model name
                # r'--pretrained', # pretrained? if so, enable it
                # r'--debug', # debug? if so, uncomment this
            ]

            print(" ".join(il))

            c = Popen(il, creationflags=CREATE_NEW_CONSOLE)

            c.wait()
    
names = []
for i in range(len(dv.data_fold)):
    p = dv.data_fold[i]

    names2 = [[p + str(f.name), i] for f in Path(p).iterdir() if f.is_file() and "_INVERSE" not in str(f.name)][dv.i_start:dv.i_end]
    names2.reverse()
    names += names2

for i in names:
    get_stock_data(i[0])
    try:
        get_stock_data(i[0].replace(".csv", "") + "_INVERSE.csv")
    except:
        inv_data(i)
        get_stock_data(i[0].replace(".csv", "") + "_INVERSE.csv")

names.reverse()

qq1 = deque(names)
qq2 = deque(names)
qq3 = deque(names)
qq4 = deque(names)


if __name__ == "__main__":
    # profile = cProfile.Profile()
    # profile.enable()
    # threading_func(qq1)
    thread_cnt = dv.threads
    threads = []
    try:
        # threads.append(Thread(target = threading_func, args = (qq1, "qq1", ))) # ignoring this for now
        # threads.append(Thread(target = threading_func, args = (qq2, "qq2", ))) # ignoring this for now
        # threads.append(Thread(target = threading_func, args = (qq3, "qq3", ))) # ignoring this for now
        # threads.append(Thread(target = threading_func, args = (qq4, "qq4", ))) # ignoring this for now
        for i in range(thread_cnt):
            threads.append(Thread(target = threading_func, args = (qq1, "", )))
        for i in threads:
            i.start()
        for i in threads:
            i.join()
    except KeyboardInterrupt:
        print("Aborted!")

    
    # profile.disable()

    # results = pstats.Stats(profile)
    # results.sort_stats("time")
    # results.print_stats(25)