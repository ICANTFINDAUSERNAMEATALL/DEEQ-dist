"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> <val-stock> <train-stock2>[--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug] 

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 15]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""
import logging
import coloredlogs

from docopt import docopt
import device_vars as dv
import pandas as pd

from inversedata import inv_from_pd
from trading_bot.agent import Agent
from trading_bot.methods import train_model3, train_model4, load_prev_data_, evaluate_model01, evaluate_model01_1
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)


def main2(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False, train_stock2 = None, old_model = None, old_stock = None, 
         old_stock_inverse = None, reward_offset = 0, preload_agent = None):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    train_data = get_stock_data(train_stock)
    train_data2 = get_stock_data(train_stock2)

    zero_val = False if val_stock != None else True
    if not zero_val:
        val_data = get_stock_data(val_stock)
        inv_d = inv_from_pd(pd.DataFrame(val_data, columns = ['Open', 'High', 'Low', 'Close', 'Volume']))
        val_data2 = [[io, ih, il, ic, iv] for io, ih, il, ic, iv in zip(list(inv_d['Open']), list(inv_d['High']), list(inv_d['Low']), list(inv_d['Close']), list(inv_d["Volume"]))]
        del inv_d
    else:
        val_data = []
        val_data2 = []

    ix = int(len(train_data)/5)
    d_train_data = train_data[ix:ix * 2]
    d_train_data2 = train_data2[ix:ix * 2]
    d_val_data = val_data[:ix]
    d_val_data2 = val_data2[:ix]

    if dv.train_1_day:
        ix = int(len(train_data)/5)
        train_data = train_data[ix:ix * 2]
        train_data2 = train_data2[ix:ix * 2]
        val_data = val_data[:ix]
        val_data2 = val_data2[:ix]

    mem_size_mult = 10

    mlen = (len(train_data) + len(train_data2)) * mem_size_mult
    if preload_agent != None:
        agent = preload_agent
    else:
        agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name, mem_len=mlen, ma_state_size=window_size)

    for episode in range(1, ep_count + 1):
        if episode == 1:
            agent.return_zero = True
        train_result1 = train_model4(agent, episode, train_data, ep_count=ep_count,
                                batch_size=batch_size, window_size=window_size, data2 = train_data2, 
                                epoch_for_mass_training = 1, ma_state_size=window_size,
                                reward_offset = reward_offset, random_state_eps = 3) #  if first_iter else 0

        # train_model4(agent, episode, d_train_data, ep_count=ep_count,
        #     batch_size=batch_size, window_size=window_size, data2 = d_train_data2, 
        #     epoch_for_mass_training = 1, ma_state_size=window_size,
        #     reward_offset = reward_offset, random_state_eps = 1)


        if not zero_val and False:
            _, _, _, _, total_trades3, _, train_reward_norm = evaluate_model01(agent, train_data, window_size, debug = False, ma_state_size = window_size)
            _, _, _, _, total_trades4, _, train_reward_inv = evaluate_model01(agent, train_data2, window_size, debug = False, ma_state_size = window_size)
            if total_trades3 == 0 or total_trades4 == 0:
                del agent, train_data, train_data2
                main2(train_stock, val_stock, window_size, batch_size, ep_count,
                    strategy, model_name, pretrained,
                    debug, train_stock2, old_model, old_stock, 
                    old_stock_inverse, reward_offset - 0.25, preload_agent = None)
                return


        agent.save(episode)


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False, train_stock2 = None, old_model = None, old_stock = None, 
         old_stock_inverse = None, reward_offset = 0, preload_agent = None):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    train_data = get_stock_data(train_stock)
    train_data2 = get_stock_data(train_stock2)

    old_data = get_stock_data(old_stock)
    old_data2 = get_stock_data(old_stock_inverse)

    zero_val = False if val_stock != None else True
    if not zero_val:
        val_data = get_stock_data(val_stock)
        inv_d = inv_from_pd(pd.DataFrame(val_data, columns = ['Open', 'High', 'Low', 'Close', 'Volume']))
        val_data2 = [[io, ih, il, ic, iv] for io, ih, il, ic, iv in zip(list(inv_d['Open']), list(inv_d['High']), list(inv_d['Low']), list(inv_d['Close']), list(inv_d["Volume"]))]
        del inv_d
    else:
        val_data = []
        val_data2 = []


    if dv.train_1_day:
        ix = int(len(train_data)/5)
        train_data = train_data[ix:ix * 2]
        train_data2 = train_data2[ix:ix * 2]
        val_data = val_data[:ix]
        val_data2 = val_data2[:ix]

    mem_size_mult = 20

    mlen = (len(train_data) + len(train_data2)) * mem_size_mult
    if preload_agent != None:
        agent = preload_agent
    else:
        agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name, mem_len=mlen, ma_state_size=window_size)

    train_epochs = 1 # 500
    current_batch_size = 1
    prev_train_loss = 0

    try:
        initial_offset = val_data[1][3] - val_data[0][3]
    except:
        initial_offset = 0
        zero_val = True
    min_val = 100
    min_val_epoch = 0

    bump_num = 0

    exitt = False
    weights = []
    reward_offset_lst = []

    diverge_t = 0
    diverge_val = 0
    best_avg_val = -999999999999
    best_avg_train = -999999999999
    best_avg = -999999999999
    max_vd_reward = -999999999999
    max_vd_epoch = 1000
    found = False
    valid_trades = []
    first_iter = True
    zero_trades_t = 0
    prev_train_reward = -999999999999
    prev_val_reward = -999999999999

    prev_aval = 0
    prev_atrain = 0


    fn = "models/" + model_name + "_val_stats.txt"

    print(fn)

    f2 = open(fn, "a")
    f2.close()

    for episode in range(1, ep_count + 1):
        if episode == 1:
            agent.return_zero = True
        train_result1 = train_model4(agent, episode, train_data, ep_count=ep_count,
                                batch_size=batch_size, window_size=window_size, data2 = train_data2, 
                                epoch_for_mass_training = int(train_epochs), ma_state_size=window_size,
                                reward_offset = 0, random_state_eps = 1) #  if first_iter else 0
        reward_offset_lst.append(reward_offset)
        if not zero_val:
            _, _, _, _, total_trades1, _, val_reward_norm = evaluate_model01(agent, val_data, window_size, debug = False, ma_state_size = window_size)
            _, _, _, _, total_trades2, _, val_reward_inv = evaluate_model01(agent, val_data2, window_size, debug = False, ma_state_size = window_size)
            _, _, _, _, total_trades3, _, train_reward_norm = evaluate_model01(agent, train_data, window_size, debug = False, ma_state_size = window_size)
            _, _, _, _, total_trades4, _, train_reward_inv = evaluate_model01(agent, train_data2, window_size, debug = False, ma_state_size = window_size)

            avg_trades = float(total_trades1 + total_trades2 + total_trades3 + total_trades4) / 4
            single_side = 0 in [total_trades1 , total_trades2 , total_trades3 , total_trades4]

            train_reward = float(train_reward_norm + train_reward_inv) / 2
            val_reward = float(val_reward_norm + val_reward_inv) / 2
            trd = train_reward - prev_train_reward # positive = getting better
            vrd = val_reward - prev_val_reward     # positive = getting better

            avg_val = 0 if (total_trades1 + total_trades2) == 0 else val_reward / (total_trades1 + total_trades2)
            avg_train = 0 if (total_trades3 + total_trades4) == 0 else train_reward / (total_trades3 + total_trades4)


            val_weight, train_weight = 2, 0
        
            weighted_reward = (avg_train * train_weight + avg_val * val_weight) / (train_weight + val_weight)

            if avg_train == 0:
                good_avg_val_n_train = False
            else:
                good_avg_val_n_train = avg_val >= -10 and avg_train >= -6.5 and abs(avg_val / avg_train) <= 2 and avg_train > avg_val

            delta_val = avg_val - prev_aval
            delta_train = avg_train - prev_atrain

            if weighted_reward < best_avg:
                diverge_val += 1
            else:
                diverge_val = 0

            prev_aval = avg_val
            prev_atrain = avg_train

            if dv.train_1_day:
                min_trades = avg_trades >= 15
                max_trades = avg_trades <= 65
            else:
                min_trades = avg_trades >= 75
                max_trades = avg_trades <= 250

            if min_trades:
                best_avg_val = max(best_avg_val, avg_val) 
                best_avg_train = max(best_avg_train, avg_train)
                bbl = [best_avg, best_avg + 1, weighted_reward, (weighted_reward * 2 + best_avg) / 3]
                bbl.sort()
                best_avg = bbl[-2]

                

            loss = train_result1[3]

            found = diverge_val >= 5
            valid_trades.append(min_trades and max_trades)
            
            if first_iter and weighted_reward != 0:
                first_iter = False
                best_avg = weighted_reward
            else:
                pass
                # found = max_trades and min_trades
            
            if avg_trades == 0:
                zero_trades_t += 1
            else:
                zero_trades_t = 0

            f2 = open(fn, "a")

            print("EPISODE NUMBER: ", episode, file = f2)
            print("Reward Offset: ", reward_offset, file = f2)
            print("AVG Trades: ", avg_trades, file = f2)
            print("VAL reward norm", val_reward_norm, file = f2)
            print("VAL reward inv" , val_reward_inv, file = f2)
            print("TRAIN reward norm", train_reward_norm, file = f2)
            print("TRAIN reward inv" , train_reward_inv, file = f2)

            print("\nTrain Reward", train_reward, file = f2)
            print("Val Reward", val_reward, file = f2)
            print("AVG VAL REWARD", avg_val, file = f2)
            print("AVG TRAIN REWARD", avg_train, file = f2)

            print("\nLOSS: ", loss, file = f2)
            print("zero_trades_t", zero_trades_t, file = f2)
            print("DIVERGING_VAL", diverge_val, file = f2)
            print("BEST AVG: ", best_avg, file = f2)
            print(50 * "*", file = f2)

            prev_train_reward = train_reward
            prev_val_reward = val_reward

            f2.close()
        else:
            val_result, _, ws, ls, tt, mwls = 0, 0, 0, 0, 0, 0

        
        show_train_result(train_result1, 0.0, initial_offset)
        logging.info("Val Stats: --- Train Reward Diff: {:.6f} Validation Reward Diff: {:.6f} Average Train Reward: {:.6f} Average Validation Reward: {:.6f}".format(trd, vrd, train_reward, val_reward))
        print("Val Stats: --- Train Reward Diff: {:.6f} Validation Reward Diff: {:.6f} Average Train Reward: {:.6f} Average Validation Reward: {:.6f}".format(trd, vrd, train_reward, val_reward))
        loss = train_result1[3] # verif_loss
        
        # agent.save(episode)

        weights.append(agent.model.get_weights())

        agent.save(episode)

        ### ### ### ### ### ### ### 
        ### 1D training TEST### ### 
        ### ### ### ### ### ### ### 
        if found and episode >= 10:
            agent.model.set_weights(weights[-6])
            if valid_trades[-6]:
                agent.save("MAX_VAL")
            else:
                agent.save("MAX_VAL_INVALID")
                agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name, mem_len=mlen, ma_state_size=window_size)
                agent.model.set_weights(weights[int((len(weights) - 7) / 2)])
                del weights, train_data, train_data2, val_data, val_data2
                return
                main(train_stock, val_stock, window_size, batch_size, ep_count,
                    strategy, model_name, pretrained,
                    debug, train_stock2, old_model, old_stock, 
                    old_stock_inverse, reward_offset + 0.25, preload_agent = agent)
            return
        elif episode >= (ep_count):
            # agent.save("FINAL_EP_NOT_VAL")
            # return
            agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name, mem_len=mlen, ma_state_size=window_size)
            agent.model.set_weights(weights[int((len(weights) - 0) / 2)])
            # del weights, train_data, train_data2, val_data, val_data2
            # main(train_stock, val_stock, window_size, batch_size, ep_count,
            #     strategy, model_name, pretrained,
            #     debug, train_stock2, old_model, old_stock, 
            #     old_stock_inverse, reward_offset + 0.1, preload_agent = agent)
            return
        elif zero_trades_t == 2:
            # agent.save("NO_TRADES")
            # return
            agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name, mem_len=mlen, ma_state_size=window_size)
            agent.model.set_weights(weights[int((len(weights) - 0) / 2)])
            del weights, train_data, train_data2, val_data, val_data2
            main(train_stock, val_stock, window_size, batch_size, ep_count,
                strategy, model_name, pretrained,
                debug, train_stock2, old_model, old_stock, 
                old_stock_inverse, reward_offset - 0.25, preload_agent = agent)
            return

        ### ### ### ### ### ### ### 
        ### 1W retrain MAIN ### ### 
        ### ### ### ### ### ### ### 


if __name__ == "__main__":
    args = docopt(__doc__)

    train_stock = args["<train-stock>"]
    val_stock = args["<val-stock>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]
    train_stock2 = args["<train-stock2>"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main2(train_stock, val_stock, window_size, batch_size,
            ep_count, strategy=strategy, model_name=model_name, 
            pretrained=pretrained, debug=debug, train_stock2=train_stock2)
    except KeyboardInterrupt:
        print("Aborted!")


