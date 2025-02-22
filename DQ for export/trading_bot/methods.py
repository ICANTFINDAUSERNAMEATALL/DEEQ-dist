import os
import logging

import numpy as np
import pandas as pd
import random
import math
import device_vars as dv

from tqdm import tqdm

from inversedata import inv_from_pd

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state,
    get_invs_state
)

"""
SETTINGS FOR THE TRADING METHOD
"""

reward_mult = 1
sl_amount = 25 * reward_mult
tp_amount = 50 * reward_mult

"""
END SETTINGS
"""

reward_tree = {}

def train_model4(agent, episode, data, ep_count=100, batch_size=64, window_size=10, 
                data2 = None, epoch_for_mass_training=0, ma_state_size=15, reward_offset = 0,
                train_interval = 500, random_state_eps = 3):

    agent.episode_number = episode

    global reward_tree
    iternum = 0
    f = open("print.txt", "a")
    total_profit = 0
    data_length = len(data) - 1

    if data2 == None:
        raise Exception("No data2 in model, data2 needed to run model")

    agent.inventory = []
    agent.i2 = [] # when to remove the "trade" from the inventory
    """
    agent.inventory structure:
    -> buy: [long, entry price, sl, tp]
    -> sell: [short, entry price, sl, tp]
    """
    avg_loss = []
    acculs = []

    def calc_tp_sl(t, data, max_t, direction, sl, tp, sl_amt = sl_amount, tp_amt = tp_amount):
        t2 = t
        raw_reward = 0
        run_run = True
        candle_counter = 0
        while t2 < max_t - 1 and run_run:  # Corrected loop condition (see point 2 below)
            o, h, l, c, v = data[t2]
            if direction == "LONG":
                if l < sl:  # Check Low for SL hit (LONG)
                    raw_reward = - sl_amt
                    run_run = False
                elif h > tp: # Check High for TP hit (LONG)
                    raw_reward = tp_amt
                    run_run = False
            elif direction == "SHORT":
                if h > sl:  # Check High for SL hit (SHORT)
                    raw_reward = - sl_amt
                    run_run = False
                elif l < tp:  # Check Low for TP hit (SHORT)
                    raw_reward = tp_amt
                    run_run = False
            else:
                raise Exception("INVALID DIRECTION")
            if not run_run:
                break # Break outer loop once SL/TP is hit
            t2 += 1
            candle_counter += 1
        return raw_reward, candle_counter + 1

    def _ema(prices, days, smoothing=2):
        # ema = [sum(prices[:days]) / days]
        ema = [prices[0]]
        for i in range(1, days):
            ema.append(sum(prices[:i]) / i)
        for price in prices[days:]:
            ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
        return ema
    
    def _sma(prices, days):
        sma = [prices[0]]
        for i in range(1, days):
            sma.append(sum(prices[: i]) / i)
        for i in range(days, len(prices)):
            sma.append(sum(prices[i-days + 1: i + 1]) / days)
        return sma

    def _llst(values, length):
        vr = []
        if len(values) < length:
            for i in range(len(values)):
                vr.append(values[:i + 1] + (length - i - 1) * [0])
            return vr
        
        for i in range(length):
            vr.append(values[:i + 1] + (length - i - 1) * [0])
        for i in range(length, len(values)):
            vr.append(values[i - length: i])
        return vr

    closes_norm = []
    closes_inv = []

    for i in data:
        closes_norm.append(i[3])
    for i in data2:
        closes_inv.append(i[3])

    ema_15_norm_dif = [i - j for i, j in zip(_ema(closes_norm, 15), closes_norm)]
    sma_50_norm_dif = [i - j for i, j in zip(_sma(closes_norm, 50), closes_norm)] # actually 50

    ema_15_inv_dif = [i - j for i, j in zip(_ema(closes_inv, 15), closes_inv)]
    sma_50_inv_dif = [i - j for i, j in zip(_sma(closes_inv, 50), closes_inv)] # actually 50

    ema_15_norm_dif_l = _llst(ema_15_norm_dif, ma_state_size)
    sma_50_norm_dif_l = _llst(sma_50_norm_dif, ma_state_size)
    norm_mas_lst = [[i, j] for i, j in zip(ema_15_norm_dif_l, sma_50_norm_dif_l)]

    ema_15_inv_dif_l = _llst(ema_15_inv_dif, ma_state_size)
    sma_50_inv_dif_l = _llst(sma_50_inv_dif, ma_state_size)
    inv_mas_lst = [[i, j] for i, j in zip(ema_15_inv_dif_l, sma_50_inv_dif_l)]

    buys = 0
    sells = 0
    data2_length = len(data2) - 1

    data1 = data.copy() # data2 the data to alternate to

    s0l   = []
    s1l   = []
    invsl = []
    datap = []
    next_states0 = []
    next_states1 = []
    sorce_num = []

    retrain_int = batch_size
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # Adds multiple different states based on different inventories ###
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


    for io in tqdm(range(data_length + data2_length), total=(data_length + data2_length), leave=True):
        for iz in range(1 + random_state_eps):
            reward = 0
            if io % 2 == 0:
                t = int(io / 2)
                data = data1
                state0, state1 = get_state(data, t, window_size + 1, 0, norm_mas_lst[t], 0) # gets random inv state
                next_state0, next_state1_noact = get_state(data, t + 1, window_size + 1, 0, inv_mas_lst[t], 0)
                next_state0, next_state1_act_win   = get_state(data, t + 1, window_size + 1, 0, inv_mas_lst[t], 0)
                next_state0, next_state1_act_loss  = get_state(data, t + 1, window_size + 1, 0, inv_mas_lst[t], 0)
                next_states0.append(next_state0)
                next_states1.append((next_state1_noact, next_state1_act_win, next_state1_act_loss))
                sorce_num.append(1)
            else:
                t = int((io - 1) / 2)
                data = data1
                state0, state1 = get_state(data, t, window_size + 1, 0, norm_mas_lst[t], 0) # gets random inv state
                next_state0, next_state1_noact = get_state(data, t + 1, window_size + 1, 0, inv_mas_lst[t], 0)
                next_state0, next_state1_act_win   = get_state(data, t + 1, window_size + 1, 0, inv_mas_lst[t], 0)
                next_state0, next_state1_act_loss  = get_state(data, t + 1, window_size + 1, 0, inv_mas_lst[t], 0)
                next_states0.append(next_state0)
                next_states1.append((next_state1_noact, next_state1_act_win, next_state1_act_loss))
                sorce_num.append(2)

            # print(len(next_state1_noact[0]), len(next_state1_act[0]))
            s0l.append(state0)
            s1l.append(state1)

        if io % retrain_int == 0 and io != 0:
            actionl = agent._act(s0l, s1l)
            # fn = open("data.txt", "a")
            for index in range(-1, 1 - len(s0l), -1):
                j, k, y, dp, ns0, ns1, cur_act , sn = s0l[index], s1l[index], invsl[index], datap[index], next_states0[index], next_states1[index], actionl[index], sorce_num[index]
                
                data, data_length, t = dp
                if cur_act == 1:
                    # BUY
                    reward, cd = calc_tp_sl(t + 1, data, data_length, "LONG", data[t][3] - sl_amount, data[t][3] + tp_amount)
                elif cur_act == 2:
                    # SELL
                    reward, cd = calc_tp_sl(t + 1, data, data_length, "SHORT", data[t][3] + sl_amount, data[t][3] - tp_amount)
                else:
                    reward = reward_offset
                    cd = 0
                    nr = 0

                done = (t == data_length - 1)

                agent.remember(j, k, cur_act, reward, ns0, ns1, done, sn)

            if dv.has_gpu:
                loss = agent.train_experience_replay_gpu(batch_size * 4, epochs = 1)
            else:
                loss = agent.train_experience_replay_2(batch_size * 4, epochs = 1)
            if loss == None:
                pass
            else:
                avg_loss.append(loss)

            s0l   = []
            s1l   = []
            invsl = []
            datap = []
            next_states0 = []
            next_states1 = []
            inum = -1

    ### ### ### ### ### ### ### ### ### ### ### ### ###
    # Calculate full reward for each possible state ###
    ### ### ### ### ### ### ### ### ### ### ### ### ###
    f.close()
    return [episode, ep_count, total_profit, np.mean(np.array(avg_loss)), 1]

def evaluate_model_speed(agent, data, window_size, debug, ma_state_size):
    agent.i2 = []
    agent.inventory = []
    iternum = 0
    total_profit = 0
    data_length = len(data) - 1

    history = []
    """
    agent.inventory structure:
    -> buy: [long, entry price, sl, tp]
    -> sell: [short, entry price, sl, tp]
    """
    wins = 0
    losses = 0
    trades = 0
    tlo = []
    max_dd = 0
    peak = 0
    ec = []
    bsl = []
    tp_sl_date = ""

    def calc_tp_sl(t, data, max_t, direction, sl, tp, sl_amt = sl_amount, tp_amt = tp_amount):
        t2 = t
        raw_reward = 0
        run_run = True
        candle_counter = 0
        while t2 < max_t - 1 and run_run:  # Corrected loop condition (see point 2 below)
            o, h, l, c, v = data[t2]
            if direction == "LONG":
                if l < sl:  # Check Low for SL hit (LONG)
                    raw_reward = - sl_amt
                    run_run = False
                elif h > tp: # Check High for TP hit (LONG)
                    raw_reward = tp_amt
                    run_run = False
            elif direction == "SHORT":
                if h > sl:  # Check High for SL hit (SHORT)
                    raw_reward = - sl_amt
                    run_run = False
                elif l < tp:  # Check Low for TP hit (SHORT)
                    raw_reward = tp_amt
                    run_run = False
            else:
                raise Exception("INVALID DIRECTION")
            if not run_run:
                break # Break outer loop once SL/TP is hit
            t2 += 1
            candle_counter += 1
        return raw_reward, candle_counter + 1

    def _ema(prices, days, smoothing=2):
        # ema = [sum(prices[:days]) / days]
        ema = [prices[0]]
        for i in range(1, days):
            ema.append(sum(prices[:i]) / i)
        for price in prices[days:]:
            ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
        return ema
    
    def _sma(prices, days):
        sma = [prices[0]]
        for i in range(1, days):
            sma.append(sum(prices[: i]) / i)
        for i in range(days, len(prices)):
            sma.append(sum(prices[i-days + 1: i + 1]) / days)
        return sma

    def _llst(values, length):
        vr = []
        if len(values) < length:
            for i in range(len(values)):
                vr.append(values[:i + 1] + (length - i - 1) * [0])
            return vr
        
        for i in range(length):
            vr.append(values[:i + 1] + (length - i - 1) * [0])
        for i in range(length, len(values)):
            vr.append(values[i - length: i])
        return vr

    closes_norm = []

    for i in data:
        closes_norm.append(i[3])
    
    ema_15_norm_dif = [i - j for i, j in zip(_ema(closes_norm, 15), closes_norm)]
    sma_50_norm_dif = [i - j for i, j in zip(_sma(closes_norm, 50), closes_norm)]

    ema_15_norm_dif_l = _llst(ema_15_norm_dif, ma_state_size)
    sma_50_norm_dif_l = _llst(sma_50_norm_dif, ma_state_size)
    norm_mas_lst = [i+j for i, j in zip(ema_15_norm_dif_l, sma_50_norm_dif_l)]

    state0, state1 = get_state(data, 0, window_size + 1, 0, norm_mas_lst[0], 0)

    buys = 0
    sells = 0
    candnum = 0
    state_lst = []
    action_lst = []
    reward_lst = []
    next_state_lst = []
    done_lst = []
    s0l = []
    s1l = []
    invsl = []

    final_reward = 0


    for t in range(data_length):
        ### ### ### ### ### ### ### ### ### ### ### ### ###
        # Get the current state with updated inventory  ###
        ### ### ### ### ### ### ### ### ### ### ### ### ###
        candnum += 1
        reward = 0
        s0l.append(state0)
        s1l.append(state1)
        next_state0, next_state1 = get_state(data, t + 1, window_size + 1, 0, norm_mas_lst[t + 1], 0)
        state0 = next_state0
        state1 = next_state1
    
    actl = agent._act(s0l, s1l, True)

    for t, actact, y, state0, state1 in tqdm(zip(range(data_length), actl, invsl, s0l, s1l)):
        ### ### ### ### ### ### ### ### ### ### ###
        # Select an action and calculate reward ###
        ### ### ### ### ### ### ### ### ### ### ###
        action = actact
        action_lst.append(action)
        reward = 0

        # BUY
        if action == 1:
            buys += 1
            bsl.append("BUY")
            reward, cd = calc_tp_sl(t + 1, data, data_length, "LONG", data[t][3] - sl_amount - eval_slippage, data[t][3] + tp_amount + eval_slippage)
            total_profit += reward
            if reward > 0:
                wins += 1
                tlo.append("WIN")
            else:
                losses += 1
                tlo.append("LOSS")
            trades += 1

            history.append((data[t], "BUY"))
            if debug:
                pass
                # logging.debug("Buy at: {}".format(format_currency(data[t][3])))
        # SELL
        elif action == 2:
            sells += 1
            bsl.append("SELL")
            reward, cd = calc_tp_sl(t + 1, data, data_length, "SHORT", data[t][3] + sl_amount + eval_slippage, data[t][3] - tp_amount - eval_slippage)
            total_profit += reward
            if reward > 0:
                wins += 1
                tlo.append("WIN")
            else:
                losses += 1
                tlo.append("LOSS")
            trades += 1

            history.append((data[t], "SELL"))
            if debug:
                pass

        else:
            history.append((data[t], "HOLD"))

        final_reward += reward

        if reward > 50:
            raise

        reward_lst.append(reward)

        peak = max(peak, total_profit)
        max_dd = min(max_dd, total_profit - peak)
        
        if action == 1 or action == 2:
            ec.append(total_profit)

        ### ### ### ### ### ### ### ### ### ###
        # Get next state given the action   ###
        ### ### ### ### ### ### ### ### ### ###

        done = (t == data_length - 1)
        done_lst.append(done)

        if done:
            mws = 0
            ws = 0
            mls = 0
            ls = 0
            for i in tlo:
                if i == "WIN":
                    ws += 1
                    mls = max(mls, ls)
                    ls = 0
                if i == "LOSS":
                    ls += 1
                    mws = max(mws, ws)
                    ws = 0

            batch = []
            for a, b, c, d, e in zip(state_lst, action_lst, reward_lst, next_state_lst, done_lst):
                batch.append([a, b, c, d, e])
            return total_profit, history, wins, losses, trades, [mws, mls, max_dd, ec, buys, sells, batch, bsl], final_reward
