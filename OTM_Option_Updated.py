#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import Normalize
import warnings
sns.set_style("darkgrid")
warnings.simplefilter(action = "ignore", category = FutureWarning)

#functions
def current_price(ticker):
    stock = yf.Ticker(ticker)
    last_price = stock.fast_info["last_price"]
    return last_price

def call_and_put(ticker, date, opt_type):
    stock = yf.Ticker(ticker)
    opt = stock.option_chain(date)
    if opt_type == "call":
        return pd.DataFrame(opt.calls)
    elif opt_type == "put":
        return pd.DataFrame(opt.puts)
    else:
        print("Enter call or put")

def opt_OTM(ticker, date, opt_type):
    opt = call_and_put(ticker, date, opt_type)
    opt_df = opt[opt["inTheMoney"] == False]

    tick = yf.Ticker(ticker)
    info = tick.info
    c_p = current_price(ticker)
    even = c_p * 0.001

    if opt_type == "call":
        opt_df = opt_df[opt_df["strike"] > (c_p + even)]
    else:
        opt_df = opt_df[opt_df["strike"] < (c_p - even)]

    volume_1 = opt_df["volume"]
    vol = volume_1.sum()*0.01
    opt_OTM = opt_df[opt_df["volume"] > vol]
    volume = opt_OTM["volume"]
    strike = opt_OTM["strike"]
    last_price = opt_OTM["lastPrice"]
    return (strike, last_price, volume, opt_OTM)

def RSI(ticker):
    close = yf.download(ticker, period = "19d")["Close"]
    delta = close.diff()
    rs = delta.clip(lower = 0).rolling(14).mean() / (-delta.clip(upper = 0).rolling(14).mean())
    return round((100 - 100 / (1 + rs)).iloc[-1].item(), 2)

def option_spread():
    #enter ticker
    while True:
        ticker = input("Enter Ticker, Example -> NVDA:").strip()
        info = yf.Ticker(ticker).info
        if info.get('regularMarketPrice') is not None:
            break
        elif ticker == "break":
            break
        else:
            print("---------------------------------------------------------------------------------------------------------")
            print("!!!ERROR!!!")
            print("Please Enter Vaild Ticker")

    #break
    if ticker == "break":
        return

    #grabbing valid expiry dates
    exp_dates = yf.Ticker(ticker).options

    #enter date
    while True:
        print(exp_dates)
        date = input("Select Expiry date (year-month-date):").strip()
        if date in exp_dates:
            break
        elif date == "break":
            break
        else:
            print("---------------------------------------------------------------------------------------------------------")
            print("!!!ERROR!!!")
            print("Please Select Vaild Expiry date")

    #break
    if date == "break":
        return

    #strikes and prices
    call_strike, call_price, call_volume, call_df = opt_OTM(ticker, date, "call")
    put_strike, put_price, put_volume, put_df = opt_OTM(ticker, date, "put")
    c_p = current_price(ticker)


    tick = yf.Ticker(ticker)
    info = tick.info

    if info.get("quoteType") == "ETF":
        c_p_20 = c_p * 0.03
        under20 = c_p - c_p_20
        over20 = c_p + c_p_20
    else:
        c_p_20 = c_p * 0.15
        under20 = c_p - c_p_20
        over20 = c_p + c_p_20

    under = call_df[call_df["strike"] <= over20]
    over = put_df[put_df["strike"] >= under20]
    over_mean = (under["lastPrice"].sum())/len(under)
    under_mean = (over["lastPrice"].sum())/len(over)
    print(over_mean)
    print(under_mean)


    #finding total volume calls and puts together
    total_volume = call_volume.sum() + put_volume.sum()

    #concatenate both OTM calls and puts
    both_strike = np.concatenate((put_strike, call_strike))
    both_price = np.concatenate((put_price, call_price))
    both_volume = np.concatenate((put_volume, call_volume))

    #fidning put and call volume percentages
    call_pct = (call_volume.sum()/total_volume.sum())*100
    put_pct = (put_volume.sum()/total_volume.sum())*100

    #replacing NAN values
    fixed_volume = np.nan_to_num(both_volume, nan=0.0)

    #log(1 + x)
    volume_log = np.log1p(fixed_volume)

    #normalize using only the log values
    vmin = volume_log.min()
    vmax = volume_log.max()
    volume_scaled = (volume_log - vmin) / (vmax - vmin)

    #no overlaping bars
    x_sorted = np.sort(np.array(both_strike))
    dx = np.diff(x_sorted)
    min_spacing = dx.min()
    bar_width = min_spacing * 0.4

    #setting colors for bars
    colors = plt.cm.magma(volume_scaled)

    #plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(both_strike, both_price, width = bar_width, color = colors, edgecolor = colors, alpha = 0.8)
    ax.axvline(x = float(c_p), color = "black", linestyle = '--', linewidth = 0.8, alpha = 1, label = "ATM")
    ax.axvline(x = float(under20), color = "black", linestyle = '--', linewidth = 0.8, alpha = 1, label = "Average Under")
    ax.axvline(x = float(over20), color = "black", linestyle = '--', linewidth = 0.8, alpha = 1, label = "Average Over")
    ax.set_xlabel("Strike Price (USD)")
    ax.set_ylabel("Option Price (USD)")
    ax.set_title(ticker + " OTM Option Prices vs Strike Price")

    #plotting heat spectrum
    sm = mpl.cm.ScalarMappable(cmap = plt.cm.inferno)
    cbar = fig.colorbar(sm, ax = ax, orientation = "vertical")
    cbar.set_ticks([]) 
    cbar.set_label("Volume Weight")

    #volume plot
    ax.plot(both_strike, volume_scaled*both_price.max(), alpha = 0.7, linewidth=0.7, label = "Volume")

    #text
    ax.text(0.99, 0.85, f"{round(put_pct)}% OTM Puts {round(call_pct)}% OTM Calls", ha = "right", va = "top", transform = ax.transAxes, fontsize = 12, color = "red")
    ax.text(0.99, 0.80, f"RSI = {RSI(ticker)}", ha="right", va="top", transform=ax.transAxes, fontsize=12, color="red")
    ax.text(0.99, 0.95, f"over = {over_mean}", ha="right", va="top", transform=ax.transAxes, fontsize=12, color="red")
    ax.text(0.99, 0.90, f"under = {under_mean}", ha="right", va="top", transform=ax.transAxes, fontsize=12, color="red")


    #show
    plt.legend()
    plt.show()

option_spread()


# In[ ]:




