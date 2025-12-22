#!/usr/bin/env python
# coding: utf-8

# In[10]:


#libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import warnings

#warnings
sns.set_style("darkgrid")
warnings.simplefilter(action = "ignore", category = FutureWarning)

#funtions
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
    
    c_p = current_price(ticker)
    even = c_p * 0.001

    if opt_type == "call":
        opt_df = opt_df[opt_df["strike"] > (c_p + even)]
    else:
        opt_df = opt_df[opt_df["strike"] < (c_p - even)]

    vol_threshold = opt_df["volume"].sum() * 0.01
    opt_OTM = opt_df[opt_df["volume"] > vol_threshold]

    return opt_OTM["strike"], opt_OTM["lastPrice"], opt_OTM["volume"], opt_OTM

def volume_grab(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period = "5d")
        if not hist.empty:
            recent_volume = hist["Volume"].iloc[-1]
            avg_volume = hist["Volume"].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            return volume_ratio
    except:
        return (1.0)
        

def calc_implied_move(calls, puts, c_p):
    if calls.empty or puts.empty:
        return None

    closest_call = None
    for _, row in calls.iterrows():
        if row["strike"] > c_p:
            closest_call = row
            break

    closest_put = None
    for _, row in puts.iterrows():
        if row["strike"] < c_p:
            closest_put = row
            break

    if closest_call is not None and closest_put is not None:
        return (closest_call["lastPrice"] + closest_put["lastPrice"]) / c_p * 100

    return None
    

def market_sentiment_grab(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        score = 0

        if "rsi" in info and info["rsi"] is not None:
            rsi = info["rsi"]
            if rsi > 70:
                score -= 0.2
            elif rsi < 30:
                score += 0.2

        if "fiftyDayAverage" in info and "twoHundredDayAverage" in info:
            f50 = info["fiftyDayAverage"]
            f200 = info["twoHundredDayAverage"]
            if f50 is not None and f200 is not None:
                if f50 > f200:
                    score += 0.1
                else:
                    score -= 0.1

        return score
    except:
        return 0
        

def calc_open_interest(calls, puts):
    try:
        if calls.empty or puts.empty:
            return 0.5, 1, 1

        call_oi = calls["openInterest"].sum()
        put_oi = puts["openInterest"].sum()
        total_oi = call_oi + put_oi
        
        if total_oi > 0:
            oi_ratio = call_oi / total_oi
        else:
            oi_ratio = 0.5
            
        call_vtoi = calls["volume"].sum() / max(call_oi, 1)
        put_vtoi = puts["volume"].sum() / max(put_oi, 1)
        
        return oi_ratio, call_vtoi, put_vtoi
    except:
        return 0.5, 1, 1
        

def norm_option(calls, puts, c_p):
    if calls.empty or puts.empty:
        return calls["lastPrice"].mean(), puts["lastPrice"].mean()

    c_strk = calls["strike"]
    c_px = calls["lastPrice"]

    call_normed = []
    i = 0
    while i < len(c_strk):
        s = c_strk.iloc[i]
        p = c_px.iloc[i]
        d = s - c_p
        if d <= 0:
            d = 0.01
        call_normed.append(p / d)
        i += 1

    p_strk = puts["strike"]
    p_px = puts["lastPrice"]

    put_normed = []
    for j in range(len(p_strk)):
        s2 = p_strk.iloc[j]
        p2 = p_px.iloc[j]
        d2 = c_p - s2
        if d2 <= 0:
            d2 = 0.01
        put_normed.append(p2 / d2)

    if len(call_normed) > 0:
        c_out = np.mean(call_normed)
    else:
        c_out = 0

    if len(put_normed) > 0:
        p_out = np.mean(put_normed)
    else:
        p_out = 0

    return c_out, p_out

#main function
def option_spread():
    while True:
        ticker = input("Enter Ticker, Example -> NVDA: 'break' to exit: ").strip().upper()
        if ticker == "BREAK":
            print("Exiting function.")
            return
        try:
            info = yf.Ticker(ticker).info
            if info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
                print("---------------------------------------------------------------------------------------------------------")
                print("!!!ERROR!!!")
                print("Please Enter Vaild Ticker")
                continue
            
            quote_type = info.get("quoteType", "")
            if quote_type == "ETF":
                print("ERROR: ETFs are not allowed. Please enter a STOCK ticker (e.g., NVDA, AAPL, TSLA).")
                continue
                
            break
        except:
            print("Invalid ticker. Please try again.")

    c_p = current_price(ticker)
    if c_p is None:
        print("Could not get current price. Please try again.")
        return
        
    t_now = pd.Timestamp.today().normalize()
    
    volume_ratio = volume_grab(ticker)
    market_sentiment = market_sentiment_grab(ticker)
    
    try:
        exp_dates = yf.Ticker(ticker).options
        if not exp_dates:
            print("No option expiries available for this stock.")
            return
        short_exp = [d for d in exp_dates if 0 < (pd.Timestamp(d) - t_now).days <= 7]
        if not short_exp:
            print("No expiries within 7 days")
            return
    except:
        print("Error getting option dates.")
        return

    all_calls = []
    all_puts = []

    for date in short_exp:
        try:
            call_strike, call_price, call_volume, call_df = opt_OTM(ticker, date, "call")
            put_strike, put_price, put_volume, put_df = opt_OTM(ticker, date, "put")

            OTM_PCT = 0.02
            calls_filtered = call_df[call_df["strike"] >= c_p * (1 + OTM_PCT)]
            puts_filtered  = put_df[put_df["strike"]  <= c_p * (1 - OTM_PCT)]

            all_calls.append(calls_filtered)
            all_puts.append(puts_filtered)
        except:
            continue

    if not all_calls or not all_puts:
        print("Insufficient OTM data for forecast.")
        return

    try:
        calls = pd.concat(all_calls, ignore_index = True)
        puts = pd.concat(all_puts, ignore_index = True)
    except:
        print("Error combining option data.")
        return
    
    if calls.empty or puts.empty:
        print("No OTM options data available.")
        return
        
    implied_move = calc_implied_move(calls, puts, c_p)
    oi_ratio, call_vtoi, put_vtoi = calc_open_interest(calls, puts)

    call_vol = calls["volume"].sum()
    put_vol  = puts["volume"].sum()

    if call_vol + put_vol == 0:
        print("Insufficient OTM volume for forecast.")
        return

    cp_imbalance = (call_vol - put_vol) / (call_vol + put_vol)

    call_pressure = np.sum(calls["volume"] / np.maximum(np.abs(calls["strike"] - c_p), 1e-6))
    put_pressure  = np.sum(puts["volume"]  / np.maximum(np.abs(c_p - puts["strike"]), 1e-6))
    net_pressure = (call_pressure - put_pressure) / (call_pressure + put_pressure + 1e-6)
    
    call_price_norm, put_price_norm = norm_option(calls, puts, c_p)

    if call_price_norm > 0 or put_price_norm > 0:
        denom = call_price_norm + put_price_norm
        if denom <= 0:
            denom = 0.01
        price_skew = (call_price_norm - put_price_norm) / denom
    else:
        price_skew = 0
    
    try:
        if "impliedVolatility" in calls.columns and "impliedVolatility" in puts.columns:
            call_iv_mean = calls["impliedVolatility"].dropna().mean()
            put_iv_mean = puts["impliedVolatility"].dropna().mean()
    
            if call_iv_mean > 0 and put_iv_mean > 0:
                iv_skew = (call_iv_mean - put_iv_mean) / ((call_iv_mean + put_iv_mean) / 2)
                price_skew = 0.7 * iv_skew + 0.3 * price_skew
    except:
        pass
    
    volume_surge = min(volume_ratio, 3)
    volume_factor = (volume_surge - 1) * 0.1
    
    oi_factor = (oi_ratio - 0.5) * 2
    
    vtoi_factor = np.tanh((call_vtoi - put_vtoi) * 0.5)

    cp_n   = np.tanh(cp_imbalance)
    pres_n = np.tanh(net_pressure)
    skew_n = np.tanh(price_skew * 2)
    
    score = (0.35 * cp_n + 
             0.25 * pres_n + 
             0.15 * skew_n + 
             0.10 * market_sentiment + 
             0.08 * oi_factor + 
             0.07 * vtoi_factor)
    
    confidence_level = min(1.0, (call_vol + put_vol) / 1000)
    confidence_text = ""
    if confidence_level < 0.3:
        confidence_text = "LOW CONFIDENCE (Low Volume)"
    elif confidence_level < 0.7:
        confidence_text = "MODERATE CONFIDENCE"
    else:
        confidence_text = "HIGH CONFIDENCE"

    if score > 0.35:
        forecast = "BULLISH"
    elif score < -0.35:
        forecast = "BEARISH"
    elif score > 0.15:
        forecast = "Mildly Bullish"
    elif score < -0.15:
        forecast = "Mildly Bearish"
    else:
        forecast = "NEUTRAL"

    conf_text = confidence_text
    
    if implied_move is not None:
        conf_text += f"\nImplied Move: ±{implied_move:.1f}%"

    both_strike = np.concatenate((puts["strike"], calls["strike"]))
    both_price  = np.concatenate((puts["lastPrice"], calls["lastPrice"]))
    both_volume = np.concatenate((puts["volume"], calls["volume"]))

    fixed_volume = np.nan_to_num(both_volume, nan = 0.0)
    volume_log = np.log1p(fixed_volume)
    
    if len(volume_log) > 0:
        vmin = volume_log.min()
        vmax = volume_log.max()
        if vmax > vmin:
            volume_scaled = (volume_log - vmin) / (vmax - vmin)
        else:
            volume_scaled = np.zeros_like(volume_log)
    else:
        volume_scaled = np.array([])

    if len(both_strike) > 1:
        x_sorted = np.sort(np.array(both_strike))
        dx = np.diff(x_sorted)
        if len(dx) > 0:
            min_spacing = dx.min()
        else:
            min_spacing = 1
        bar_width = min_spacing * 0.4
    else:
        bar_width = 0.4
    
    call_mask = both_strike > c_p
    put_mask = both_strike < c_p

    fig, ax = plt.subplots(figsize = (14, 7))
    
    if np.any(call_mask):
        ax.bar(both_strike[call_mask], both_price[call_mask], 
               width = bar_width, color = "green", alpha = 0.6, 
               edgecolor = "darkgreen", label = "Calls")
    
    if np.any(put_mask):
        ax.bar(both_strike[put_mask], both_price[put_mask], 
               width = bar_width, color = "red", alpha = 0.6, 
               edgecolor = "darkred", label = "Puts")
    
    ax.axvline(x = float(c_p), color = "black", linestyle = '--', 
               linewidth = 1.5, label = "Current Price")

    c_p_20 = c_p * 0.15
    under20 = c_p - c_p_20
    over20 = c_p + c_p_20
    ax.axvline(x = under20, color = "red", linestyle = ':', 
               linewidth = 1, alpha = 0.7, label = f"-{c_p_20/c_p*100:.1f}%")
    ax.axvline(x = over20, color = "green", linestyle = ':', 
               linewidth = 1, alpha = 0.7, label = f"+{c_p_20/c_p*100:.1f}%")

    ax.set_xlabel("Strike Price (USD)")
    ax.set_ylabel("Option Price (USD)")
    
    title = f"{ticker} OTM Option Prices vs Strike Price (≤7D Expiry)"
    if volume_ratio > 1.5:
        title += " - HIGH VOLUME"
        ax.set_title(title, fontweight = 'bold', color = 'blue')
    else:
        ax.set_title(title)

    call_pct = (call_vol / (call_vol + put_vol)) * 100
    put_pct = (put_vol / (call_vol + put_vol)) * 100

    if put_price_norm > 0:
        raw_call_put_ratio = call_price_norm / put_price_norm
    else:
        raw_call_put_ratio = float("inf")
    
    forecast_box = ax.text(0.01, 0.95,
            f"FORECAST: {forecast}\n"
            f"Score: {score:.3f}\n"
            f"{conf_text}\n"
            f"Call Vol: {call_vol:,} ({call_pct:.1f}%)\n"
            f"Put Vol: {put_vol:,} ({put_pct:.1f}%)\n"
            f"OI Ratio: {oi_ratio*100:.1f}% Calls\n"
            f"Volume Factor: {volume_ratio:.2f}x"
            f"Norm. C/P Ratio: {raw_call_put_ratio:.2f}\n"
            f"Price Skew: {price_skew:.3f}",
            ha = "left", va = "top",
            transform = ax.transAxes,
            fontsize = 11,
            color = "black",
            bbox = dict(facecolor = "white", alpha = 0.9, 
                       edgecolor = "black", boxstyle = "round,pad=0.5"))
    
    if market_sentiment != 0:
        ax.text(0.01, 0.60,
                f"Market Sentiment: {'+' if market_sentiment > 0 else ''}{market_sentiment:.2f}",
                ha = "left", va = "top",
                transform = ax.transAxes,
                fontsize = 9,
                color = "blue",
                bbox = dict(facecolor = "lightblue", alpha = 0.7, edgecolor = "blue"))

    if len(volume_scaled) > 0 and len(both_price) > 0:
        if both_price.max() > 0:
            ax.plot(both_strike, volume_scaled * both_price.max(), 
                   alpha = 0.7, linewidth = 0.7, label = "Volume Scaled")

    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()

option_spread()


# In[ ]:




