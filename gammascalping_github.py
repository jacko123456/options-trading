#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import re
import schedule
import time
import scipy.stats as si
from scipy.stats import norm
from scipy.optimize import newton
import yfinance as yf
from IPython.display import display, HTML
import json

# Configuration
mode = 'Paper'  # 'Live' or 'Paper'
ticker = 'TSLA' 

# option chain DTE filters
start = 7 # day(s)
end = 10 # day(s)

# function frequency
hedge = 30 # delta hedging frequency (seconds)
status = 90 # account update notification frequency (minutes)

# schedule closing functions (military time)
hour = 15 # stop hedging at (hour)
minute = 57 # stop hedging at (minute) - should be 1 minute before close
close = "15:58"
liquidate = "15:59"
stop = "16:00" 

def acct_id(mode):
    if mode == 'Live':
        id = 'LIVE ID HERE' 
    elif mode == 'Paper':
        id = 'PAPER ID HERE' 
    return id
def access_tkn(mode):
    if mode == 'Live':
        token = 'LIVE TOKEN HERE' 
    elif mode == 'Paper':
        token = 'PAPER TOKEN HERE' 
    return token
def base_url(mode):
    if mode == 'Live':
        type = 'https://api.tradier.com/v1/' 
    elif mode == 'Paper':
        type = 'https://sandbox.tradier.com/v1/' 
    return type

### initialize variables
account_id = acct_id(mode)
access_token = access_tkn(mode)
api_base_url = base_url(mode)
headers={'Authorization': 'Bearer {}'.format(access_token), 'Accept': 'application/json'}
current_date = date.today()

### API link extensions
get_account = '{}user/profile'.format(api_base_url) # get user profile
get_balances = '{}accounts/{}/balances'.format(api_base_url, account_id) # get account balances
get_positions = '{}accounts/{}/positions'.format(api_base_url, account_id) # get positions
stock_url = '{}markets/quotes'.format(api_base_url) # get stock quotes
options_lookup = '{}/markets/options/lookup'.format(api_base_url) # get options symbols
options_url = '{}markets/options/chains'.format(api_base_url) # get options chain
option_quotes = '{}markets/quotes'.format(api_base_url) # get option quotes 
option_order_url = '{}accounts/{}/orders'.format(api_base_url, account_id) # execute option order
stock_order_url = '{}accounts/{}/orders'.format(api_base_url, account_id) # execute stock order
check_orders = '{}accounts/{}/orders'.format(api_base_url, account_id) # check orders

if ticker in ['SPY', 'SPX', 'XSP', 'QQQ', 'VIX', 'UXVY']:
    trade_type = 'Index'
else:
    trade_type = 'Single Stock'

equity_statusrep = requests.get(get_balances, params={}, headers=headers)
equity_status = equity_statusrep.json()
equity = equity_status['balances']['total_equity']

print("")
display(HTML(f'<b>{trade_type} Gamma Scalping</b>'))
print("")
print(f"Date: {current_date}")
print(f"Underlying: {ticker}")
print(f"Bankroll: {equity}")
print("")

print("")
print("=" * 141)
print("")

def UserAccount():
    display(HTML('<b>ACCOUNT STATUS</b>'))
    print("")
    
    get_account_response = requests.get(get_account, params={}, headers=headers)
    account_info = get_account_response.json()
    print(f"API Response Status: {get_account_response.status_code}")
    display(HTML('<b>Account Information</b>'))
    print(json.dumps(account_info, indent=4, sort_keys=True))
    print("")
    get_balances_response = requests.get(get_balances, params={}, headers=headers)
    acct_balances = get_balances_response.json()
    print(f"API Response Status: {get_balances_response.status_code}")
    display(HTML('<b>Account Balances</b>'))
    print(json.dumps(acct_balances, indent=4, sort_keys=True))
    print("")

    print("")
    print("=" * 141)
    print("")

def OpenOrders():
    display(HTML('<b>BUY STRADDLE</b>'))
    
    global order_size, call, put, call_id, put_id
    
    stock_url_response = requests.get(stock_url, params={'symbols': ticker}, headers=headers)
    stock_quotes = stock_url_response.json()
    last_price = stock_quotes['quotes']['quote']['last']
    print(f"{ticker} Underlying Price: {last_price}")
    print("")
    options_lookup_response = requests.get(options_lookup, params={'underlying': ticker}, headers=headers)
    optionsymbols_json_response = options_lookup_response.json()
    option_symbols = optionsymbols_json_response.get('symbols', [{}])[0].get('options', [])

    def extract_strike_price(option_symbol):
        if len(ticker) == 2:
             strike_price_str = option_symbol[10:]
        if len(ticker) == 3:
             strike_price_str = option_symbol[11:]
        if len(ticker) == 4:
             strike_price_str = option_symbol[12:]
        strike_price = float(strike_price_str) / 1000
        return strike_price

    def extract_expiration_date(option_symbol):
        if len(ticker) == 2:
             expiration_str = option_symbol[2:8]
        if len(ticker) == 3:
             expiration_str = option_symbol[3:9]
        if len(ticker) == 4:
            expiration_str = option_symbol[4:10]
        expiration_date = datetime.strptime(expiration_str, '%y%m%d').date()
        return expiration_date

    if ticker == 'SPY':
        k = int(last_price)
    else:
        interval = 2.50 
        k = round(last_price / interval) * interval

    today = datetime.now().date()
    start_range = today + timedelta(days=start) 
    end_range = today + timedelta(days=end) 
    
    close_strike_options = [
        symbol for symbol in option_symbols
        if abs(extract_strike_price(symbol) - last_price) <= 10 and start_range <= extract_expiration_date(symbol) <= end_range
    ]

    calls = []
    puts = []
    
    for sym in close_strike_options:
        if 'C' in sym[5:]:
            calls.append(sym)
        if 'P' in sym[5:]:
            puts.append(sym)

    closest_call_diff = float('inf') 
    closest_put_diff = float('inf')  
    call = None  
    put = None  

    for call_sym in calls:
        strike_price = extract_strike_price(call_sym)
        diff = abs(strike_price - last_price)
        if diff < closest_call_diff:
            closest_call_diff = diff
            call = call_sym

    for put_sym in puts:
        strike_price = extract_strike_price(put_sym)
        diff = abs(strike_price - last_price)
        if diff < closest_put_diff:
            closest_put_diff = diff
            put = put_sym

    call_quotes = requests.get(option_quotes,
    params={'symbols': call, 'greeks': 'false'}, headers=headers)
    call_price = call_quotes.json()
    call_bid = call_price['quotes']['quote']['bid']
    call_ask = call_price['quotes']['quote']['ask']
    call_theo_calc = call_bid + ((call_ask - call_bid) / 2)
    call_theo = round(call_theo_calc, 2)

    put_quotes = requests.get(option_quotes,
    params={'symbols': put, 'greeks': 'false'}, headers=headers)
    put_price = put_quotes.json()
    put_bid = put_price['quotes']['quote']['bid']
    put_ask = put_price['quotes']['quote']['ask']
    put_theo_calc = put_bid + ((put_ask - put_bid) / 2)
    put_theo = round(put_theo_calc, 2)
    
    balances = requests.get(get_balances, params={}, headers=headers)
    current_balances = balances.json()
    bankroll = current_balances['balances']['total_equity']

    margin = (put_theo * 100) + (call_theo * 100) + (last_price * 5) # adjustable
        
    order_size = bankroll // margin

    print(call_price['quotes']['quote']['description'])
    print(f"Call Bid: {call_bid}, Call Ask: {call_ask}")
    print(f"Call Theo: {call_theo}")
    print(f"Quantity: {order_size}")
    print("")

    print(put_price['quotes']['quote']['description'])
    print(f"Put Bid: {put_bid}, Put Ask: {put_ask}")
    print(f"Put Theo: {put_theo}")
    print(f"Quantity: {order_size}")
    print("")

    positions_response = requests.get(get_positions, params={}, headers=headers)
    positions = positions_response.json()
    print(f"API Response Status: {positions_response.status_code}")
    print("OPEN POSITIONS")
    print("")
    print(json.dumps(positions, indent=4, sort_keys=True))
    print("")

    call_filled = False
    put_filled = False

    if positions['positions'] != 'null':
        if not isinstance(positions['positions']['position'], list):
            positions['positions']['position'] = [positions['positions']['position']]
            
        for position in positions['positions']['position']:
            if ticker in position['symbol'] and len(position['symbol']) >= 5:
                if 'C' in position['symbol'][5:]:
                    call_filled = True
            if ticker in position['symbol'] and len(position['symbol']) >= 5:
                if 'P' in position['symbol'][5:]:
                    put_filled = True
    
    if call and put:  
        call_strike = extract_strike_price(call)
        put_strike = extract_strike_price(put)

    if call_strike == put_strike and call_filled == False and put_filled == False:

        call_order_req = requests.post(option_order_url,
            data={'class': 'option', 'symbol': ticker, 'option_symbol': call, 'side': 'buy_to_open', 'quantity': order_size, 'type': 'limit', 'duration': 'day', 'price': call_ask, 'tag': 'my-tag-example-1'},
            headers=headers
        )
        call_order = call_order_req.json()
        call_id = call_order['order']['id']
        print(f"API Response Status: {call_order_req.status_code}")
        print(json.dumps(call_order, indent=4, sort_keys=True))
        print("")

        put_order_req = requests.post(option_order_url,
            data={'class': 'option', 'symbol': ticker, 'option_symbol': put, 'side': 'buy_to_open', 'quantity': order_size, 'type': 'limit', 'duration': 'day', 'price': put_ask, 'tag': 'my-tag-example-1'},
            headers=headers
        )
        put_order = put_order_req.json()
        put_id = put_id = put_order['order']['id']
        print(f"API Response Status: {put_order_req.status_code}")
        print(json.dumps(put_order, indent=4, sort_keys=True))
        print("")

    print("")
    print("=" * 141)
    print("")
    
def DeltaHedge():
    time = datetime.now()

    if time.hour >= 15 and time.minute >= 57:
        return

    display(HTML('<b>DELTA HEDGE</b>'))
    print("")
    
    positions_response = requests.get(get_positions, params={}, headers=headers)
    positions = positions_response.json()
    print(f"API Response Status: {positions_response.status_code}")
    print("OPEN POSITIONS")
    print("")
    print(json.dumps(positions, indent=4, sort_keys=True))
    print("")

    if positions['positions'] != 'null':

        C = None
        P = None
    
        if not isinstance(positions['positions']['position'], list):
            positions['positions']['position'] = [positions['positions']['position']]

        for pos in positions['positions']['position']:
            if 'C' in pos['symbol'][4:]:
                C = pos['symbol']
            if 'P' in pos['symbol'][4:]:
                P = pos['symbol']

            if C is not None:
                get_call_rep = requests.get(option_quotes, params={'symbols': C, 'greeks': 'false'}, headers=headers)
                get_call = get_call_rep.json()
            if P is not None:
                get_put_rep = requests.get(option_quotes, params={'symbols': P, 'greeks': 'false'}, headers=headers)
                get_put = get_put_rep.json()

            def call_calculation(S, K, T, r, q, call_price):
                d1 = (np.log(S / K) + (r - q + (0.2 ** 2) / 2) * T) / (0.2 * np.sqrt(T))
                d2 = d1 - 0.2 * np.sqrt(T)
                call_price_calculated = lambda volatility: S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - call_price
                try:
                    call_imp_volatility = newton(lambda volatility: S * np.exp(-q * T) * norm.cdf((np.log(S / K) + (r - q + (volatility ** 2) / 2) * T) / (volatility * np.sqrt(T))) - K * np.exp(-r * T) * norm.cdf((np.log(S / K) + (r - q + (volatility ** 2) / 2) * T) / (volatility * np.sqrt(T)) - volatility * np.sqrt(T)) - call_price, 0.2)
                except RuntimeError:
                    call_imp_volatility = 0.005  

                call_delta = lambda S, K, T, r, q, call_imp_volatility: si.norm.cdf((np.log(S/K) + (r - q + 0.5*call_imp_volatility**2)*T)/(call_imp_volatility*np.sqrt(T)))*np.exp(-q*T)
                cd = call_delta(S, K, T, r, q, call_imp_volatility)
                return cd

            def put_calculation(S, K, T, r, q, put_price):
                d1 = (np.log(S / K) + (r - q + (0.2 ** 2) / 2) * T) / (0.2 * np.sqrt(T))
                d2 = d1 - 0.2 * np.sqrt(T)
                try:
                    put_imp_volatility = newton(lambda volatility: K * np.exp(-r * T) * norm.cdf(-(np.log(S / K) + (r - q + (volatility ** 2) / 2) * T) / (volatility * np.sqrt(T))) - S * np.exp(-q * T) * norm.cdf(-(np.log(S / K) + (r - q + (volatility ** 2) / 2) * T) / (volatility * np.sqrt(T)) - volatility * np.sqrt(T)) - put_price, 0.2)
                except RuntimeError:
                    put_imp_volatility = 0.005  

                put_delta = lambda S, K, T, r, q, put_imp_volatility: si.norm.cdf(-((np.log(S/K) + (r - q + 0.5*put_imp_volatility**2)*T)/(put_imp_volatility*np.sqrt(T))))*np.exp(-q*T) + np.exp(-q*T) - 1
                pd = -put_delta(S, K, T, r, q, put_imp_volatility)
                return pd

        call_delta = 0
        put_delta = 0

        stock_url_response = requests.get(stock_url, params={'symbols': ticker}, headers=headers)
        stock_quotes = stock_url_response.json()
        
        dividend = yf.Ticker(ticker)
        if len(dividend.dividends) > 0:
            dividend_rate = dividend.dividends[-1]
        else:
            dividend_rate = 0
     
        shortterm_symbol = '^IRX'
        shortterm_data = yf.download(shortterm_symbol)
        shortterm_rate = shortterm_data['Close'].iloc[-1]
        print("")

        S = stock_quotes['quotes']['quote']['last']
        r = shortterm_rate / 100
        q = dividend_rate / 100

        if C is not None:
            display(HTML('<b>Call Delta Calculation</b>'))
            expiration = datetime.strptime(get_call['quotes']['quote']['expiration_date'], "%Y-%m-%d")
            expiration = expiration.replace(hour=16, minute=0)
            now = datetime.now()
            time_to_expiration_seconds = (expiration - now).total_seconds()

            T = time_to_expiration_seconds / (365.25 * 24 * 3600)
            K = get_call['quotes']['quote']['strike']
    
            call_bid = get_call['quotes']['quote']['bid']
            call_ask = get_call['quotes']['quote']['ask']
            call_price = call_bid + ((call_ask - call_bid)/2)
            call_delta = call_calculation(S, K, T, r, q, call_price)
            print(f"Option Premium: {call_price}")

            for pos in positions['positions']['position']:
                if len(pos['symbol']) >= 5 and 'C' in pos['symbol'][5:]:
                    call_multiplier = abs(pos['quantity'])

            adjusted_calldelta = call_delta * call_multiplier

            print(f"Underlying Price: {S}")
            print(f"Strike: {K}")
            print(f"Time to Expiry: {T}")
            print(f"Interest Rate: {r}")
            print(f"Dividend Yield: {q}")
            print(f"Multiplier: {call_multiplier}")
            print(f"Call Delta: {adjusted_calldelta}")
            print("")

        if P is not None:
            display(HTML('<b>Put Delta Calculation</b>'))
            expiration = datetime.strptime(get_put['quotes']['quote']['expiration_date'], "%Y-%m-%d")
            expiration = expiration.replace(hour=16, minute=0)
            now = datetime.now()
            time_to_expiration_seconds = (expiration - now).total_seconds()
            T = time_to_expiration_seconds / (365.25 * 24 * 3600)
            K = get_put['quotes']['quote']['strike']
    
            put_bid = get_put['quotes']['quote']['bid']
            put_ask = get_put['quotes']['quote']['ask']
            put_price = put_bid + ((put_ask - put_bid)/2)
            put_delta = put_calculation(S, K, T, r, q, put_price)
            print(f"Option Premium: {put_price}")

            for pos in positions['positions']['position']:
                if len(pos['symbol']) >= 5 and 'P' in pos['symbol'][5:]:
                    put_multiplier = abs(pos['quantity'])

            adjusted_putdelta = put_delta * put_multiplier

            print(f"Underlying Price: {S}")
            print(f"Strike: {K}")
            print(f"Time to Expiry: {T}")
            print(f"Interest Rate: {r}")
            print(f"Dividend Yield: {q}")
            print(f"Multiplier: {put_multiplier}")
            print(f"Put Delta: {adjusted_putdelta}")
            print("")

        display(HTML('<b>Portfolio Delta</b>'))
        
        net_delta = int((adjusted_calldelta + adjusted_putdelta) * 100)
        
        print(f"Net Delta: {net_delta}")
        print("")
        
        target_hedge = -net_delta
        print(f"Target Hedge: {target_hedge} Shares {ticker}")
        print("")

        positions_rep = requests.get(get_positions, params={}, headers=headers)
        upd_positions = positions_rep.json()

        existing = 0
        
        for pos in upd_positions['positions']['position']:
            if len(pos['symbol']) <= 4:
                existing = pos['quantity']

        print(f"Existing Underlying Position: {existing}")

        trade1_type = None
        trade1_qty = 0
        trade2_type = None
        trade2_qty = 0
        trades_required = 0

        if existing != target_hedge:

            if existing < 0:  
                if target_hedge <= existing:
                    trade1_type = 'sell_short'
                    trade1_qty = abs(target_hedge - existing)
                    trades_required = 1
                else:
                    trade1_type = 'buy_to_cover'
                    trade1_qty = min(abs(existing), abs(target_hedge - existing))
                    trades_required = 1
                    if target_hedge > 0:
                        trade2_type = 'buy'
                        trade2_qty = target_hedge
                        trades_required = 2
            elif existing > 0:  
                if target_hedge >= existing:
                    trade1_type = 'buy'
                    trade1_qty = target_hedge - existing
                    trades_required = 1
                else:
                    trade1_type = 'sell'
                    trade1_qty = min(existing, abs(existing - target_hedge))
                    trades_required = 1
                    if target_hedge < 0:
                        trade2_type = 'sell_short'
                        trade2_qty = abs(target_hedge)
                        trades_required = 2
            else: 
                if target_hedge > 0:
                    trade1_type = 'buy'
                    trade1_qty = target_hedge
                    trades_required = 1
                elif target_hedge < 0:
                    trade1_type = 'sell_short'
                    trade1_qty = abs(target_hedge)
                    trades_required = 1

            if trades_required < 2:
                trade2_type = None
                trade2_qty = 0

            print("")
            print(f"Trades Required: {trades_required}")
            print(f"Trade 1 Type: {trade1_type}, Trade 1 Quantity: {trade1_qty}")
            print("")
            
            if trades_required > 1:
                print(f"Trade 2 Type: {trade2_type}, Trade 2 Quantity: {trade2_qty}")
                print("")

        if trade1_type is not None:
            hedge1_order = requests.post(stock_order_url, data={'class': 'equity', 'symbol': ticker, 'side': trade1_type, 'quantity': abs(trade1_qty), 'type': 'market', 'duration': 'day', 'tag': 'my-tag-example-1'}, headers=headers)
            hedge1 = hedge1_order.json()
            print(f"API Response Status: {hedge1_order.status_code}")
            print(json.dumps(hedge1, indent=4, sort_keys=True))
            print("")

        if trade2_type is not None:
            hedge2_order = requests.post(stock_order_url, data={'class': 'equity', 'symbol': ticker, 'side': trade2_type, 'quantity': abs(trade2_qty), 'type': 'market', 'duration': 'day', 'tag': 'my-tag-example-1'}, headers=headers)
            hedge2 = hedge2_order.json()
            print(f"API Response Status: {hedge2_order.status_code}")
            print(json.dumps(hedge2, indent=4, sort_keys=True))
            print("")

    print("")
    print("=" * 141)
    print("")

def CloseOrders():
    display(HTML('<b>CLOSE ORDERS</b>'))
    
    global closecall_id, closeput_id
    
    closecall_id = 0
    closeput_id = 0
    
    positions_response = requests.get(get_positions, params={}, headers=headers)
    positions = positions_response.json()
    print(f"API Response Status: {positions_response.status_code}")
    print("OPEN POSITIONS")
    print("")
    print(json.dumps(positions, indent=4, sort_keys=True))
    print("")

    if positions['positions'] != 'null':
        if not isinstance(positions['positions']['position'], list):
            positions['positions']['position'] = [positions['positions']['position']]

        C = None
        P = None
        
        for pos in positions['positions']['position']:
            if 'C' in pos['symbol'][4:]:
                C = pos['symbol']
            if 'P' in pos['symbol'][4:]:
                P = pos['symbol']

        call_quotes = requests.get(option_quotes,
        params={'symbols': call, 'greeks': 'false'}, headers=headers)
        call_price = call_quotes.json()
        call_bid = call_price['quotes']['quote']['bid']
        call_ask = call_price['quotes']['quote']['ask']
        print(f"Call Bid: {call_bid}, Call Ask: {call_ask}")

        put_quotes = requests.get(option_quotes,
        params={'symbols': put, 'greeks': 'false'}, headers=headers)
        put_price = put_quotes.json()
        put_bid = put_price['quotes']['quote']['bid']
        put_ask = put_price['quotes']['quote']['ask']
        print(f"Put Bid: {put_bid}, Put Ask: {put_ask}")
        print("")

        if C is not None:
            close_call_rep = requests.post(option_order_url,
                data={'class': 'option', 'symbol': ticker, 'option_symbol': C, 'side': 'sell_to_close', 'quantity': order_size, 'type': 'limit', 'duration': 'day', 'price': call_ask, 'tag': 'my-tag-example-1'},
                headers=headers
            )
            close_call = close_call_rep.json()
            closecall_id = close_call['order']['id']
            print(f'Close Call ID: {closecall_id}')
            print(f"API Response Status: {close_call_rep.status_code}")
            print(json.dumps(close_call, indent=4, sort_keys=True))
            print("")
        if P is not None:
            close_put_rep = requests.post(option_order_url,
                data={'class': 'option', 'symbol': ticker, 'option_symbol': P, 'side': 'sell_to_close', 'quantity': order_size, 'type': 'limit', 'duration': 'day', 'price': put_ask, 'tag': 'my-tag-example-1'},
                headers=headers
            )
            close_put = close_put_rep.json()
            closeput_id = close_put['order']['id']
            print(f'Close Put ID: {closeput_id}')
            print(f"API Response Status: {close_put_rep.status_code}")
            print(json.dumps(close_put, indent=4, sort_keys=True))
            print("")
            
        for pos in positions['positions']['position']:
            if len(pos['symbol']) <= 5:
                stock = pos['quantity']
            else:
                stock = 0

            if stock != 0:
                if stock >= 0:
                    side = 'sell'
                else:
                    side = 'buy_to_cover'

                close_stock_rep = requests.post(stock_order_url,
                    data={'class': 'equity', 'symbol': ticker, 'side': side, 'quantity': abs(stock), 'type': 'market', 'duration': 'day', 'tag': 'my-tag-example-1'},
                    headers=headers
                )
                close_stock = close_stock_rep.json()
                print(f"API Response Status: {close_stock_rep.status_code}")
                print(json.dumps(close_stock, indent=4, sort_keys=True))
                print("")

    print("")
    print("=" * 141)
    print("")

def Liquidate():
    display(HTML('<b>LIQUIDATE</b>'))
    
    positions_response = requests.get(get_positions, params={}, headers=headers)
    positions = positions_response.json()
    print(f"API Response Status: {positions_response.status_code}")
    print("OPEN POSITIONS")
    print("")
    print(json.dumps(positions, indent=4, sort_keys=True))
    print("")

    if positions['positions'] != 'null':
        if not isinstance(positions['positions']['position'], list):
            positions['positions']['position'] = [positions['positions']['position']]

        call_open = False
        put_open = False
        
        for pos in positions['positions']['position']:
            if 'C' in pos['symbol'][4:]:
                call_open = True
            if 'P' in pos['symbol'][4:]:
                put_open = True

        liquidate_call = '{}accounts/{}/orders/{}'.format(api_base_url, account_id, closecall_id)
        liquidate_put = '{}accounts/{}/orders/{}'.format(api_base_url, account_id, closeput_id)
        
        call_quotes = requests.get(option_quotes,
        params={'symbols': call, 'greeks': 'false'}, headers=headers)
        call_price = call_quotes.json()
        call_bid = call_price['quotes']['quote']['bid']
        call_ask = call_price['quotes']['quote']['ask']
        print(f"Call Bid: {call_bid}, Call Ask: {call_ask}")

        put_quotes = requests.get(option_quotes,
        params={'symbols': put, 'greeks': 'false'}, headers=headers)
        put_price = put_quotes.json()
        put_bid = put_price['quotes']['quote']['bid']
        put_ask = put_price['quotes']['quote']['ask']
        print(f"Put Bid: {put_bid}, Put Ask: {put_ask}")
        print("")
        
        if call_open:
            liq_call_rep = requests.put(liquidate_call, data={'type': 'limit', 'duration': '', 'price': call_bid}, headers=headers)
            liq_call = liq_call_rep.json()
            print(f"API Response Status: {liq_call_rep.status_code}")
            print(json.dumps(liq_call, indent=4, sort_keys=True))
            print("")
        if put_open:
            liq_put_rep = requests.put(liquidate_put, data={'type': 'limit', 'duration': '', 'price': put_bid}, headers=headers)
            liq_put = liq_put_rep.json()
            print(f"API Response Status: {liq_put_rep.status_code}")
            print(json.dumps(liq_put, indent=4, sort_keys=True))
            print("")

    print("")
    print("=" * 141)
    print("")

def StopScript():
    display(HTML('<b>STOP ALGORITHM</b>'))

    global running
    global eod_equity
    global pl
    global retrn

    equity_endrep = requests.get(get_balances, params={}, headers=headers)
    equity_end = equity_endrep.json()
    eod_equity = equity_end['balances']['total_equity']
    pl = eod_equity - equity
    retrn = ((eod_equity - equity) / equity) * 100
    
    print(f"DAILY STATISTICS {current_date}:")
    print(f"Starting Bankroll: {equity}")
    print(f"Ending Bankroll: {eod_equity}")
    print(f"P/L: {round(pl, 2)}")
    print(f"Return: {round(retrn, 2)}%")
    print("")

    running = False

OpenOrders()
schedule.every(status).minutes.do(UserAccount)
schedule.every(hedge).seconds.do(DeltaHedge) 
schedule.every().day.at(close).do(CloseOrders)
schedule.every().day.at(liquidate).do(Liquidate)
schedule.every().day.at(stop).do(StopScript)

running = True
while running:
    schedule.run_pending()
    time.sleep(1)

