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

strategy = 'Market Making'
ticker = 'RH'
latency = 2 # seconds 
frequency = 30 # minutes (delta hedging freq)
leverage = .25 # as % of underlying needed for margin (the higher the value, the lower the order size)

# specify option DTE range 
exp_start = 4 # days
exp_stop = 12 # days

# stop hedging
hour = 13 # hour
minute = 59 # minute - should be 1 before stop

# stop algo
stop_algo = "14:00" # military time

date = date.today()
api_base_url = 'https://sandbox.tradier.com/v1/' # Live: 'https://api.tradier.com/v1/'
account_id = 'ID HERE'
access_tkn = 'ACCESS TOKEN HERE'
headers ={'Authorization': 'Bearer {}'.format(access_tkn), 'Accept': 'application/json'}

get_balances = '{}accounts/{}/balances'.format(api_base_url, account_id)
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
balances_rep = requests.get(get_balances, params={}, headers=headers)
balances = balances_rep.json()
equity = balances['balances']['total_equity']
replace_call = True
replace_put = True

print("")
display(HTML('<b>Option Market Making</b>'))
print(f'Date: {date}')
print(f'Bankroll: {equity}')
print(f'Underlying: {ticker}')
print(f'Latency: {latency} Seconds')
print("")

print("")
print("=" * 141)
print("")

def sizing():
    response = requests.get(get_balances, params={}, headers=headers)
    json_response = response.json()
    bankroll = json_response['balances']['total_equity']
    size = bankroll // (leverage * (underlying(ticker) * 100))
    return int(size / 2)

def positions():
    response = requests.get(get_positions, params={}, headers=headers)
    json_response = response.json()
    return json_response
    
def bid(symbol):
    price_rep = requests.get(stock_url, params={'symbols': symbol, 'greeks': 'false'}, headers=headers)
    price = price_rep.json()
    bid = price['quotes']['quote']['bid']
    discount = round((bid + (.01 * bid)), 2)
    return discount

def ask(symbol):
    price_rep = requests.get(stock_url, params={'symbols': symbol, 'greeks': 'false'}, headers=headers)
    price = price_rep.json()
    ask = price['quotes']['quote']['ask']
    discount = round((ask - (.03 * ask)), 2)
    return ask

def underlying(ticker):
    stock_url = '{}markets/quotes'.format(api_base_url)
    response = requests.get(stock_url, params={'symbols': ticker}, headers=headers)
    json_response = response.json()
    last = json_response['quotes']['quote']['last']
    return last

def get_order(arg, input):
    response = requests.get(input, params={'includeTags': 'true'}, headers=headers)
    json_response = response.json()
    return json_response

def option_order(symbol, side, qty, price):
    global closecall_id, closeput_id
    response = requests.post(option_order_url,
        data={'class': 'option', 'symbol': ticker, 'option_symbol': symbol, 'side': side, 'quantity': qty, 
              'type': 'limit', 'duration': 'day', 'price': price, 'tag': 'my-tag-example-1'}, headers=headers)
    json_response = response.json()
    if 'C' in symbol:
        closecall_id = json_response['order']['id']
    if 'P' in symbol:
        closeput_id = json_response['order']['id']
    print(response.status_code)
    print(json_response)
    print("")
    
def Order():
    global call, put, call_id, put_id

    stock_url_response = requests.get(stock_url, params={'symbols': ticker}, headers=headers)
    stock_quotes = stock_url_response.json()
    last_price = stock_quotes['quotes']['quote']['last']
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

    if ticker == 'SPY' or 'ILMN':
        k = int(last_price)
    else:
        interval = 2.50 
        k = round(last_price / interval) * interval

    today = datetime.now().date()
    start_range = today + timedelta(days=exp_start) 
    end_range = today + timedelta(days=exp_stop) 
    
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
            
    call_pricerep = requests.get(option_quotes, params={'symbols': call}, headers=headers)
    call_price = call_pricerep.json()
    put_pricerep = requests.get(option_quotes, params={'symbols': put}, headers=headers)
    put_price = put_pricerep.json()

    positions_response = requests.get(get_positions, params={}, headers=headers)
    positions = positions_response.json()

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
                    
    # if call_filled == False
    if replace_call:
        try:
            call_order_req = requests.post(option_order_url,
                data={'class': 'option', 'symbol': ticker, 'option_symbol': call, 'side': 'buy_to_open', 'quantity': sizing(),
                      'type': 'limit', 'duration': 'day', 'price': bid(call), 'tag': 'my-tag-example-1'}, headers=headers)
            call_order = call_order_req.json()
            call_id = call_order['order']['id']
            print(call_order_req.status_code)
            print(call_order)
            print("")
        except Exception as e:
            print('Order Rejected')

    # if put_filled == False:
    if replace_put:
        try:
            put_order_req = requests.post(option_order_url,
                data={'class': 'option', 'symbol': ticker, 'option_symbol': put, 'side': 'buy_to_open', 'quantity': sizing(),
                      'type': 'limit', 'duration': 'day', 'price': bid(put), 'tag': 'my-tag-example-1'}, headers=headers)
            put_order = put_order_req.json()
            put_id = put_id = put_order['order']['id']
            print(put_order_req.status_code)
            print(put_order)
            print("")
        except Exception as e:
            print('Order Rejected')

def Modify(price, modify):
    response = requests.put(modify, data={'type': 'limit', 'duration': '', 'price': price}, headers=headers)
    json_response = response.json()
    print(response.status_code)
    print(json_response)
    print("")

def Update():
    global modifycall, modifyput, modifyclosecall, modifycloseput, replace_call, replace_put
    replace_call = False
    replace_put = False
    modifycall = '{}accounts/{}/orders/{}'.format(api_base_url, account_id, call_id)
    modifyput = '{}accounts/{}/orders/{}'.format(api_base_url, account_id, put_id)
    try:
        modifyclosecall = '{}accounts/{}/orders/{}'.format(api_base_url, account_id, closecall_id)
    except Exception as e:
        print('')
    try:
        modifycloseput = '{}accounts/{}/orders/{}'.format(api_base_url, account_id, closeput_id)
    except Exception as e:
        print('')
    
    if get_order(call, modifycall)['order']['status'] == 'open':
        Modify(bid(call), modifycall)
    else:
        try:
            if get_order(call, modifyclosecall)['order']['status'] != 'open':
                option_order(call, 'sell_to_close', get_order(call, modifycall)['order']['quantity'], ask(call))
                replace_call = True
                Order()
        except Exception as e:
                option_order(call, 'sell_to_close', get_order(call, modifycall)['order']['quantity'], ask(call))
                replace_call = True
                Order()
    
    if get_order(put, modifyput)['order']['status'] == 'open':
        Modify(bid(put), modifyput)
    else:
        try:
            if get_order(put, modifycloseput)['order']['status'] != 'open':
                option_order(put, 'sell_to_close', get_order(put, modifyput)['order']['quantity'], ask(put))
                replace_put = True
                Order()
        except Exception as e:
                option_order(put, 'sell_to_close', get_order(put, modifyput)['order']['quantity'], ask(put))
                replace_put = True
                Order()

def Hedge():
    time = datetime.now()

    if time.hour >= hour and time.minute >= minute:
        return

    display(HTML('<b>DELTA HEDGE</b>'))
    
    positions_response = requests.get(get_positions, params={}, headers=headers)
    positions = positions_response.json()
    print(positions_response.status_code)
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
        try:
            dividend_rate = dividend.dividends[-1]
        except Exception as e:
            dividend_rate = 0
        shortterm_symbol = '^IRX'
        shortterm_data = yf.download(shortterm_symbol)
        shortterm_rate = shortterm_data['Close'].iloc[-1]

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

            call_delta = -call_delta

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

            put_delta = -put_delta

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
            print(hedge1_order.status_code)
            print(json.dumps(hedge1, indent=4, sort_keys=True))
            print("")

        if trade2_type is not None:
            hedge2_order = requests.post(stock_order_url, data={'class': 'equity', 'symbol': ticker, 'side': trade2_type, 'quantity': abs(trade2_qty), 'type': 'market', 'duration': 'day', 'tag': 'my-tag-example-1'}, headers=headers)
            hedge2 = hedge2_order.json()
            print(hedge2_order.status_code)
            print(json.dumps(hedge2, indent=4, sort_keys=True))
            print("")

def Stop():
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
    current_date = date.today()
    
    print(f"DAILY STATISTICS {current_date}:")
    print(f"Starting Bankroll: {equity}")
    print(f"Ending Bankroll: {eod_equity}")
    print(f"P/L: {round(pl, 2)}")
    print(f"Return: {round(retrn, 2)}%")
    print("")

    print("")
    print("=" * 141)
    print("")
    
    running = False

Order()
schedule.every(latency).seconds.do(Update) 
schedule.every(frequency).minutes.do(Hedge)
schedule.every().day.at(stop_algo).do(Stop)

running = True
while running:
    schedule.run_pending()
    time.sleep(1)

