import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import yfinance as yf
from transformers import TimeSeriesTransformerModel, pipeline
from torch_geometric.nn import GATConv
from scipy.stats import norm
from collections import deque
import logging
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import optuna
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import shap
from pypfopt import EfficientFrontier
from fredapi import Fred

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMicrostructureModel:
    def __init__(self, volatility, avg_daily_volume, spread=0.001):
        self.volatility = volatility
        self.avg_daily_volume = avg_daily_volume
        self.spread = spread
        self.eta = 0.002
        self.gamma = 0.0002
        self.latency = 0.0001
        self.impact_decay = 0.95
        
    def execution_cost(self, volume, time_horizon, is_option=False, dark_pool=False, market_volatility=1.0):
        scale = 100 if is_option else 1
        vol_adjustment = np.sqrt(market_volatility / 0.2)
        time_pressure = np.sqrt(1 / max(time_horizon, 1e-6))
        temp_impact = (self.eta * vol_adjustment * time_pressure * 
                       (volume / self.avg_daily_volume) * self.volatility * scale * 
                       (0.5 if dark_pool else 1))
        perm_impact = (self.gamma * vol_adjustment * 
                       (volume / self.avg_daily_volume) * scale)
        self.eta *= self.impact_decay
        self.gamma *= self.impact_decay
        return temp_impact + perm_impact + self.spread / 2 + self.latency

class TradingEnv:
    def __init__(self, data_dict, options_data_dict, initial_balance=1000000, hft_interval=1):
        self.data_dict = data_dict
        self.options_data_dict = options_data_dict
        self.assets = list(data_dict.keys())
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {asset: {'stock': 0, 'call': {}, 'put': {}, 'variance_swap': 0} for asset in self.assets}
        self.max_steps = min(len(data) for data in data_dict.values()) - 1
        self.transaction_cost = 0.0001
        self.slippage = 0.00005
        self.risk_free_rate = 0.02
        self.volatility_dict = {asset: 0.2 for asset in self.assets}
        self.hft_interval = hft_interval
        self.micro_model = EnhancedMicrostructureModel(0.2, avg_daily_volume=1e7)
        self.sentiment_history = {asset: deque(maxlen=50) for asset in self.assets}
        self.macro_factors = {}
        self.order_imbalance_history = {asset: deque(maxlen=20) for asset in self.assets}
        self.portfolio_values = deque(maxlen=1000)
        
        self.alpaca_api = tradeapi.REST(
            key_id='YOUR_API_KEY',
            secret_key='YOUR_SECRET_KEY',
            base_url='https://paper-api.alpaca.markets'
        )
        self.fred = Fred(api_key='YOUR_FRED_KEY')
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {asset: {'stock': 0, 'call': {}, 'put': {}, 'variance_swap': 0} for asset in self.assets}
        for asset in self.sentiment_history:
            self.sentiment_history[asset].clear()
        self.order_imbalance_history = {asset: deque(maxlen=20) for asset in self.assets}
        self.portfolio_values.clear()
        self.portfolio_values.append(self.initial_balance)
        return self._get_state()
    
    def _update_macro_factors(self):
        try:
            self.macro_factors = {
                'vix': self.fred.get_series('VIXCLS').iloc[-1] / 100,
                'us10y': self.fred.get_series('DGS10').iloc[-1] / 100,
                'inflation': self.fred.get_series('CPIAUCSL').pct_change().iloc[-1] * 12,
                'unemployment': self.fred.get_series('UNRATE').iloc[-1] / 100
            }
        except Exception as e:
            logger.warning(f"Failed to update macro factors: {str(e)}")
            self.macro_factors = {'vix': 0.2, 'us10y': 0.03, 'inflation': 0.02, 'unemployment': 0.035}
    
    def _get_live_order_book(self, asset):
        try:
            snapshot = self.alpaca_api.get_snapshot(asset)
            bids = np.array([[float(bid.price), float(bid.size)] for bid in snapshot.bids][:10])
            asks = np.array([[float(ask.price), float(ask.size)] for ask in snapshot.asks][:10])
            return np.concatenate((bids, asks))
        except Exception as e:
            logger.warning(f"Failed to get order book for {asset}: {str(e)}")
            mid_price = self.data_dict[asset][self.current_step]
            spread = mid_price * 0.001
            return np.array([[mid_price - spread * (i+1), 1000] for i in range(10)] + 
                           [[mid_price + spread * (i+1), 1000] for i in range(10)])

    def step(self, action_dict, sentiment_dict, rf_preds, lr_preds, ensemble_weights, macro_factors=None):
        self._update_macro_factors()
        current_prices = {}
        next_prices = {}
        for asset in self.assets:
            try:
                current_price = float(self.alpaca_api.get_latest_trade(asset).price)
                current_prices[asset] = current_price
                next_prices[asset] = current_price * (1 + np.random.normal(0, self.volatility_dict[asset]/np.sqrt(252*390)))
            except:
                current_prices[asset] = self.data_dict[asset][self.current_step]
                next_prices[asset] = self.data_dict[asset][self.current_step + 1]
            
            order_book = self._get_live_order_book(asset)
            bid_vol = np.sum(order_book[:10, 1])
            ask_vol = np.sum(order_book[10:, 1])
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
            self.order_imbalance_history[asset].append(imbalance)
        
        for asset in self.assets:
            self.sentiment_history[asset].append(sentiment_dict.get(asset, 0.0))
        
        reward = 0
        for asset in self.assets:
            action = action_dict[asset]
            stock_vol, call_vol, put_vol, strike_idx, expiry_idx, var_swap_vol = action
            strikes = list(self.options_data_dict[asset].keys())
            expiries = list(self.options_data_dict[asset][strikes[0]].keys())
            strike = strikes[int(strike_idx * (len(strikes) - 1))]
            expiry = expiries[int(expiry_idx * (len(expiries) - 1))]

            price_dist, vol_dist = self._monte_carlo_forecast(current_prices[asset], self.volatility_dict[asset])
            order_imbalance = np.mean(self.order_imbalance_history[asset]) if self.order_imbalance_history[asset] else 0
            rf_boost = 1.1 if rf_preds[asset] == 1 else 0.9
            lr_risk = 1 - lr_preds[asset] * 0.5  # Continuous scaling
            ensemble_adj = ensemble_weights[self.assets.index(asset)] * 0.5 + 0.5  # Weighted adjustment
            imbalance_adj = 1 + 0.2 * order_imbalance
            
            signal_strength = rf_boost * lr_risk * ensemble_adj * imbalance_adj
            stock_vol = self._optimized_position(current_prices[asset], price_dist, 'stock', signal_strength) * np.sign(stock_vol)
            call_vol = self._optimized_position(current_prices[asset], price_dist, 'call', signal_strength, strike, expiry) * np.sign(call_vol)
            put_vol = self._optimized_position(current_prices[asset], price_dist, 'put', signal_strength, strike, expiry) * np.sign(put_vol)
            var_swap_vol = self._optimized_position(current_prices[asset], vol_dist, 'variance_swap', signal_strength) * np.sign(var_swap_vol)

            asset_reward, _ = self._execute_trade(asset, current_prices[asset], next_prices[asset], stock_vol, call_vol, put_vol, strike, expiry, var_swap_vol)
            reward += asset_reward
        
        portfolio_value = sum(self._portfolio_value(asset, current_prices[asset]) for asset in self.assets) + self.balance
        self.portfolio_values.append(portfolio_value)
        portfolio_returns = np.diff(list(self.portfolio_values)) / list(self.portfolio_values)[:-1]
        var_95 = np.percentile(portfolio_returns, 5) if len(portfolio_returns) > 0 else 0
        if var_95 < -0.05:
            reward -= 0.5  # VaR penalty
        
        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= self.max_steps - 1
        return next_state, reward, done, {}

    def _get_state(self):
        state = []
        for asset in self.assets:
            price_change = self.data_dict[asset][self.current_step] - self.data_dict[asset][max(0, self.current_step - 1)]
            sma_short = np.mean(self.data_dict[asset][max(0, self.current_step - 5):self.current_step + 1])
            sma_long = np.mean(self.data_dict[asset][max(0, self.current_step - 20):self.current_step + 1])
            stock_value = self.positions[asset]['stock'] * self.data_dict[asset][self.current_step]
            call_value = sum(self._heston_cir_price(self.data_dict[asset][self.current_step], k, t, 'call') * v 
                             for k, t_v in self.positions[asset]['call'].items() for t, v in t_v.items())
            put_value = sum(self._heston_cir_price(self.data_dict[asset][self.current_step], k, t, 'put') * v 
                            for k, t_v in self.positions[asset]['put'].items() for t, v in t_v.items())
            var_swap_value = self.positions[asset]['variance_swap'] * (self.volatility_dict[asset] ** 2 - 0.04)
            sentiment_avg = np.mean(self.sentiment_history[asset]) if self.sentiment_history[asset] else 0.0
            state.extend([price_change, sma_short, sma_long, stock_value, call_value, put_value, self.volatility_dict[asset], sentiment_avg])
        state.append(self.balance)
        return np.array(state, dtype=np.float32)

    def _execute_trade(self, asset, current_price, next_price, stock_vol, call_vol, put_vol, strike, expiry, var_swap_vol):
        portfolio_value_before = self._portfolio_value(asset, current_price)
        
        if stock_vol != 0:
            volume = abs(stock_vol) * 10000
            dark_pool = np.random.rand() < 0.5
            impact_cost = self.micro_model.execution_cost(volume, self.hft_interval / 86400, dark_pool=dark_pool, market_volatility=self.volatility_dict[asset])
            price_adj = current_price * (1 + impact_cost * np.sign(stock_vol))
            if stock_vol > 0 and self.balance >= volume * price_adj:
                self.positions[asset]['stock'] += volume
                self.balance -= volume * price_adj * (1 + self.transaction_cost + self.slippage)
            elif stock_vol < 0 and self.positions[asset]['stock'] >= volume:
                self.positions[asset]['stock'] -= volume
                self.balance += volume * price_adj * (1 - self.transaction_cost - self.slippage)

        # Multi-leg options strategy (straddle)
        if call_vol > 0 and put_vol > 0:
            contracts_call = abs(call_vol) * 100
            contracts_put = abs(put_vol) * 100
            premium_call = self._heston_cir_price(current_price, strike, expiry, 'call')
            premium_put = self._heston_cir_price(current_price, strike, expiry, 'put')
            impact_cost_call = self.micro_model.execution_cost(contracts_call * 100, self.hft_interval / 86400, is_option=True)
            impact_cost_put = self.micro_model.execution_cost(contracts_put * 100, self.hft_interval / 86400, is_option=True)
            price_adj_call = premium_call * (1 + impact_cost_call)
            price_adj_put = premium_put * (1 + impact_cost_put)
            total_cost = (contracts_call * price_adj_call + contracts_put * price_adj_put) * 100
            if self.balance >= total_cost:
                self.positions[asset]['call'].setdefault(strike, {}).setdefault(expiry, 0)
                self.positions[asset]['put'].setdefault(strike, {}).setdefault(expiry, 0)
                self.positions[asset]['call'][strike][expiry] += contracts_call
                self.positions[asset]['put'][strike][expiry] += contracts_put
                self.balance -= total_cost * (1 + self.transaction_cost + self.slippage)
        else:
            for opt_type, vol in [('call', call_vol), ('put', put_vol)]:
                if vol != 0:
                    premium = self._heston_cir_price(current_price, strike, expiry, opt_type)
                    contracts = abs(vol) * 100
                    impact_cost = self.micro_model.execution_cost(contracts * 100, self.hft_interval / 86400, is_option=True)
                    price_adj = premium * (1 + impact_cost * np.sign(vol))
                    if vol > 0 and self.balance >= contracts * price_adj * 100:
                        self.positions[asset][opt_type].setdefault(strike, {}).setdefault(expiry, 0)
                        self.positions[asset][opt_type][strike][expiry] += contracts
                        self.balance -= contracts * price_adj * 100 * (1 + self.transaction_cost + self.slippage)
                    elif vol < 0 and contracts <= self.positions[asset][opt_type].get(strike, {}).get(expiry, 0):
                        self.positions[asset][opt_type][strike][expiry] -= contracts
                        self.balance += contracts * price_adj * 100 * (1 - self.transaction_cost - self.slippage)

        if var_swap_vol != 0:
            notional = abs(var_swap_vol) * 1000000
            var_swap_value = (self.volatility_dict[asset] ** 2 - 0.04) * notional
            if var_swap_vol > 0 and self.balance >= var_swap_value:
                self.positions[asset]['variance_swap'] += notional
                self.balance -= var_swap_value
            elif var_swap_vol < 0 and self.positions[asset]['variance_swap'] >= notional:
                self.positions[asset]['variance_swap'] -= notional
                self.balance += var_swap_value

        for opt_type in ['call', 'put']:
            for strike in list(self.positions[asset][opt_type].keys()):
                for t in list(self.positions[asset][opt_type][strike].keys()):
                    new_t = t - self.hft_interval / 86400
                    self.positions[asset][opt_type][strike][new_t] = self.positions[asset][opt_type][strike].pop(t)
                    if new_t <= 0:
                        payoff = max(0, next_price - strike) if opt_type == 'call' else max(0, strike - next_price)
                        self.balance += payoff * self.positions[asset][opt_type][strike][0] * 100

        portfolio_value_after = self._portfolio_value(asset, next_price)
        returns = (portfolio_value_after - portfolio_value_before) / portfolio_value_before if portfolio_value_before > 0 else 0
        cvar = self._calculate_cvar(returns)
        reward = returns - 0.3 * cvar
        return reward, False

    def _portfolio_value(self, asset, price):
        stock_value = self.positions[asset]['stock'] * price
        options_value = sum(self._heston_cir_price(price, k, t, 'call') * v 
                            for k, t_v in self.positions[asset]['call'].items() for t, v in t_v.items()) + \
                        sum(self._heston_cir_price(price, k, t, 'put') * v 
                            for k, t_v in self.positions[asset]['put'].items() for t, v in t_v.items())
        var_swap_value = self.positions[asset]['variance_swap'] * (self.volatility_dict[asset] ** 2 - 0.04)
        return stock_value + options_value * 100 + var_swap_value

    def _heston_cir_price(self, stock_price, strike_price, time_to_expiry, option_type, kappa=2, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04, cir_k=1, cir_theta=0.02):
        if time_to_expiry <= 0:
            return max(0, stock_price - strike_price) if option_type == 'call' else max(0, strike_price - stock_price)
        dt = time_to_expiry / 252
        vol_path = v0
        rate_path = self.risk_free_rate
        for _ in range(int(252)):
            dW_v = np.random.normal(0, np.sqrt(dt))
            dW_r = np.random.normal(0, np.sqrt(dt))
            vol_path += kappa * (theta - vol_path) * dt + sigma * np.sqrt(max(0, vol_path)) * dW_v
            rate_path += cir_k * (cir_theta - rate_path) * dt + 0.1 * np.sqrt(max(0, rate_path)) * dW_r
        adj_vol = np.sqrt(max(0, vol_path))
        adj_rate = max(0, rate_path)
        d1 = (np.log(stock_price / strike_price) + (adj_rate + 0.5 * adj_vol ** 2) * time_to_expiry) / (adj_vol * np.sqrt(time_to_expiry))
        d2 = d1 - adj_vol * np.sqrt(time_to_expiry)
        if option_type == 'call':
            return stock_price * norm.cdf(d1) - strike_price * np.exp(-adj_rate * time_to_expiry) * norm.cdf(d2)
        return strike_price * np.exp(-adj_rate * time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)

    def _monte_carlo_forecast(self, current_price, volatility, n_simulations=5000, horizon=1/252):
        dt = horizon / n_simulations
        price_paths = np.zeros(n_simulations)
        vol_paths = np.zeros(n_simulations)
        price_paths[0] = current_price
        vol_paths[0] = volatility ** 2
        for t in range(1, n_simulations):
            dW_p = np.random.normal(0, np.sqrt(dt))
            dW_v = np.random.normal(0, np.sqrt(dt))
            price_paths[t] = price_paths[t-1] * np.exp((self.risk_free_rate - 0.5 * vol_paths[t-1]) * dt + np.sqrt(vol_paths[t-1]) * dW_p)
            vol_paths[t] = max(0, vol_paths[t-1] + 2 * (0.04 - vol_paths[t-1]) * dt + 0.3 * np.sqrt(vol_paths[t-1]) * dW_v)
        return price_paths, np.sqrt(vol_paths)

    def _optimized_position(self, current_price, dist, asset_type, signal_strength=1.0, strike=None, expiry=None):
        kelly_pos = self._kelly_position(current_price, dist, asset_type, strike, expiry)
        adjusted_pos = kelly_pos * signal_strength
        total_portfolio_value = sum(self._portfolio_value(asset, self.data_dict[asset][self.current_step]) for asset in self.assets) + self.balance
        if asset_type == 'stock':
            current_allocation = sum(abs(self.positions[asset]['stock']) * self.data_dict[asset][self.current_step] for asset in self.assets) / total_portfolio_value
        else:
            current_allocation = sum(sum(abs(v) * 100 * self._heston_cir_price(self.data_dict[asset][self.current_step], k, t, 'call' if asset_type == 'call' else 'put') 
                                    for k, t_v in self.positions[asset][asset_type].items() for t, v in t_v.items()) for asset in self.assets) / total_portfolio_value
        max_allocation = 0.2
        allocation_ratio = min(max_allocation / (current_allocation + 1e-6), 1.0)
        return adjusted_pos * allocation_ratio

    def _kelly_position(self, current_price, dist, asset_type, strike=None, expiry=None):
        if asset_type == 'variance_swap':
            expected_value = np.mean(dist)
            profit = expected_value - 0.04
            loss = 0.04 - np.percentile(dist, 5)
        else:
            expected_price = np.mean(dist)
            if asset_type == 'stock':
                profit = expected_price - current_price
                loss = current_price - np.percentile(dist, 5)
            else:
                premium = self._heston_cir_price(current_price, strike, expiry, asset_type)
                payoff_dist = [max(0, p - strike) if asset_type == 'call' else max(0, strike - p) for p in dist]
                profit = np.mean(payoff_dist) - premium
                loss = premium - np.percentile(payoff_dist, 5)
        if profit <= 0 or loss <= 0:
            return 0
        win_prob = np.mean(dist > (current_price if asset_type == 'stock' else 0.04 if asset_type == 'variance_swap' else premium))
        kelly_fraction = win_prob - (1 - win_prob) * (loss / profit)
        return max(0, min(3, kelly_fraction))

    def _calculate_cvar(self, return_val, alpha=0.05):
        return -return_val if return_val < 0 else 0

class EnhancedRLTradingAgent:
    def __init__(self, state_size=25, action_size=24, initial_balance=1000000):  # Updated for 4 assets
        self.state_size = state_size
        self.action_size = action_size
        self.initial_balance = initial_balance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=1000000)
        self.stop_loss = 0.01
        self.max_drawdown = 0.15
        
        self.transformer = TimeSeriesTransformerModel(d_model=256, n_heads=16, n_encoder_layers=12, d_ff=1024, dropout=0.1).to(self.device)
        self.gat = GATConv(in_channels=state_size, out_channels=128, heads=8, dropout=0.1).to(self.device)
        self.sentiment_analyzer = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", 
                                           device=0 if torch.cuda.is_available() else -1)
        self.volatility_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01, num_leaves=64, objective='quantile', alpha=0.5)
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=42, class_weight='balanced')
        self.lr_model = LogisticRegression(penalty='elasticnet', solver='saga', C=1.0, l1_ratio=0.5, max_iter=1000, random_state=42, class_weight='balanced')
        self.meta_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.assets = ['AAPL', 'TSLA', 'BTC-USD', 'SPY']
        
        self.portfolio_optimizer = EfficientFrontier(None, weight_bounds=(0, 0.2))
        self.explainer = None
        
        self._init_rl_model()
        
        # WebSocket for live data
        self.stream = Stream('YOUR_API_KEY', 'YOUR_SECRET_KEY', base_url='wss://paper-api.alpaca.markets/stream')
        for asset in self.assets:
            self.stream.subscribe_trades(self._price_handler, asset)
        self.stream.run_in_background()

    def _price_handler(self, msg):
        asset = msg.symbol
        self.data_dict[asset] = np.append(self.data_dict[asset], float(msg.price))[-1000:]

    def _init_rl_model(self):
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])])
        self.rl = PPO("MlpPolicy", make_vec_env(lambda: TradingEnv(
            {'AAPL': np.zeros(100), 'TSLA': np.zeros(100), 'BTC-USD': np.zeros(100), 'SPY': np.zeros(100)},
            {'AAPL': {90: {5: 10}}, 'TSLA': {90: {5: 10}}, 'BTC-USD': {90: {5: 10}}, 'SPY': {90: {5: 10}}}
        )), learning_rate=3e-5, n_steps=4096, batch_size=2048, n_epochs=15, gamma=0.9999, gae_lambda=0.98, 
                     ent_coef=0.02, max_grad_norm=0.8, clip_range=0.3, clip_range_vf=0.3, policy_kwargs=policy_kwargs, device=self.device)

    def explain_decision(self, state):
        if self.explainer is None:
            background = np.random.randn(100, self.state_size)
            self.explainer = shap.TreeExplainer(self.rf_model)
        shap_values = self.explainer.shap_values(state.reshape(1, -1))
        return shap_values

    def _prepare_macro_features(self):
        fred = Fred(api_key='YOUR_FRED_KEY')
        try:
            vix = fred.get_series('VIXCLS').iloc[-1] / 100
            us10y = fred.get_series('DGS10').iloc[-1] / 100
            inflation = fred.get_series('CPIAUCSL').pct_change().iloc[-1]
            unemployment = fred.get_series('UNRATE').iloc[-1] / 100
            return np.array([vix, us10y, inflation, unemployment])
        except:
            return np.array([0.2, 0.03, 0.02, 0.035])

    def _prepare_features(self, data_dict):
        features_dict = {}
        macro_features = self._prepare_macro_features()
        for asset in self.assets:
            data = data_dict[asset]
            returns = np.diff(np.log(data[-21:])) * 100
            sma_short = np.mean(data[-6:-1])
            sma_long = np.mean(data[-21:-1])
            volatility = self._update_volatility(data)
            rsi = self._calculate_rsi(data[-15:])
            macd = self._calculate_macd(data[-26:])
            bollinger = self._calculate_bollinger(data[-21:])
            features = np.array([returns[-1], sma_short, sma_long, volatility, rsi, macd, bollinger, *macro_features]).reshape(1, -1)
            features_dict[asset] = self.scaler.fit_transform(features) if not hasattr(self.scaler, 'mean_') else self.scaler.transform(features)
        return features_dict

    def _calculate_rsi(self, prices, window=14):
        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum()/window
        down = -seed[seed < 0].sum()/window
        rs = up/down
        rsi = 100 - (100/(1+rs))
        for i in range(window, len(deltas)):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            up = (up*(window-1) + upval)/window
            down = (down*(window-1) + downval)/window
            rs = up/down
            rsi = np.append(rsi, 100 - (100/(1+rs)))
        return rsi[-1]/100

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return (macd[-1] - signal_line[-1]) / prices[-1]

    def _calculate_bollinger(self, prices, window=20):
        sma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        upper = sma + 2*std
        lower = sma - 2*std
        return (prices[-1] - lower) / (upper - lower)

    def _update_volatility(self, price_data):
        returns = np.diff(np.log(price_data)) * 100
        X = pd.DataFrame({
            'lag1': np.roll(returns, 1)[1:],
            'lag5': np.roll(returns, 5)[1:],
            'lag22': np.roll(returns, 22)[1:]
        }).fillna(0)
        y = returns[1:]
        self.volatility_model.fit(X, y)
        pred = self.volatility_model.predict(X[-1:].values.reshape(1, -1))
        return abs(pred[0]) / 100

    def _analyze_sentiment(self, text):
        if not text:
            return 0.0
        result = self.sentiment_analyzer(text)[0]
        score = result['score'] if result['label'] in ['POSITIVE', 'positive'] else -result['score']
        if "bullish" in text.lower() or "breakout" in text.lower():
            score *= 1.3
        elif "bearish" in text.lower() or "crash" in text.lower():
            score *= 0.7
        return np.clip(score, -1, 1)

    def _ensemble_predict(self, data_dict):
        rf_preds = {}
        lr_preds = {}
        ensemble_weights = []
        features_dict = self._prepare_features(data_dict)
        meta_features = []
        for asset in self.assets:
            rf_pred = self.rf_model.predict_proba(features_dict[asset])[0, 1]
            lr_pred = self.lr_model.predict_proba(features_dict[asset])[0, 1]
            lgb_pred = self.volatility_model.predict(features_dict[asset])[0]
            meta_features.append([rf_pred, lr_pred, lgb_pred])
            rf_preds[asset] = int(rf_pred > 0.5)
            lr_preds[asset] = lr_pred  # Continuous probability
        meta_X = np.array(meta_features)
        ensemble_weights = self.meta_model.predict_proba(meta_X)[:, 1]
        return rf_preds, lr_preds, ensemble_weights

    def act(self, data_dict, sentiment_dict=None, order_book_dict=None):
        state = self._encode_state(data_dict, sentiment_dict, order_book_dict)
        rf_preds, lr_preds, ensemble_weights = self._ensemble_predict(data_dict)
        action, _ = self.rl.predict(state, deterministic=True)
        action = np.clip(action, -1, 1)
        
        action_dict = {}
        portfolio_weights = {}
        for i, asset in enumerate(self.assets):
            start_idx = i * 6
            raw_action = action[start_idx:start_idx + 6]
            raw_action[3:5] = (raw_action[3:5] + 1) / 2
            portfolio_weights[asset] = {'stock': abs(raw_action[0]), 'options': (abs(raw_action[1]) + abs(raw_action[2])) / 2, 'var_swap': abs(raw_action[5])}
        
        mu = np.array([sum(w.values()) for w in portfolio_weights.values()])
        cov = np.cov(np.vstack([np.diff(np.log(data_dict[asset][-30:])) for asset in self.assets]))
        try:
            self.portfolio_optimizer.expected_returns = mu
            self.portfolio_optimizer.cov_matrix = cov
            weights = self.portfolio_optimizer.max_sharpe()
            weights = {k: v for k, v in zip(self.assets, weights.values())}
        except:
            logger.warning("Portfolio optimization failed, using equal weights")
            weights = {asset: 1/len(self.assets) for asset in self.assets}
        
        for i, asset in enumerate(self.assets):
            start_idx = i * 6
            raw_action = action[start_idx:start_idx + 6]
            scale = weights[asset] * len(self.assets)
            scaled_action = raw_action * scale
            scaled_action[3:5] = (scaled_action[3:5] + 1) / 2
            action_dict[asset] = scaled_action
        
        return action_dict, rf_preds, lr_preds, ensemble_weights

    def _encode_state(self, data_dict, sentiment_dict=None, order_book_dict=None):
        state = []
        for asset in self.assets:
            raw_data = data_dict[asset]
            seq_len = min(100, len(raw_data))
            data_tensor = torch.tensor(raw_data[-seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)
            time_enc = self.transformer(data_tensor).mean(dim=1).detach().cpu().numpy()
            order_book = order_book_dict.get(asset, np.random.randn(10, 2)) if order_book_dict else np.random.randn(10, 2)
            edge_index = torch.tensor([[i, i+1] for i in range(9)], dtype=torch.long).t().to(self.device)
            order_tensor = torch.tensor(order_book, dtype=torch.float32).to(self.device)
            gat_enc = self.gat(order_tensor, edge_index).mean(dim=0).detach().cpu().numpy()
            sentiment = self._analyze_sentiment(sentiment_dict.get(asset, "")) if sentiment_dict else 0.0
            volatility = self._update_volatility(raw_data) if len(raw_data) > 20 else 0.2
            state.extend([time_enc.flatten()[0], gat_enc[0], sentiment, volatility])
        state.append(self.initial_balance)
        return np.array(state, dtype=np.float32)

    def train_supervised(self, data_dict):
        X_all = []
        y_rf_all = []
        y_lr_all = []
        meta_X_all = []
        meta_y_all = []
        
        for asset in self.assets:
            data = data_dict[asset]
            returns = np.diff(np.log(data)) * 100
            X = pd.DataFrame({
                'lag1': np.roll(returns, 1)[1:],
                'lag5': np.roll(returns, 5)[1:],
                'lag22': np.roll(returns, 22)[1:],
                'sma_short': np.convolve(data, np.ones(5)/5, mode='valid')[:-1],
                'sma_long': np.convolve(data, np.ones(20)/20, mode='valid')[:-1],
                'rsi': [self._calculate_rsi(data[i-15:i]) for i in range(15, len(data)-1)],
                'macd': [self._calculate_macd(data[i-26:i]) for i in range(26, len(data)-1)],
                'bollinger': [self._calculate_bollinger(data[i-21:i]) for i in range(21, len(data)-1)],
                'volume_ratio': np.random.random(len(data)-1)[1:],
                **{f'macro_{i}': np.random.random(len(data)-1)[1:] for i in range(4)}
            }).dropna()
            y_rf = (np.sign(returns[22:]) + 1) // 2
            y_lr = (np.abs(returns[22:]) > np.percentile(np.abs(returns), 75)).astype(int)
            X_scaled = self.scaler.fit_transform(X)
            rf_pred = self.rf_model.fit(X_scaled, y_rf).predict_proba(X_scaled)[:, 1]
            lr_pred = self.lr_model.fit(X_scaled, y_lr).predict_proba(X_scaled)[:, 1]
            lgb_pred = self.volatility_model.fit(X_scaled, y_rf).predict(X_scaled)
            meta_X = np.column_stack((rf_pred, lr_pred, lgb_pred))
            X_all.append(X_scaled)
            y_rf_all.append(y_rf)
            y_lr_all.append(y_lr)
            meta_X_all.append(meta_X)
            meta_y_all.append(y_rf)
        
        X_combined = np.vstack(X_all)
        y_rf_combined = np.hstack(y_rf_all)
        y_lr_combined = np.hstack(y_lr_all)
        meta_X_combined = np.vstack(meta_X_all)
        meta_y_combined = np.hstack(meta_y_all)
        
        self.rf_model.fit(X_combined, y_rf_combined)
        self.lr_model.fit(X_combined, y_lr_combined)
        self.volatility_model.fit(X_combined, y_rf_combined)
        self.meta_model.fit(meta_X_combined, meta_y_combined)
        
        self.explainer = shap.TreeExplainer(self.rf_model)
        logger.info("Supervised training completed")

    def tune_hyperparameters(self, data_dict, trials=50):
        def objective(trial):
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_leaf', 1, 10)
            }
            lr_params = {
                'C': trial.suggest_loguniform('lr_C', 1e-3, 10),
                'l1_ratio': trial.suggest_uniform('lr_l1', 0, 1)
            }
            lgb_params = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
                'learning_rate': trial.suggest_loguniform('lgb_lr', 1e-3, 0.1),
                'num_leaves': trial.suggest_int('lgb_leaves', 31, 127)
            }
            xgb_params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('xgb_lr', 1e-3, 0.1)
            }
            ppo_params = {
                'learning_rate': trial.suggest_loguniform('ppo_lr', 1e-5, 1e-3),
                'n_steps': trial.suggest_int('ppo_n_steps', 1024, 8192),
                'batch_size': trial.suggest_int('ppo_batch_size', 256, 2048)
            }
            self.rf_model = RandomForestClassifier(**rf_params, random_state=42)
            self.lr_model = LogisticRegression(**lr_params, solver='saga', max_iter=1000, random_state=42)
            self.volatility_model = lgb.LGBMRegressor(**lgb_params)
            self.meta_model = xgb.XGBClassifier(**xgb_params, random_state=42)
            self.rl = PPO("MlpPolicy", self.rl.env, **ppo_params, gamma=0.9999, device=self.device)
            self.train_supervised(data_dict)
            self.rl.learn(total_timesteps=10000)
            results = self.backtest(data_dict, options_data_dict, sentiment_dict_list, days=30)
            return results['final_balance']
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials)
        logger.info(f"Best params: {study.best_params}")
        return study.best_params

    def train(self, data_dict, options_data_dict, sentiment_dict_list, episodes=50, timesteps_per_episode=5000):
        env = TradingEnv(data_dict, options_data_dict)
        self.rl.env = make_vec_env(lambda: env, n_envs=8)
        self.train_supervised(data_dict)
        
        class RiskCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.agent = EnhancedRLTradingAgent(state_size=25, action_size=24)

            def _on_step(self):
                env = self.training_env.envs[0]
                portfolio_value = sum(env._portfolio_value(asset, env.data_dict[asset][env.current_step]) for asset in env.assets) + env.balance
                drawdown = (env.initial_balance - portfolio_value) / env.initial_balance
                if drawdown > self.agent.max_drawdown or portfolio_value < env.initial_balance * (1 - self.agent.stop_loss):
                    logger.info("Stopping training due to risk limits.")
                    return False
                return True

        sentiment_dict_list = sentiment_dict_list if sentiment_dict_list else [{'AAPL': None, 'TSLA': None, 'BTC-USD': None, 'SPY': None}] * max(len(data) for data in data_dict.values())
        for episode in range(episodes):
            obs = env.reset()
            for t in range(timesteps_per_episode):
                action, rf_preds, lr_preds, ensemble_weights = self.act({asset: data[:env.current_step + 1] for asset, data in data_dict.items()}, 
                                                                       sentiment_dict_list[env.current_step])
                obs, reward, done, _ = env.step(action, {asset: self._analyze_sentiment(sentiment_dict_list[env.current_step][asset]) 
                                                        for asset in self.assets}, rf_preds, lr_preds, ensemble_weights)
                if done:
                    break
        self.rl.learn(total_timesteps=episodes * timesteps_per_episode, callback=RiskCallback())
        return self.rl

    def backtest(self, data_dict, options_data_dict, sentiment_dict_list, initial_balance=1000, days=30):
        env = TradingEnv(data_dict, options_data_dict, initial_balance=initial_balance)
        obs = env.reset()
        portfolio_values = [initial_balance]
        steps_per_day = 390 * 60
        total_steps = days * steps_per_day
        
        sentiment_dict_list = sentiment_dict_list if sentiment_dict_list else [{'AAPL': None, 'TSLA': None, 'BTC-USD': None, 'SPY': None}] * max(len(data) for data in data_dict.values())
        for t in range(min(total_steps, env.max_steps)):
            action, rf_preds, lr_preds, ensemble_weights = self.act({asset: data[:t + 1] for asset, data in data_dict.items()}, sentiment_dict_list[t])
            obs, reward, done, _ = env.step(action, {asset: self._analyze_sentiment(sentiment_dict_list[t][asset]) for asset in env.assets}, rf_preds, lr_preds, ensemble_weights)
            portfolio_value = sum(env._portfolio_value(asset, env.data_dict[asset][t + 1]) for asset in env.assets) + env.balance
            portfolio_values.append(portfolio_value)
            if done:
                break
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        annualized_return = ((portfolio_values[-1] / initial_balance) ** (252 / days) - 1) * 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 390 * 60) if np.std(returns) > 0 else 0
        max_drawdown = max([0] + [(max(portfolio_values[:i+1]) - portfolio_values[i]) / max(portfolio_values[:i+1]) for i in range(len(portfolio_values))]) * 100
        
        return {
            'final_balance': portfolio_values[-1],
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values
        }

class LiveTradingWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.alpaca = tradeapi.REST('YOUR_API_KEY', 'YOUR_SECRET_KEY', base_url='https://paper-api.alpaca.markets')
        self.current_positions = {}
        
    def run_live(self):
        logger.info("Starting live trading session")
        while True:
            try:
                market_data = self._fetch_market_data()
                sentiment_data = self._fetch_sentiment_data()
                self._update_positions()
                actions, rf_preds, lr_preds, ensemble_weights = self.agent.act(market_data['price'], sentiment_data, market_data['order_book'])
                self._execute_trades(actions)
                self._log_performance()
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in live trading loop: {str(e)}")
                time.sleep(300)
    
    def _fetch_market_data(self):
        data = {'price': {}, 'order_book': {}}
        for asset in self.agent.assets:
            try:
                last_trade = self.alpaca.get_latest_trade(asset)
                data['price'][asset] = np.array([float(last_trade.price)])
                snapshot = self.alpaca.get_snapshot(asset)
                bids = np.array([[float(bid.price), float(bid.size)] for bid in snapshot.bids][:10])
                asks = np.array([[float(ask.price), float(ask.size)] for ask in snapshot.asks][:10])
                data['order_book'][asset] = np.concatenate((bids, asks))
            except Exception as e:
                logger.warning(f"Failed to get data for {asset}: {str(e)}")
                data['price'][asset] = np.array([100.0])
                mid = 100.0
                data['order_book'][asset] = np.array([[mid - 0.1*(i+1), 100] for i in range(10)] + 
                                                    [[mid + 0.1*(i+1), 100] for i in range(10)])
        return data
    
    def _fetch_sentiment_data(self):
        # Placeholder for real sentiment API
        return {asset: "Positive market sentiment" for asset in self.agent.assets}
    
    def _update_positions(self):
        positions = self.alpaca.list_positions()
        self.current_positions = {pos.symbol: float(pos.qty) for pos in positions}
    
    def _execute_trades(self, actions):
        for asset, action in actions.items():
            stock_vol, call_vol, put_vol, strike_idx, expiry_idx, var_swap_vol = action
            current_pos = self.current_positions.get(asset, 0)
            target_pos = stock_vol * 100
            
            if target_pos > current_pos:
                qty = target_pos - current_pos
                side = 'buy'
            elif target_pos < current_pos:
                qty = current_pos - target_pos
                side = 'sell'
            else:
                continue
            
            if qty > 0:
                try:
                    self.alpaca.submit_order(
                        symbol=asset, qty=str(qty), side=side, type='limit', time_in_force='gtc',
                        limit_price=str(self._calculate_limit_price(asset, side))
                    )
                    logger.info(f"Submitted {side} order for {qty} shares of {asset}")
                except Exception as e:
                    logger.error(f"Failed to submit order for {asset}: {str(e)}")
            
            # Options straddle (simplified for demo)
            if call_vol > 0 and put_vol > 0:
                try:
                    self.alpaca.submit_order(
                        symbol=asset, qty=str(abs(call_vol) * 100), side='buy', type='limit', time_in_force='gtc',
                        limit_price=str(self._calculate_limit_price(asset, 'buy')), option_type='call', strike=strike_idx, expiration=expiry_idx
                    )
                    self.alpaca.submit_order(
                        symbol=asset, qty=str(abs(put_vol) * 100), side='buy', type='limit', time_in_force='gtc',
                        limit_price=str(self._calculate_limit_price(asset, 'buy')), option_type='put', strike=strike_idx, expiration=expiry_idx
                    )
                    logger.info(f"Submitted straddle for {asset}")
                except Exception as e:
                    logger.error(f"Failed to submit options order for {asset}: {str(e)}")
    
    def _calculate_limit_price(self, asset, side):
        snapshot = self.alpaca.get_snapshot(asset)
        return float(snapshot.asks[0].price) * 0.999 if side == 'buy' else float(snapshot.bids[0].price) * 1.001
    
    def _log_performance(self):
        account = self.alpaca.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        pnl = equity - float(account.last_equity)
        logger.info(f"Portfolio Update - Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | PnL: ${pnl:+,.2f}")

if __name__ == "__main__":
    tickers = ['AAPL', 'TSLA', 'BTC-USD', 'SPY']
    data_dict = {}
    for ticker in tickers:
        data = yf.download(ticker, interval="1d", period="2y")['Close'].values
        volatility = np.std(np.diff(np.log(data))) * np.sqrt(252)
        dt = 1 / (252 * 390 * 60)
        uhf_data = [data[0]]
        for i in range(30 * 390 * 60 - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            uhf_data.append(uhf_data[-1] * np.exp((0.02 - 0.5 * volatility**2) * dt + volatility * dW))
        data_dict[ticker] = np.array(uhf_data)
    
    options_data_dict = {
        'AAPL': {data_dict['AAPL'][-1] * k: {5/252: 10*(1-abs(k-1))} for k in [0.9, 1.0, 1.1]},
        'TSLA': {data_dict['TSLA'][-1] * k: {5/252: 10*(1-abs(k-1))} for k in [0.9, 1.0, 1.1]},
        'BTC-USD': {data_dict['BTC-USD'][-1] * k: {5/252: 100*(1-abs(k-1))} for k in [0.9, 1.0, 1.1]},
        'SPY': {data_dict['SPY'][-1] * k: {5/252: 10*(1-abs(k-1))} for k in [0.9, 1.0, 1.1]}
    }
    
    sentiment_dict_list = [
        {'AAPL': "Apple announces breakthrough in AI technology", 'TSLA': "Tesla recalls affect production targets",
         'BTC-USD': "Bitcoin ETF approval expected soon", 'SPY': "Fed signals potential rate cuts"}
    ] * (30 * 390 * 60)
    
    agent = EnhancedRLTradingAgent(state_size=25, action_size=24)
    agent.train_supervised(data_dict)
    best_params = agent.tune_hyperparameters(data_dict, trials=50)
    trained_model = agent.train(data_dict, options_data_dict, sentiment_dict_list, episodes=50)
    live_trader = LiveTradingWrapper(agent)
    live_trader.run_live()
