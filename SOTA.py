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
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna

# Set up logging
logging.basicConfig(level=logging.INFO)

class MicrostructureModel:
    def __init__(self, volatility, avg_daily_volume, spread=0.001):
        self.volatility = volatility
        self.avg_daily_volume = avg_daily_volume
        self.spread = spread
        self.eta = 0.002
        self.gamma = 0.0002
        self.latency = 0.0001

    def execution_cost(self, volume, time_horizon, is_option=False, dark_pool=False):
        scale = 100 if is_option else 1
        temp_impact = self.eta * (volume / self.avg_daily_volume) * self.volatility * scale * (0.5 if dark_pool else 1)
        perm_impact = self.gamma * (volume / self.avg_daily_volume) * scale
        return temp_impact + perm_impact + self.spread / 2 + self.latency

class TradingEnv:
    def __init__(self, data_dict, options_data_dict, initial_balance=1000, hft_interval=1):
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
        self.micro_model = MicrostructureModel(0.2, avg_daily_volume=1e7)
        self.sentiment_history = {asset: deque(maxlen=50) for asset in self.assets}

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {asset: {'stock': 0, 'call': {}, 'put': {}, 'variance_swap': 0} for asset in self.assets}
        for asset in self.sentiment_history:
            self.sentiment_history[asset].clear()
        return self._get_state()

    def step(self, action_dict, sentiment_dict, rf_preds, lr_preds):
        current_prices = {asset: self.data_dict[asset][self.current_step] for asset in self.assets}
        next_prices = {asset: self.data_dict[asset][self.current_step + 1] for asset in self.assets}
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
            rf_boost = 1.1 if rf_preds[asset] == 1 else 0.9  # RF direction boost
            lr_risk = 0.8 if lr_preds[asset] == 1 else 1.0  # LR high volatility dampener
            stock_vol = self._kelly_position(current_prices[asset], price_dist, 'stock') * np.sign(stock_vol) * rf_boost * lr_risk
            call_vol = self._kelly_position(current_prices[asset], price_dist, 'call', strike, expiry) * np.sign(call_vol) * rf_boost * lr_risk
            put_vol = self._kelly_position(current_prices[asset], price_dist, 'put', strike, expiry) * np.sign(put_vol) * rf_boost * lr_risk
            var_swap_vol = self._kelly_position(current_prices[asset], vol_dist, 'variance_swap') * np.sign(var_swap_vol) * rf_boost * lr_risk

            asset_reward, _ = self._execute_trade(asset, current_prices[asset], next_prices[asset], stock_vol, call_vol, put_vol, strike, expiry, var_swap_vol)
            reward += asset_reward
        
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
            impact_cost = self.micro_model.execution_cost(volume, self.hft_interval / 86400, dark_pool=dark_pool)
            price_adj = current_price * (1 + impact_cost * np.sign(stock_vol))
            if stock_vol > 0 and self.balance >= volume * price_adj:
                self.positions[asset]['stock'] += volume
                self.balance -= volume * price_adj * (1 + self.transaction_cost + self.slippage)
            elif stock_vol < 0 and self.positions[asset]['stock'] >= volume:
                self.positions[asset]['stock'] -= volume
                self.balance += volume * price_adj * (1 - self.transaction_cost - self.slippage)

        for opt_type, vol in [('call', call_vol), ('put', put_vol)]:
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
        returns = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
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

class RLTradingAgent:
    def __init__(self, state_size=25, action_size=18, initial_balance=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.initial_balance = initial_balance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=100000)
        
        self.stop_loss = 0.01
        self.max_drawdown = 0.15
        
        self.transformer = TimeSeriesTransformerModel(d_model=128, n_heads=16, n_encoder_layers=8, d_ff=512, dropout=0.05).to(self.device)
        self.gat = GATConv(in_channels=state_size, out_channels=64, heads=4).to(self.device)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
        self.volatility_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.assets = ['AAPL', 'TSLA', 'BTC-USD']
        
        self.rl = PPO(
            "MlpPolicy",
            make_vec_env(lambda: TradingEnv({'AAPL': np.zeros(100), 'TSLA': np.zeros(100), 'BTC-USD': np.zeros(100)}, 
                                           {'AAPL': {90: {5: 10}}, 'TSLA': {90: {5: 10}}, 'BTC-USD': {90: {5: 10}}})),
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=1024,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            ent_coef=0.01,
            device=self.device
        )

    def _prepare_features(self, data_dict):
        features_dict = {}
        for asset in self.assets:
            data = data_dict[asset]
            returns = np.diff(np.log(data[-21:])) * 100
            sma_short = np.mean(data[-6:-1])
            sma_long = np.mean(data[-21:-1])
            volatility = self._update_volatility(data)
            features = np.array([returns[-1], sma_short, sma_long, volatility]).reshape(1, -1)
            features_dict[asset] = self.scaler.fit_transform(features) if not hasattr(self.scaler, 'mean_') else self.scaler.transform(features)
        return features_dict

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
        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        if "bullish" in text.lower() or "breakout" in text.lower():
            score *= 1.3
        elif "bearish" in text.lower() or "crash" in text.lower():
            score *= 0.7
        return np.clip(score, -1, 1)

    def _ensemble_predict(self, data_dict):
        rf_preds = {}
        lr_preds = {}
        features_dict = self._prepare_features(data_dict)
        for asset in self.assets:
            rf_preds[asset] = self.rf_model.predict(features_dict[asset])[0]  # 1=up, 0=down
            lr_preds[asset] = self.lr_model.predict(features_dict[asset])[0]  # 1=high vol, 0=low vol
        return rf_preds, lr_preds

    def act(self, data_dict, sentiment_dict=None, order_book_dict=None):
        state = self._encode_state(data_dict, sentiment_dict, order_book_dict)
        rf_preds, lr_preds = self._ensemble_predict(data_dict)
        action, _ = self.rl.predict(state, deterministic=True)
        action = np.clip(action, -1, 1)
        action_dict = {}
        for i, asset in enumerate(self.assets):
            start_idx = i * 6
            action_dict[asset] = action[start_idx:start_idx + 6]
            action_dict[asset][3:5] = (action_dict[asset][3:5] + 1) / 2  # Strike/expiry
        return action_dict, rf_preds, lr_preds

    def train_supervised(self, data_dict):
        for asset in self.assets:
            data = data_dict[asset]
            returns = np.diff(np.log(data)) * 100
            X = pd.DataFrame({
                'lag1': np.roll(returns, 1)[1:],
                'lag5': np.roll(returns, 5)[1:],
                'lag22': np.roll(returns, 22)[1:],
                'sma_short': np.convolve(data, np.ones(5)/5, mode='valid')[:-1],
                'sma_long': np.convolve(data, np.ones(20)/20, mode='valid')[:-1]
            }).dropna()
            y_rf = (np.sign(returns[19:]) + 1) // 2  # 1=up, 0=down
            y_lr = (np.abs(returns[19:]) > np.percentile(np.abs(returns), 75)).astype(int)  # 1=high vol
            X_scaled = self.scaler.fit_transform(X)
            self.rf_model.fit(X_scaled, y_rf)
            self.lr_model.fit(X_scaled, y_lr)

    def tune_hyperparameters(self, data_dict, trials=50):
        def objective(trial):
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20)
            }
            lr_params = {
                'C': trial.suggest_loguniform('lr_C', 1e-5, 1e2)
            }
            ppo_params = {
                'learning_rate': trial.suggest_loguniform('ppo_lr', 1e-5, 1e-3),
                'n_steps': trial.suggest_int('ppo_n_steps', 1024, 4096),
                'batch_size': trial.suggest_int('ppo_batch_size', 256, 1024)
            }
            
            self.rf_model = RandomForestClassifier(**rf_params, random_state=42)
            self.lr_model = LogisticRegression(**lr_params, max_iter=1000, random_state=42)
            self.rl = PPO("MlpPolicy", self.rl.env, **ppo_params, gamma=0.999, gae_lambda=0.95, ent_coef=0.01, device=self.device)
            
            self.train_supervised(data_dict)
            self.rl.learn(total_timesteps=10000)  # Short run for tuning
            results = self.backtest(data_dict, options_data_dict, sentiment_dict_list, days=30)
            return results['final_balance']

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials)
        logging.info(f"Best params: {study.best_params}")
        return study.best_params

    def train(self, data_dict, options_data_dict, sentiment_dict_list, episodes=500, timesteps_per_episode=5000):
        env = TradingEnv(data_dict, options_data_dict)
        self.rl.env = make_vec_env(lambda: env, n_envs=8)
        self.train_supervised(data_dict)
        
        class RiskCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.agent = RLTradingAgent(state_size=25, action_size=18)

            def _on_step(self):
                env = self.training_env.envs[0]
                portfolio_value = sum(env._portfolio_value(asset, env.data_dict[asset][env.current_step]) for asset in env.assets) + env.balance
                drawdown = (env.initial_balance - portfolio_value) / env.initial_balance
                if drawdown > self.agent.max_drawdown or portfolio_value < env.initial_balance * (1 - self.agent.stop_loss):
                    logging.info("Stopping training due to risk limits.")
                    return False
                return True

        sentiment_dict_list = sentiment_dict_list if sentiment_dict_list else [{'AAPL': None, 'TSLA': None, 'BTC-USD': None}] * max(len(data) for data in data_dict.values())
        for episode in range(episodes):
            obs = env.reset()
            for t in range(timesteps_per_episode):
                action, rf_preds, lr_preds = self.act({asset: data[:env.current_step + 1] for asset, data in data_dict.items()}, 
                                                      sentiment_dict_list[env.current_step])
                obs, reward, done, _ = env.step(action, {asset: self._analyze_sentiment(sentiment_dict_list[env.current_step][asset]) 
                                                        for asset in self.assets}, rf_preds, lr_preds)
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
        
        sentiment_dict_list = sentiment_dict_list if sentiment_dict_list else [{'AAPL': None, 'TSLA': None, 'BTC-USD': None}] * max(len(data) for data in data_dict.values())
        for t in range(min(total_steps, env.max_steps)):
            action, rf_preds, lr_preds = self.act({asset: data[:t + 1] for asset, data in data_dict.items()}, sentiment_dict_list[t])
            obs, reward, done, _ = env.step(action, {asset: self._analyze_sentiment(sentiment_dict_list[t][asset]) for asset in env.assets}, rf_preds, lr_preds)
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

# Example usage with backtest
if __name__ == "__main__":
    tickers = ['AAPL', 'TSLA', 'BTC-USD']
    data_dict = {}
    dt = 1 / (252 * 390 * 60)
    for ticker in tickers:
        data = yf.download(ticker, interval="1d", period="2y")['Close'].values
        volatility = np.std(np.diff(np.log(data))) * np.sqrt(252)
        uhf_data = [data[0]]
        for i in range(30 * 390 * 60 - 1):
            dW = np.random.normal(0, np.sqrt(dt))
            uhf_data.append(uhf_data[-1] * np.exp((0.02 - 0.5 * volatility ** 2) * dt + volatility * dW))
        data_dict[ticker] = np.array(uhf_data)
    
    options_data_dict = {
        'AAPL': {data_dict['AAPL'][-1] * 0.9: {5/252: 10}, data_dict['AAPL'][-1]: {5/252: 8}, data_dict['AAPL'][-1] * 1.1: {5/252: 6}},
        'TSLA': {data_dict['TSLA'][-1] * 0.9: {5/252: 10}, data_dict['TSLA'][-1]: {5/252: 8}, data_dict['TSLA'][-1] * 1.1: {5/252: 6}},
        'BTC-USD': {data_dict['BTC-USD'][-1] * 0.9: {5/252: 100}, data_dict['BTC-USD'][-1]: {5/252: 80}, data_dict['BTC-USD'][-1] * 1.1: {5/252: 60}}
    }
    
    sentiment_dict_list = [
        {'AAPL': "Apple AI innovations soar!", 'TSLA': "Tesla breaks production records", 'BTC-USD': "Bitcoin rallies on ETF news"},
        {'AAPL': "Tech stocks under pressure", 'TSLA': "Elon tweets spark volatility", 'BTC-USD': "Crypto market cautious"},
        {'AAPL': "Apple earnings beat estimates", 'TSLA': "Tesla bullish on Cybertruck", 'BTC-USD': "BTC hits new high"}
    ] * (30 * 390 * 60 // 3 + 1)
    sentiment_dict_list = sentiment_dict_list[:30 * 390 * 60]
    
    agent = RLTradingAgent(state_size=25, action_size=18)
    best_params = agent.tune_hyperparameters(data_dict, trials=10)  # Reduced trials for demo
    agent.train(data_dict, options_data_dict, sentiment_dict_list)
    results = agent.backtest(data_dict, options_data_dict, sentiment_dict_list, days=30)
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Annualized Return: {results['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
