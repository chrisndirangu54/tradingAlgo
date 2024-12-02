import numpy as np
import pandas as pd
from keras.layers import LSTM
from collections import deque
from sklearn.model_selection import RandomizedSearchCV
import optuna
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.callbacks import EarlyStopping
from a3c import A3CAgent
from textblob import TextBlob
from transformers import pipeline
import re
import logging
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import yfinance as yf
from keras.optimizers import Adam, AdamW, RMSprop
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


def online_random_forest_forecast(data):
    """Use HistGradientBoosting for online learning (tree-based)."""
    X = np.arange(len(data)).reshape(-1, 1)
    Y = np.array(data)

    model = HistGradientBoostingRegressor(max_iter=1, warm_start=True)
    model.fit(X, Y)  # Train on available data

    # Predict for the next time step
    return model.predict([[len(data)]])[0]

def feature_importance_based_feature_engineering(data):
    """Generate new features based on RandomForest feature importance."""
    X = np.arange(len(data)).reshape(-1, 1)
    Y = np.array(data)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, Y)

    # Get feature importance and create new features based on them
    importances = model.feature_importances_
    # Example: Create a moving average feature (this could be done dynamically)
    moving_avg = pd.Series(data).rolling(window=5).mean()

    return moving_avg




from sklearn.linear_model import LogisticRegression

def ensemble_models(models, data):
    """Combine predictions from multiple models using a meta-model."""
    predictions = [model.forecast(data) for model in models]
    meta_model = LogisticRegression()
    meta_model.fit(np.array(predictions).T, data[-len(predictions[0]):])  # Train on the model predictions
    return meta_model.predict(np.array(predictions).T)

def adjust_stop_loss(self, volatility):
    """Adjust stop loss based on volatility."""
    # Example: If volatility increases, decrease stop-loss threshold
    self.stop_loss = max(0.01, 0.05 - (volatility * 0.1))  # Simplified, could use more complex formula

# Advanced Hyperparameter Tuning
def advanced_hyperparameter_tuning(model, params, data):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(model, param_grid=params, cv=TimeSeriesSplit())
    grid_search.fit(data)
    return grid_search.best_params_

class RLTradingAgent:
    def __init__((self, state_size, action_size, optimizer='adam', stop_loss=0.05, 
                 max_drawdown=0.2, initial_balance=100000, transaction_cost=0.001, 
                 slippage=0.001, learning_rate=0.001, gamma=0.95, epsilon_decay=0.995
):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_balance = initial_balance
        self.drawdown = 0
        self.stop_loss = stop_loss
        self.max_drawdown = max_drawdown
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self._set_optimizer(optimizer)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.positions = 0  # or whatever the initial value should be
        self.learning_rate = learning_rate
        self.volatility_model = GARCH(p=1, q=1)  # Example: use a GARCH model for volatility

    
    def _set_optimizer(self, optimizer):
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate=self.learning_rate)
        elif optimizer == 'adamw':
            self.optimizer = AdamW(learning_rate=self.learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(learning_rate=self.learning_rate)


    # Initialize DQN
    self.dqn = self._build_dqn_agent()
    
    # A3C is more complex and usually involves multiple workers
    # Here, we'll just set up the model for A3C
    self.a3c_model = self._build_a3c_model()
       
    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(1, self.state_size), activation="relu"))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def _build_dqn_agent(self):
        model = self._build_model()
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = EpsGreedyQPolicy()
        dqn = DQNAgent(model=model, nb_actions=self.action_size, memory=memory, 
                       nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        return dqn

    def _build_a3c_model(self):
        # A3C model with shared layers for both actor and critic
        input_state = Input(shape=(self.state_size,))
        
        # Shared layers
        shared = Dense(64, activation="relu")(input_state)
        shared = Dense(64, activation="relu")(shared)
        
        # Actor (Policy) Output
        policy = Dense(self.action_size, activation="softmax")(shared)  # For discrete action space
        
        # Critic (Value) Output
        value = Dense(1)(shared)  # Linear activation for value function
        
        # Combine into a model
        model = Model(inputs=input_state, outputs=[policy, value])
        
        # Different losses for actor and critic
        def combined_loss(y_true, y_pred):
            # Assuming y_true is [action, _]
            action_taken, value_target = y_true[:, :self.action_size], y_true[:, self.action_size:]
            
            # Policy loss (actor)
            policy_loss = K.mean(-K.log(K.clip(K.sum(policy * K.one_hot(K.cast(K.flatten(action_taken), 'int32'), self.action_size), axis=-1), K.epsilon(), 1.0)))
            
            # Value loss (critic)
            value_loss = K.mean(K.square(value_target - value))
            
            return policy_loss + value_loss
    
        model.compile(optimizer=self.optimizer, loss=combined_loss)
        return model

    def _build_incremental_model(self):
        # A model capable of incremental learning
        from skmultiflow.trees import HoeffdingTreeRegressor
        return HoeffdingTreeRegressor()

    def learn_online(self, X, y):
        self.incremental_model.learn_one(X, y)
        # Update your DQN or policy network with the latest predictions from the incremental model

    def act(self, state):
        # Use the incrementally updated model for state prediction or feature extraction before decision
        prediction = self.incremental_model.predict_one(state)
        # Now use this prediction in your DQN decision process
        return self.dqn.forward([prediction])[0]

    def update_policy(self, state, action, reward, next_state, done):
        # Here you would update your DQN or A3C model with the new experience,
        # but with less emphasis on retraining everything from scratch
        self.remember(state, action, reward, next_state, done)
        self.replay(batch_size=32)  # or whatever your batch size is


    def act_a3c(self, state):
        """Choose an action using the A3C model."""
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        action_probabilities = self.a3c_model.predict(state)[0]
        return np.random.choice(self.action_size, p=action_probabilities)

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in the DQN agent's memory."""
        self.dqn.memory.append(state, action, reward, next_state, done)

    def remember_a3c(self, state, action, reward, next_state, done):
        """For A3C, we directly update with each step, but for simplicity:
        This method is a placeholder for how A3C would handle a single trajectory."""
        pass  # A3C doesn't use a memory buffer in the same way

    def replay(self, batch_size):
        """Train the DQN model."""
        self.dqn.fit(batch_size=batch_size, nb_epochs=1, verbose=0)

    def replay_a3c(self):
        """A3C generally doesn't use replay; it updates online."""
        pass  # Placeholder for A3C training logic

    def evaluate_trade(self, current_price, predicted_price, action, is_option=False, option_type=None):
        """
        Evaluate the trade outcome based on the current price and predicted price.
        This method can handle both stock and option trades.

        :param current_price: Current market price or stock price
        :param predicted_price: Predicted price from the model or target price for options
        :param action: 0 for buy/long, 1 for sell/short, 2 for hold
        :param is_option: Boolean to indicate if the trade involves options
        :param option_type: 'call' or 'put' for options, None for stocks
        :return: Boolean indicating whether the trade should proceed
        """
        if action == 2:  # Hold
            return True  # No change in balance for hold

        if is_option and option_type:
            # Calculate option value using Black-Scholes model
            time_to_expiry = 1  # Example, you'd want to set this dynamically
            option_value = self.black_scholes(current_price, predicted_price, time_to_expiry, self.risk_free_rate, self.volatility, option_type)
            
            if action == 0:  # Buy a call or put option
                profit_loss = option_value
            elif action == 1:  # Sell a call or put option
                profit_loss = -option_value  # Selling an option involves receiving the premium
        else:  # Handling stock trades
            profit_loss = predicted_price - current_price if action == 0 else current_price - predicted_price
            profit_loss *= self.positions  # Multiply by number of shares or contracts

        # Apply transaction costs and slippage
        profit_loss -= profit_loss * (self.transaction_cost + self.slippage)

        if profit_loss < -self.stop_loss * (current_price * self.positions):  # For options, this might be adjusted based on option value
            print("Stop-loss triggered!")
            return False  # Stop trade

        # Update balance and drawdown
        self.balance += profit_loss
        self.max_balance = max(self.balance, self.max_balance)
        self.drawdown = (self.max_balance - self.balance) / self.max_balance
        
        if self.drawdown > self.max_drawdown:
            print("Maximum drawdown limit reached!")
            return False  # Stop trading

        return True  # Proceed with trade
    def update_volatility(self, price_data):
        self.volatility = self.volatility_model.fit_predict(price_data)

    def black_scholes(self, stock_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type='call'):
        """
        Calculate the price of a European option using the Black-Scholes model.
        """
        # Update volatility before pricing
        self.update_volatility(current_price_history)
        # Now use this updated volatility in Black-Scholes calculation
        d1 = (np.log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        if option_type == 'call':
            return (stock_price * norm.cdf(d1, 0.0, 1.0) - strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2, 0.0, 1.0))
        elif option_type == 'put':
            return (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2, 0.0, 1.0) - stock_price * norm.cdf(-d1, 0.0, 1.0))
        else:
            raise ValueError("Option type must be 'call' or 'put'")

    def kelly_bet_fraction(self, win_prob, win_ratio):
        """
        Calculate the Kelly bet fraction.
        """
        return (win_prob * (win_ratio + 1) - 1) / win_ratio

    def determine_position_size(self, current_price, predicted_price, action):
        """
        Determine the position size using Kelly Criterion.
        """
        if action == 0:  # Buy or Long Option
            win_prob = self.estimate_win_probability(current_price, predicted_price)
            win_ratio = (predicted_price / current_price) - 1
        elif action == 1:  # Sell or Short Option
            win_prob = self.estimate_win_probability(predicted_price, current_price)  # Probability of price falling below prediction
            win_ratio = ((current_price - predicted_price) / current_price)

        fraction = self.kelly_bet_fraction(win_prob, win_ratio)
        # Adjust for practical considerations (like not betting full Kelly due to risk aversion)
        position_size = self.balance * (fraction * 0.5)  # Betting half Kelly for safety
        return position_size

    def estimate_win_probability(self, current_price, predicted_price):
        # Placeholder for model prediction accuracy or some other metric
        # This could be based on historical accuracy or real-time model confidence
        # For simplicity, we'll use the difference between current and predicted price
        price_diff = abs(predicted_price - current_price)
        # Assuming a normal distribution with mean at current price and volatility as standard deviation
        prob = norm.cdf(current_price, loc=current_price, scale=self.volatility * current_price)
        return prob

def manage_overnight_risk(self, current_price, forecasted_price):
    """Manage overnight positions to mitigate risk."""
    if self.positions > 0:  # If we have open positions
    price_difference = current_price - forecasted_price
    if price_difference > self.stop_loss * current_price:
    # Close position if the difference is beyond our stop loss threshold
    self.positions = 0
    self.balance -= self.positions * current_price  # Adjust balance for closing positions
    logging.info("Overnight risk management: Position closed due to price drop.")
    else:
    logging.info("Overnight risk assessed, no action taken.")
            
def adjust_reward_for_risk(reward, position_size, volatility, risk_threshold=0.05):
    """Adjust the reward by penalizing high risk positions."""
    if volatility > risk_threshold:
        reward -= 0.1 * position_size  # Penalize large positions during high volatility
    return reward

# Example usage in the RL agent:
reward = adjust_reward_for_risk(reward, position_size, current_volatility)


    def integrate_sentiment(self, state, sentiment_score):
        """Integrate sentiment score into state."""
        return np.append(state, sentiment_score)

# Example usage for DQN:
state = np.random.rand(1, 3)  # Example state
agent = RLTradingAgent(state_size=3, action_size=2)
action = agent.act(state)
print("DQN Action:", action)

# Example usage for A3C:
action_a3c = agent.act_a3c(state)
print("A3C Action:", action_a3c)
 
# RL-enhanced Backtesting
def backtest_with_rl(data, episodes=50, batch_size=32, initial_balance=10000):
    state_size = 3  # Example state size: [price change, SMA, portfolio balance]
    action_size = 3  # Actions: Buy, Sell, Hold
    agent = RLTradingAgent(state_size=state_size, action_size=action_size, initial_balance=initial_balance)
    
    all_balances = []

    for episode in range(episodes):
        balance = initial_balance
        positions = 0
        state = np.array([0, 0, balance]).reshape(1, state_size)
        episode_balances = [balance]  # Keep track of balance throughout the episode

        for t in range(len(data) - 1):
            # Predict price movement or volatility if needed
            # Here, we're using a very simple prediction, you might replace this with a more complex model
            
            price_change = data[t+1] - data[t]
            mean_price = np.mean(data[max(0, t-5):t])
            
            # Integrate sentiment if available from external sources or as part of state
            state = np.array([price_change, mean_price, balance]).reshape(1, state_size)

            # Choose action based on the current state
            action = agent.act(state)

            # Apply action
            if action == 0:  # Buy
                buy_price_with_costs = data[t] * (1 + agent.transaction_cost + agent.slippage)
                shares_to_buy = int(balance // buy_price_with_costs)  # Ensure integer shares
                if shares_to_buy > 0:
                    positions += shares_to_buy
                    balance -= shares_to_buy * buy_price_with_costs
            elif action == 1:  # Sell
                if positions > 0:
                    sell_price_with_costs = data[t] * (1 - agent.transaction_cost - agent.slippage)
                    balance += positions * sell_price_with_costs
                    positions = 0
            
            # Update state for next iteration
            balance_with_positions = balance + positions * data[t+1]
            next_state = np.array([data[t+1] - data[t], mean_price, balance_with_positions]).reshape(1, state_size)
            
            # Calculate reward
            reward = balance_with_positions - (balance + positions * data[t])
            reward = agent.adjust_reward_for_risk(reward, positions, np.std(data[max(0, t-20):t+1]))
            
            done = t == (len(data) - 2)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Update balance history
            episode_balances.append(balance_with_positions)

            # Check for risk management conditions
            if not agent.evaluate_trade(data[t+1], data[t], action):
                print("Trading stopped due to risk management rules.")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Adjust stop-loss based on recent volatility
            if t > 20:  # Need enough data to calculate volatility
                agent.adjust_stop_loss(np.std(data[max(0, t-20):t+1]))

        # Manage overnight risk at the end of each trading day or episode
        agent.manage_overnight_risk()

        # Log episode results
        all_balances.append(episode_balances)
        logging.info(f"Episode {episode + 1}/{episodes} - Final Balance: ${balance_with_positions:.2f}, Max Drawdown: {max(0, (max(episode_balances) - balance_with_positions) / max(episode_balances)):.2%}")

    return all_balances, balance_with_positions  # Return all balances for further analysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def generate_weighted_trade_signal(historical_data, sentiment_score=None, weights=None):
    """
    Generate a trade signal using weighted combination of indicators.

    :param historical_data: Historical price data up to current time
    :param sentiment_score: Sentiment score from sentiment analysis
    :param weights: Dictionary specifying weights for different signals
    :return: 'buy', 'sell', or 'hold'
    """
    if len(historical_data) < 30:
        return 'hold'
    
    close = pd.Series(historical_data, name='close')
    
    # Calculate indicators
    rsi = RSIIndicator(close, window=14).rsi()
    macd = MACD(close).macd()
    bb = BollingerBands(close, window=20, window_dev=2)
    
    current_price = close[-1]
    ma5 = np.mean(close[-5:])
    
    # Define default weights if not provided
    if weights is None:
        weights = {
            'price_trend': 0.3,
            'technical': 0.5,
            'sentiment': 0.2
        }
    
    # Compute signals
    signals = {
        'price_trend': 0,
        'technical': 0,
        'sentiment': 0
    }
    
    # Price Trend Signal
    if current_price > ma5 and all(close.iloc[-i] <= close.iloc[-i+1] for i in range(1, 4)):
        signals['price_trend'] = 1
    elif current_price < ma5 and all(close.iloc[-i] >= close.iloc[-i+1] for i in range(1, 4)) or current_price / close[-2] < 0.98:
        signals['price_trend'] = -1
    
    # Technical Indicators Signal
    if rsi[-1] < 30 and macd[-1] > macd[-2]:
        signals['technical'] = 1
    elif rsi[-1] > 70 and macd[-1] < macd[-2]:
        signals['technical'] = -1
    elif current_price < bb.bollinger_lband()[-1]:
        signals['technical'] = 1
    
    # Sentiment Signal
    if sentiment_score is not None:
        if sentiment_score > 0.7:
            signals['sentiment'] = 1
        elif sentiment_score < -0.3:
            signals['sentiment'] = -1
    
    # Combine signals with weights
    weighted_sum = sum(signals[signal] * weights[signal] for signal in weights.keys())
    
    if weighted_sum > 0.3:  # Threshold for buying
        return 'buy'
    elif weighted_sum < -0.3:  # Threshold for selling
        return 'sell'
    else:
        return 'hold'

# Example usage:
# signal = generate_weighted_trade_signal(historical_data, sentiment_score)

def train_and_use_rf_model(historical_data, labels, test_data, sentiment_score=None):
    """
    Train a Random Forest model on historical data and use it to predict trading signals for new data.

    :param historical_data: Historical features data with known outcomes
    :param labels: Labels corresponding to buy/sell/hold from historical data
    :param test_data: Features data for which to generate signals
    :param sentiment_score: Sentiment score for new data
    :return: Predicted signal
    """
    # Prepare features
    features = ['rsi', 'macd', 'bb_distance', 'price_trend', 'sentiment']
    X = prepare_data_for_rf(historical_data, labels, features)
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    
    # Predict on test data
    new_features = prepare_data_for_rf([test_data], [sentiment_score], features)
    prediction = rf_model.predict(new_features)
    
    return prediction[0]  # Assuming a single prediction

def prepare_data_for_rf(data, sentiment_scores, features):
    """
    Prepare features for Random Forest model.

    :param data: Price data
    :param sentiment_scores: Sentiment scores for each data point
    :param features: List of features to prepare
    :return: DataFrame with features
    """
    close = pd.Series(data)
    rsi = RSIIndicator(close, window=14).rsi()
    macd = MACD(close).macd()
    bb = BollingerBands(close, window=20, window_dev=2)
    
    X = pd.DataFrame({
        'rsi': rsi,
        'macd': macd,
        'bb_distance': (close - bb.bollinger_mavg()) / bb.bollinger_hband(),
        'price_trend': np.sign(close.diff().rolling(window=3).mean()),
        'sentiment': sentiment_scores
    })
    
    return X[features].fillna(0)

# Example usage:
# historical_data_with_labels = ...  # This should include past decisions and outcomes
# new_data = ...  # Your current market data
# sentiment_score = ...  # Current sentiment score
# signal = train_and_use_rf_model(historical_data_with_labels, new_data, sentiment_score)

# Sharpe Ratio Calculation
def calculate_sharpe(balance_history, risk_free_rate=0.02):
    returns = np.diff(balance_history) / balance_history[:-1]
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

# Monte Carlo Simulation Module
def monte_carlo_simulation(current_price, n_simulations=1000, n_days=252, volatility=None):
    """Simulate future price paths based on historical volatility or provided value."""
    try:
        if volatility is None:
            returns = np.diff(np.log(data.values))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        simulated_paths = []
        for _ in range(n_simulations):
            daily_returns = np.random.normal(loc=0, scale=volatility / np.sqrt(252), size=n_days)
            price_path = [current_price]
            for ret in daily_returns:
                price_path.append(price_path[-1] * np.exp(ret))
            simulated_paths.append(price_path)
        return simulated_paths
    except Exception as e:
        logging.error(f"Error in Monte Carlo simulation: {e}")
        return []


# Main Workflow
if __name__ == "__main__":
    ticker = "AAPL"
    try:
        data = yf.download(ticker, interval="1d", period="1y")['Close']
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        exit()

    # Sentiment Analysis
    news_headline = "Apple announces record-breaking earnings this quarter."
    sentiment_result = SentimentAnalysis.analyze(news_headline)
    sentiment_score = sentiment_result['polarity']  # or transformer_score if preferred

    # Monte Carlo Simulation
    current_price = data.values[-1]
    monte_carlo_results = monte_carlo_simulation(current_price)
    
    # Interactive Plot
    fig = go.Figure()
    for path in monte_carlo_results:
        fig.add_trace(go.Scatter(y=path, mode='lines', line=dict(color='blue', width=0.5), opacity=0.1))
    fig.add_trace(go.Scatter(y=[current_price], mode='markers', marker=dict(color='red', size=8), name='Current Price'))
    fig.update_layout(title="Monte Carlo Simulation of Future Price Paths", xaxis_title="Days", yaxis_title="Price")
    fig.show()

    # Dynamic Model Selection
    best_model = ForecastingModels.dynamic_model_selection(data.values)
    logging.info(f"Best model selected: {best_model}")

    # Backtesting with Metrics
    # Assuming we have historical labels for training the RF model
    labels = np.random.choice(['buy', 'sell', 'hold'], size=len(data))  # Placeholder, should use real data
    signal = train_and_use_rf_model(data.values, labels, data.values[-1], sentiment_score)
    final_balance, balance_history = backtest_with_rl(data.values)  # Changed to use RL backtesting
    
    sharpe_ratio = calculate_sharpe(balance_history)
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # RL Backtesting
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data.values), n_trials=100)

    print("Best hyperparameters:", study.best_params)
    print("Best final balance:", -study.best_value)  # Convert back to positive

    # Now use these best parameters for your agent
    best_params = study.best_params
    best_agent = RLTradingAgent(
        state_size=3, 
        action_size=3, 
        initial_balance=initial_balance,
        learning_rate=best_params['learning_rate'],
        gamma=best_params['gamma'],
        epsilon_decay=best_params['epsilon_decay']
    )

    # Run backtest with the best parameters
    _, final_best_balance = backtest_with_rl(data.values, episodes=100, batch_size=best_params['batch_size'])
    
    print(f"Best Strategy Final Balance: ${final_best_balance:.2f}")
    print(f"Best Strategy Sharpe Ratio: {calculate_sharpe(final_best_balance)}")

